"""
LoRA/QLoRA微调训练脚本
使用此脚本对Qwen模型进行角色语言风格微调
"""

import os
import sys
import yaml
import torch
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
import transformers

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from datasets import load_dataset


class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, config_path: str = "config/train_config.yaml"):
        """初始化训练器
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']
        self.lora_config = self.config['lora']
        self.train_config = self.config['training']
        self.data_config = self.config['data']
        self.other_config = self.config['other']
        self.character_config = self.config['character']
        
        # 设置随机种子
        transformers.set_seed(self.other_config['seed'])
        
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.eval_dataset = None
        
    def load_model_and_tokenizer(self):
        """加载模型和分词器"""
        print("="*60)
        print("加载基础模型和分词器")
        print("="*60)
        
        base_model = self.model_config['base_model']
        print(f"基础模型: {base_model}")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=self.model_config['trust_remote_code'],
            padding_side='right',  # 训练时使用右填充
        )
        
        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"✓ 分词器加载完成")
        
        # 配置量化
        quantization_config = None
        if self.model_config['load_in_4bit']:
            print("使用4bit量化（QLoRA）")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif self.model_config['load_in_8bit']:
            print("使用8bit量化")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quantization_config,
            trust_remote_code=self.model_config['trust_remote_code'],
            device_map="auto",
            torch_dtype=torch.float16,
        )
        
        print(f"✓ 模型加载完成")
        
        # 准备模型用于k-bit训练
        if quantization_config:
            self.model = prepare_model_for_kbit_training(self.model)
            print("✓ 模型已准备用于量化训练")
        
        # 配置LoRA
        peft_config = LoraConfig(
            r=self.lora_config['r'],
            lora_alpha=self.lora_config['lora_alpha'],
            target_modules=self.lora_config['target_modules'],
            lora_dropout=self.lora_config['lora_dropout'],
            bias=self.lora_config['bias'],
            task_type=self.lora_config['task_type'],
        )
        
        self.model = get_peft_model(self.model, peft_config)
        
        # 打印可训练参数
        self.model.print_trainable_parameters()
        
    def load_datasets(self):
        """加载数据集"""
        print("\n" + "="*60)
        print("加载训练数据")
        print("="*60)
        
        processed_path = self.data_config['processed_data_path']
        
        # 加载训练集
        train_file = os.path.join(processed_path, 'train.jsonl')
        if not os.path.exists(train_file):
            raise FileNotFoundError(
                f"训练数据不存在: {train_file}\n"
                f"请先运行 python scripts/prepare_data.py 处理数据"
            )
        
        self.train_dataset = load_dataset('json', data_files=train_file)['train']
        print(f"✓ 训练集: {len(self.train_dataset)} 条")
        
        # 加载验证集
        val_file = os.path.join(processed_path, 'validation.jsonl')
        if os.path.exists(val_file):
            self.eval_dataset = load_dataset('json', data_files=val_file)['train']
            print(f"✓ 验证集: {len(self.eval_dataset)} 条")
        else:
            print("⚠ 未找到验证集")
            self.eval_dataset = None
        
        # 处理数据集
        max_length = self.data_config['max_length']
        
        def tokenize_function(examples):
            """分词函数"""
            # 对文本进行分词
            result = self.tokenizer(
                examples['text'],
                truncation=True,
                max_length=max_length,
                padding=False,  # 不在这里padding，使用data collator
                return_tensors=None,
            )
            
            # 添加labels（与input_ids相同，用于language modeling）
            # 需要复制列表而不是引用
            result["labels"] = [ids[:] for ids in result["input_ids"]]
            
            return result
        
        print("\n开始分词...")
        self.train_dataset = self.train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=self.train_dataset.column_names,
            desc="分词训练集"
        )
        
        if self.eval_dataset:
            self.eval_dataset = self.eval_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=self.eval_dataset.column_names,
                desc="分词验证集"
            )
        
        print("✓ 数据集分词完成")
        
    def train(self):
        """开始训练"""
        print("\n" + "="*60)
        print("开始训练")
        print("="*60)
        
        character_name = self.character_config['name']
        output_dir = os.path.join(
            self.train_config['output_dir'],
            f"{character_name}_{self.model_config['base_model'].split('/')[-1]}"
        )
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.train_config['num_train_epochs'],
            per_device_train_batch_size=self.train_config['per_device_train_batch_size'],
            per_device_eval_batch_size=self.train_config['per_device_eval_batch_size'],
            gradient_accumulation_steps=self.train_config['gradient_accumulation_steps'],
            learning_rate=self.train_config['learning_rate'],
            weight_decay=self.train_config['weight_decay'],
            warmup_ratio=self.train_config['warmup_ratio'],
            lr_scheduler_type=self.train_config['lr_scheduler_type'],
            logging_dir=self.other_config['logging_dir'],
            logging_steps=self.train_config['logging_steps'],
            save_steps=self.train_config['save_steps'],
            eval_steps=self.train_config['eval_steps'] if self.eval_dataset else None,
            save_total_limit=self.train_config['save_total_limit'],
            fp16=self.train_config['fp16'],
            bf16=self.train_config['bf16'],
            gradient_checkpointing=self.train_config['gradient_checkpointing'],
            optim=self.train_config['optim'],
            max_grad_norm=self.train_config['max_grad_norm'],
            report_to=self.other_config['report_to'],
            eval_strategy="steps" if self.eval_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=True if self.eval_dataset else False,
            metric_for_best_model="loss" if self.eval_dataset else None,
            greater_is_better=False,
            ddp_find_unused_parameters=False,
            remove_unused_columns=False,
        )
        
        # Data collator - 使用默认的padding
        from transformers import DataCollatorForSeq2Seq
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            pad_to_multiple_of=8,  # 优化性能
        )
        
        # 创建Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
        )
        
        # 从checkpoint恢复训练
        resume_checkpoint = self.train_config.get('resume_from_checkpoint', None)
        
        # 开始训练
        print(f"\n训练配置:")
        print(f"  - 输出目录: {output_dir}")
        print(f"  - 训练轮数: {self.train_config['num_train_epochs']}")
        print(f"  - Batch Size: {self.train_config['per_device_train_batch_size']}")
        print(f"  - 梯度累积步数: {self.train_config['gradient_accumulation_steps']}")
        print(f"  - 有效Batch Size: {self.train_config['per_device_train_batch_size'] * self.train_config['gradient_accumulation_steps']}")
        print(f"  - 学习率: {self.train_config['learning_rate']}")
        print(f"\n开始训练...\n")
        
        trainer.train(resume_from_checkpoint=resume_checkpoint)
        
        # 保存最终模型
        final_model_path = os.path.join(output_dir, "final_model")
        trainer.save_model(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)
        
        print("\n" + "="*60)
        print("✓ 训练完成！")
        print(f"✓ 模型已保存到: {final_model_path}")
        print("="*60)
        
        # 保存配置信息
        config_save_path = os.path.join(final_model_path, "training_config.yaml")
        with open(config_save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, allow_unicode=True)
        print(f"✓ 训练配置已保存到: {config_save_path}")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("角色语言风格微调训练")
    print("="*60 + "\n")
    
    trainer = ModelTrainer()
    trainer.load_model_and_tokenizer()
    trainer.load_datasets()
    trainer.train()


if __name__ == "__main__":
    main()

