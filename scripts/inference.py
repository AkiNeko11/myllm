"""
模型推理脚本
使用训练好的模型进行对话测试
"""

import os
import sys
import yaml
import torch
from typing import List, Dict
from pathlib import Path

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel


class ChatBot:
    """聊天机器人"""
    
    def __init__(
        self,
        model_path: str,
        base_model: str = None,
        load_in_4bit: bool = False,
        device: str = "auto"
    ):
        """初始化聊天机器人
        
        Args:
            model_path: 微调后的模型路径（LoRA权重）
            base_model: 基础模型名称或路径（如果为None则从配置文件读取）
            load_in_4bit: 是否使用4bit量化加载
            device: 设备
        """
        self.model_path = model_path
        self.device = device
        self.load_in_4bit = load_in_4bit
        
        # 尝试加载训练配置
        config_path = os.path.join(model_path, "training_config.yaml")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            if base_model is None:
                base_model = self.config['model']['base_model']
            self.system_prompt = self.config['character'].get('system_prompt', '')
            self.character_name = self.config['character'].get('name', '角色')
        else:
            self.config = None
            self.system_prompt = ""
            self.character_name = "角色"
        
        if base_model is None:
            raise ValueError("未找到基础模型信息，请指定base_model参数")
        
        self.base_model = base_model
        
        print("="*60)
        print(f"加载模型: {self.character_name}")
        print("="*60)
        
        self.load_model()
        
    def load_model(self):
        """加载模型"""
        print(f"基础模型: {self.base_model}")
        print(f"LoRA权重: {self.model_path}")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("✓ 分词器加载完成")
        
        # 配置量化
        quantization_config = None
        if self.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            print("使用4bit量化加载")
        
        # 加载基础模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=quantization_config,
            trust_remote_code=True,
            device_map=self.device,
            torch_dtype=torch.float16,
        )
        
        print("✓ 基础模型加载完成")
        
        # 加载LoRA权重
        self.model = PeftModel.from_pretrained(
            self.model,
            self.model_path,
            device_map=self.device,
        )
        
        print("✓ LoRA权重加载完成")
        
        # 设置为评估模式
        self.model.eval()
        
        print("✓ 模型准备完成")
        print("="*60)
        
    def format_conversation(self, messages: List[Dict[str, str]]) -> str:
        """格式化对话历史
        
        Args:
            messages: 对话历史
            
        Returns:
            格式化后的文本
        """
        formatted = ""
        
        # 添加系统提示
        if self.system_prompt:
            formatted += f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
        
        # 添加对话历史
        for msg in messages:
            role = msg['role']
            content = msg['content']
            formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        
        # 添加assistant开始标记
        formatted += "<|im_start|>assistant\n"
        
        return formatted
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
    ) -> str:
        """生成回复
        
        Args:
            messages: 对话历史
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_p: nucleus sampling参数
            top_k: top-k sampling参数
            repetition_penalty: 重复惩罚
            do_sample: 是否采样
            
        Returns:
            生成的回复
        """
        # 格式化输入
        prompt = self.format_conversation(messages)
        
        # 编码
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.model.device)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # 解码
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # 移除可能的结束标记
        response = response.replace("<|im_end|>", "").strip()
        
        return response
    
    def chat(self):
        """交互式对话"""
        print(f"\n开始与 {self.character_name} 对话")
        print("输入 'exit' 或 'quit' 退出，输入 'clear' 清空对话历史")
        print("="*60 + "\n")
        
        conversation_history = []
        
        while True:
            try:
                # 获取用户输入
                user_input = input("你: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['exit', 'quit']:
                    print("\n再见！")
                    break
                
                if user_input.lower() == 'clear':
                    conversation_history = []
                    print("\n✓ 对话历史已清空\n")
                    continue
                
                # 添加用户消息
                conversation_history.append({
                    'role': 'user',
                    'content': user_input
                })
                
                # 生成回复
                response = self.generate(conversation_history)
                
                # 添加助手消息
                conversation_history.append({
                    'role': 'assistant',
                    'content': response
                })
                
                # 打印回复
                print(f"{self.character_name}: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\n再见！")
                break
            except Exception as e:
                print(f"\n错误: {e}\n")
                continue


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="模型推理脚本")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="微调后的模型路径"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="基础模型名称或路径（可选，会尝试从训练配置读取）"
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="是否使用4bit量化加载"
    )
    parser.add_argument(
        "--single_turn",
        action="store_true",
        help="单轮对话模式（非交互式）"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="单轮对话的输入（需配合--single_turn使用）"
    )
    
    args = parser.parse_args()
    
    # 创建聊天机器人
    bot = ChatBot(
        model_path=args.model_path,
        base_model=args.base_model,
        load_in_4bit=args.load_in_4bit,
    )
    
    # 单轮对话或交互式对话
    if args.single_turn:
        if not args.query:
            print("错误: 单轮对话模式需要提供--query参数")
            return
        
        messages = [{'role': 'user', 'content': args.query}]
        response = bot.generate(messages)
        print(f"\n{bot.character_name}: {response}\n")
    else:
        bot.chat()


if __name__ == "__main__":
    main()

