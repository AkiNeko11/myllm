"""
数据预处理脚本
支持多种对话格式，将原始数据转换为训练所需格式
"""

import json
import os
import yaml
from pathlib import Path
from typing import List, Dict, Any
from datasets import Dataset
import random


class DataProcessor:
    """数据处理器，支持多种对话格式"""
    
    def __init__(self, config_path: str = "config/train_config.yaml"):
        """初始化数据处理器
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
        self.character_config = self.config['character']
        
    def load_raw_data(self) -> List[Dict[str, Any]]:
        """加载原始数据
        
        Returns:
            原始数据列表
        """
        raw_path = self.data_config['raw_data_path']
        
        if not os.path.exists(raw_path):
            raise FileNotFoundError(
                f"原始数据文件不存在: {raw_path}\n"
                f"请将您的对话数据放在该路径，或修改config/train_config.yaml中的raw_data_path"
            )
        
        with open(raw_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✓ 成功加载 {len(data)} 条原始数据")
        return data
    
    def format_standard(self, conversations: List[Dict[str, str]]) -> str:
        """标准格式转换（默认格式）
        
        标准格式示例:
        [
          {
            "conversation": [
              {"role": "user", "content": "你好"},
              {"role": "assistant", "content": "你好！"}
            ]
          }
        ]
        
        Args:
            conversations: 对话列表
            
        Returns:
            格式化后的文本
        """
        formatted_text = ""
        system_prompt = self.character_config.get('system_prompt', '')
        
        if system_prompt:
            formatted_text += f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        
        for turn in conversations:
            role = turn['role']
            content = turn['content']
            formatted_text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        
        return formatted_text.strip()
    
    def format_sharegpt(self, conversations: List[Dict[str, str]]) -> str:
        """ShareGPT格式转换
        
        ShareGPT格式示例:
        [
          {
            "conversations": [
              {"from": "human", "value": "你好"},
              {"from": "gpt", "value": "你好！"}
            ]
          }
        ]
        
        Args:
            conversations: 对话列表
            
        Returns:
            格式化后的文本
        """
        # 转换角色名称
        role_map = {
            'human': 'user',
            'gpt': 'assistant',
            'system': 'system'
        }
        
        standard_conversations = []
        for turn in conversations:
            role = role_map.get(turn.get('from', 'human'), 'user')
            content = turn.get('value', '')
            standard_conversations.append({'role': role, 'content': content})
        
        return self.format_standard(standard_conversations)
    
    def format_alpaca(self, item: Dict[str, str]) -> str:
        """Alpaca格式转换
        
        Alpaca格式示例:
        [
          {
            "instruction": "任务描述",
            "input": "输入（可选）",
            "output": "输出"
          }
        ]
        
        Args:
            item: 单条数据
            
        Returns:
            格式化后的文本
        """
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output = item.get('output', '')
        
        if input_text:
            user_content = f"{instruction}\n\n{input_text}"
        else:
            user_content = instruction
        
        conversations = [
            {'role': 'user', 'content': user_content},
            {'role': 'assistant', 'content': output}
        ]
        
        return self.format_standard(conversations)
    
    def process_data(self) -> Dataset:
        """处理数据
        
        Returns:
            处理后的数据集
        """
        raw_data = self.load_raw_data()
        data_format = self.data_config['data_format']
        
        processed_texts = []
        
        print(f"开始处理数据，格式: {data_format}")
        
        for idx, item in enumerate(raw_data):
            try:
                if data_format == 'standard':
                    # 标准格式
                    conversations = item.get('conversation', [])
                    text = self.format_standard(conversations)
                    
                elif data_format == 'sharegpt':
                    # ShareGPT格式
                    conversations = item.get('conversations', [])
                    text = self.format_sharegpt(conversations)
                    
                elif data_format == 'alpaca':
                    # Alpaca格式
                    text = self.format_alpaca(item)
                    
                else:
                    raise ValueError(f"不支持的数据格式: {data_format}")
                
                processed_texts.append({'text': text})
                
                # 打印第一条数据作为示例
                if idx == 0:
                    print(f"\n示例数据格式化结果:\n{'-'*50}")
                    print(text)
                    print(f"{'-'*50}\n")
                    
            except Exception as e:
                print(f"⚠ 警告: 处理第 {idx} 条数据时出错: {e}")
                continue
        
        print(f"✓ 成功处理 {len(processed_texts)} 条数据")
        
        # 创建数据集
        dataset = Dataset.from_list(processed_texts)
        
        # 分割训练集和验证集
        val_split = self.data_config['validation_split']
        if val_split > 0:
            split_dataset = dataset.train_test_split(
                test_size=val_split,
                seed=self.config['other']['seed']
            )
            train_dataset = split_dataset['train']
            val_dataset = split_dataset['test']
            
            print(f"✓ 训练集: {len(train_dataset)} 条")
            print(f"✓ 验证集: {len(val_dataset)} 条")
            
            return train_dataset, val_dataset
        else:
            print(f"✓ 训练集: {len(dataset)} 条（无验证集）")
            return dataset, None
    
    def save_processed_data(self, train_dataset: Dataset, val_dataset: Dataset = None):
        """保存处理后的数据
        
        Args:
            train_dataset: 训练数据集
            val_dataset: 验证数据集（可选）
        """
        output_dir = self.data_config['processed_data_path']
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存训练集
        train_path = os.path.join(output_dir, 'train.jsonl')
        with open(train_path, 'w', encoding='utf-8') as f:
            for item in train_dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"✓ 训练集已保存到: {train_path}")
        
        # 保存验证集
        if val_dataset:
            val_path = os.path.join(output_dir, 'validation.jsonl')
            with open(val_path, 'w', encoding='utf-8') as f:
                for item in val_dataset:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"✓ 验证集已保存到: {val_path}")


def main():
    """主函数"""
    print("="*60)
    print("数据预处理")
    print("="*60)
    
    processor = DataProcessor()
    train_dataset, val_dataset = processor.process_data()
    processor.save_processed_data(train_dataset, val_dataset)
    
    print("\n" + "="*60)
    print("✓ 数据预处理完成！")
    print("="*60)


if __name__ == "__main__":
    main()

