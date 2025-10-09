"""
数据格式转换工具
将不同来源的数据转换为标准训练格式
"""

import json
import csv
import re
from typing import List, Dict, Any
from pathlib import Path


class DataConverter:
    """数据格式转换器"""
    
    @staticmethod
    def from_txt_dialogue(input_file: str, output_file: str, separator: str = "---"):
        """从文本对话文件转换
        
        文本格式示例：
        用户: 你好
        角色: 你好！很高兴见到你。
        ---
        用户: 今天天气怎么样？
        角色: 今天天气不错。
        ---
        
        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
            separator: 对话分隔符
        """
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 按分隔符分割对话
        dialogues = content.split(separator)
        
        conversations = []
        for dialogue in dialogues:
            dialogue = dialogue.strip()
            if not dialogue:
                continue
            
            lines = dialogue.split('\n')
            conversation = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # 匹配 "用户: " 或 "角色: " 格式
                if line.startswith('用户:') or line.startswith('user:'):
                    content = line.split(':', 1)[1].strip()
                    conversation.append({'role': 'user', 'content': content})
                elif line.startswith('角色:') or line.startswith('assistant:') or line.startswith('AI:'):
                    content = line.split(':', 1)[1].strip()
                    conversation.append({'role': 'assistant', 'content': content})
            
            if conversation:
                conversations.append({'conversation': conversation})
        
        # 保存
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 转换完成！共 {len(conversations)} 条对话")
        print(f"✓ 已保存到: {output_file}")
    
    @staticmethod
    def from_csv(input_file: str, output_file: str, 
                 user_column: str = 'user', 
                 assistant_column: str = 'assistant',
                 encoding: str = 'utf-8'):
        """从CSV文件转换
        
        CSV格式示例：
        user,assistant
        你好,你好！很高兴见到你。
        今天天气怎么样？,今天天气不错。
        
        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
            user_column: 用户列名
            assistant_column: 助手列名
            encoding: 文件编码
        """
        conversations = []
        
        with open(input_file, 'r', encoding=encoding) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if user_column in row and assistant_column in row:
                    conversation = [
                        {'role': 'user', 'content': row[user_column].strip()},
                        {'role': 'assistant', 'content': row[assistant_column].strip()}
                    ]
                    conversations.append({'conversation': conversation})
        
        # 保存
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 转换完成！共 {len(conversations)} 条对话")
        print(f"✓ 已保存到: {output_file}")
    
    @staticmethod
    def from_weibo(input_file: str, output_file: str):
        """从微博数据转换
        
        微博格式示例（每条一行JSON）：
        {"text": "今天天气真好", "user": "张三", "time": "2024-01-01"}
        
        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
        """
        conversations = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    text = data.get('text', '').strip()
                    
                    if text:
                        # 将每条微博作为一个独立的对话
                        # 可以添加一个通用的用户提问
                        conversation = [
                            {'role': 'user', 'content': '说点什么吧'},
                            {'role': 'assistant', 'content': text}
                        ]
                        conversations.append({'conversation': conversation})
                except json.JSONDecodeError:
                    continue
        
        # 保存
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 转换完成！共 {len(conversations)} 条对话")
        print(f"✓ 已保存到: {output_file}")
    
    @staticmethod
    def from_chat_history(input_file: str, output_file: str, 
                         target_name: str = "角色"):
        """从聊天记录转换
        
        聊天记录格式示例：
        [2024-01-01 10:00] 张三: 你好
        [2024-01-01 10:01] 角色: 你好！
        [2024-01-01 10:02] 张三: 今天天气怎么样？
        [2024-01-01 10:03] 角色: 今天天气不错。
        
        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
            target_name: 目标角色的名字
        """
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 解析聊天记录
        pattern = re.compile(r'\[(.*?)\] (.*?): (.*)')
        
        current_conversation = []
        conversations = []
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_conversation:
                    conversations.append({'conversation': current_conversation})
                    current_conversation = []
                continue
            
            match = pattern.match(line)
            if match:
                timestamp, name, content = match.groups()
                
                if name == target_name:
                    role = 'assistant'
                else:
                    role = 'user'
                
                current_conversation.append({
                    'role': role,
                    'content': content.strip()
                })
        
        # 添加最后一个对话
        if current_conversation:
            conversations.append({'conversation': current_conversation})
        
        # 保存
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 转换完成！共 {len(conversations)} 条对话")
        print(f"✓ 已保存到: {output_file}")


def main():
    """命令行工具"""
    import argparse
    
    parser = argparse.ArgumentParser(description="数据格式转换工具")
    parser.add_argument("--input", required=True, help="输入文件路径")
    parser.add_argument("--output", required=True, help="输出文件路径")
    parser.add_argument("--format", required=True, 
                       choices=['txt', 'csv', 'weibo', 'chat'],
                       help="输入格式: txt/csv/weibo/chat")
    parser.add_argument("--separator", default="---", help="TXT格式的对话分隔符")
    parser.add_argument("--user_column", default="user", help="CSV格式的用户列名")
    parser.add_argument("--assistant_column", default="assistant", help="CSV格式的助手列名")
    parser.add_argument("--target_name", default="角色", help="聊天记录中的目标角色名")
    
    args = parser.parse_args()
    
    converter = DataConverter()
    
    print("="*60)
    print("数据格式转换")
    print("="*60)
    
    if args.format == 'txt':
        converter.from_txt_dialogue(args.input, args.output, args.separator)
    elif args.format == 'csv':
        converter.from_csv(args.input, args.output, 
                          args.user_column, args.assistant_column)
    elif args.format == 'weibo':
        converter.from_weibo(args.input, args.output)
    elif args.format == 'chat':
        converter.from_chat_history(args.input, args.output, args.target_name)


if __name__ == "__main__":
    main()

