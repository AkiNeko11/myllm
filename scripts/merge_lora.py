"""
合并LoRA权重到基础模型
将LoRA适配器权重合并到基础模型，生成完整的独立模型
"""

import os
import sys
import yaml
import torch
import argparse
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_lora_weights(
    base_model_path: str,
    lora_path: str,
    output_path: str,
    device: str = "cpu"
):
    """合并LoRA权重到基础模型
    
    Args:
        base_model_path: 基础模型路径
        lora_path: LoRA权重路径
        output_path: 输出路径
        device: 设备
    """
    print("="*60)
    print("合并LoRA权重到基础模型")
    print("="*60)
    
    print(f"\n基础模型: {base_model_path}")
    print(f"LoRA权重: {lora_path}")
    print(f"输出路径: {output_path}")
    
    # 加载分词器
    print("\n加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        lora_path,
        trust_remote_code=True,
    )
    print("✓ 分词器加载完成")
    
    # 加载基础模型
    print("\n加载基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        device_map=device,
        torch_dtype=torch.float16,
    )
    print("✓ 基础模型加载完成")
    
    # 加载LoRA权重
    print("\n加载LoRA权重...")
    model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        device_map=device,
    )
    print("✓ LoRA权重加载完成")
    
    # 合并权重
    print("\n合并权重...")
    merged_model = model.merge_and_unload()
    print("✓ 权重合并完成")
    
    # 保存合并后的模型
    print(f"\n保存模型到 {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("✓ 模型保存完成")
    
    # 复制训练配置（如果存在）
    config_path = os.path.join(lora_path, "training_config.yaml")
    if os.path.exists(config_path):
        output_config_path = os.path.join(output_path, "training_config.yaml")
        import shutil
        shutil.copy(config_path, output_config_path)
        print("✓ 训练配置已复制")
    
    print("\n" + "="*60)
    print("✓ 合并完成！")
    print(f"✓ 完整模型已保存到: {output_path}")
    print("="*60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="合并LoRA权重到基础模型")
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="基础模型路径或名称"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="LoRA权重路径"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="合并后模型的输出路径"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="设备 (cpu/cuda/auto)"
    )
    
    args = parser.parse_args()
    
    merge_lora_weights(
        base_model_path=args.base_model,
        lora_path=args.lora_path,
        output_path=args.output_path,
        device=args.device,
    )


if __name__ == "__main__":
    main()

