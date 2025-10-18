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

from typing import Optional


def _find_local_snapshot(cache_root: Path, repo_id: str) -> Optional[Path]:
    """在本地缓存目录下查找给定 repo_id 的最新 snapshot 目录。

    期望的目录结构：
      <cache_root>/models--<org>--<name>/snapshots/<hash>/
    """
    parts = repo_id.split("/")
    if len(parts) != 2:
        return None

    org, name = parts
    snapshots_dir = cache_root / f"models--{org}--{name}" / "snapshots"
    if not snapshots_dir.exists() or not snapshots_dir.is_dir():
        return None

    candidates = [p for p in snapshots_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest

def merge_lora_weights(
    base_model_path: str,
    lora_path: str,
    output_path: str,
    device: str = "cpu",
    base_cache_dir: Optional[str] = None,
    offline: bool = False,
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
    
    # 解析基础模型本地路径（若提供的是 repo id，则尝试在本地缓存中解析 snapshot）
    resolved_base = base_model_path
    cache_root_guess = None
    if base_cache_dir:
        cache_root_guess = Path(base_cache_dir)
    else:
        # 默认尝试在脚本同级的上一级目录下寻找 base_models 作为缓存根
        # e.g. llm/myllm/base_models
        cache_root_guess = Path(__file__).resolve().parent.parent / "base_models"

    if not Path(resolved_base).is_dir():
        if cache_root_guess.exists():
            local_snapshot = _find_local_snapshot(cache_root_guess, base_model_path)
            if local_snapshot is not None:
                print(f"检测到本地基础模型快照: {local_snapshot}")
                resolved_base = str(local_snapshot)
        else:
            print(f"未发现本地缓存根目录: {cache_root_guess}")

    if offline and not Path(resolved_base).is_dir():
        raise RuntimeError(
            "离线模式已启用，但未能解析到本地基础模型快照。"
            " 请提供本地目录作为 --base_model，或指定 --base_cache_dir 并确保缓存完整。"
        )

    # 加载基础模型
    print("\n加载基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        resolved_base,
        trust_remote_code=True,
        device_map=device,
        torch_dtype=torch.float16,
        local_files_only=offline,
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
    parser.add_argument(
        "--base_cache_dir",
        type=str,
        default=None,
        help="本地基础模型缓存根目录（包含 models--<org>--<name> 的目录）"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="仅使用本地文件，不访问网络"
    )
    
    args = parser.parse_args()
    
    merge_lora_weights(
        base_model_path=args.base_model,
        lora_path=args.lora_path,
        output_path=args.output_path,
        device=args.device,
        base_cache_dir=args.base_cache_dir,
        offline=args.offline,
    )


if __name__ == "__main__":
    main()

