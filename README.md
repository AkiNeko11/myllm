# 角色语言风格微调框架

一个基于LoRA/QLoRA的角色语言风格迁移学习框架，用于微调大语言模型，使其具备特定角色的语言风格。

## 📋 项目特点

- ✅ **模板化设计**：只需准备语料和修改配置，即可训练不同角色
- ✅ **参数高效**：使用LoRA/QLoRA，4060笔记本即可训练
- ✅ **多格式支持**：支持标准对话、ShareGPT、Alpaca等数据格式
- ✅ **开箱即用**：完整的数据处理、训练、推理流程
- ✅ **基于Qwen-1.8B-Chat**：阿里巴巴开源中文对话模型

## 🚀 快速开始

### 1. 环境配置

```bash
# 创建虚拟环境（推荐）
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

> **注意**：如果使用4060笔记本，确保已安装CUDA和对应版本的PyTorch

### 2. 准备数据

将您收集的角色对话语料放在 `data/raw/` 目录下，支持以下格式：

#### 标准格式（推荐）

```json
[
  {
    "conversation": [
      {"role": "user", "content": "你好"},
      {"role": "assistant", "content": "你好！很高兴见到你。"},
      {"role": "user", "content": "今天天气怎么样？"},
      {"role": "assistant", "content": "抱歉，我无法获取实时天气信息。"}
    ]
  }
]
```

#### ShareGPT格式

```json
[
  {
    "conversations": [
      {"from": "human", "value": "你好"},
      {"from": "gpt", "value": "你好！"}
    ]
  }
]
```

#### Alpaca格式

```json
[
  {
    "instruction": "请介绍一下自己",
    "input": "",
    "output": "我是一个AI助手..."
  }
]
```

### 3. 配置训练参数

编辑 `config/train_config.yaml`，修改以下关键配置：

```yaml
# 角色信息
character:
  name: "你的角色名"  # 会用于模型保存命名
  description: "角色描述"
  system_prompt: "你是XXX，你的性格是..."  # 定义角色设定

# 数据配置
data:
  raw_data_path: "data/raw/conversations.json"  # 你的语料文件
  data_format: "standard"  # 数据格式: standard/sharegpt/alpaca
  max_length: 2048
  validation_split: 0.05  # 5%作为验证集

# LoRA配置
lora:
  r: 64  # LoRA秩，越大容量越大（推荐32-64）
  lora_alpha: 16

# 训练配置
training:
  num_train_epochs: 3  # 训练轮数
  per_device_train_batch_size: 4  # batch size
  gradient_accumulation_steps: 4  # 梯度累积
  learning_rate: 2.0e-4
```

### 4. 数据预处理

```bash
python scripts/prepare_data.py
```

这会将原始对话数据转换为训练格式，并保存在 `data/processed/` 目录。

### 5. 开始训练

```bash
python scripts/train.py
```

训练过程中会：
- 自动下载Qwen-1.8B-Chat基础模型（首次运行）
- 使用QLoRA进行参数高效微调
- 定期保存checkpoint
- 生成TensorBoard日志

**查看训练日志**：

```bash
tensorboard --logdir logs
```

### 6. 模型推理

训练完成后，使用以下命令进行对话测试：

```bash
# 交互式对话
python scripts/inference.py --model_path checkpoints/你的角色名_Qwen-1.8B-Chat/final_model --load_in_4bit

# 单轮对话
python scripts/inference.py --model_path checkpoints/你的角色名_Qwen-1.8B-Chat/final_model --load_in_4bit --single_turn --query "你好"
```

### 7. 合并模型（可选）

如果需要将LoRA权重合并到基础模型生成完整模型：

```bash
python scripts/merge_lora.py \
  --base_model Qwen/Qwen-1.8B-Chat \
  --lora_path checkpoints/你的角色名_Qwen-1.8B-Chat/final_model \
  --output_path output/merged_model
```

## 📁 项目结构

```
myllm/
├── config/                    # 配置文件
│   └── train_config.yaml     # 训练配置（核心配置文件）
├── data/                      # 数据目录
│   ├── raw/                  # 原始语料（放你的对话数据）
│   └── processed/            # 处理后的数据
├── scripts/                   # 脚本
│   ├── prepare_data.py       # 数据预处理
│   ├── train.py              # 训练脚本
│   ├── inference.py          # 推理脚本
│   └── merge_lora.py         # 合并LoRA权重
├── checkpoints/               # 训练checkpoint
├── logs/                      # 训练日志
├── output/                    # 输出目录
└── requirements.txt           # Python依赖
```

## 💡 使用技巧

### 数据准备建议

1. **数据量**：建议至少准备100-500轮对话，质量比数量更重要
2. **数据质量**：确保对话符合角色特点，清洗掉不相关内容
3. **对话长度**：单轮对话不要太长，建议每轮100-500字
4. **多样性**：包含不同话题和场景的对话

### 训练参数调优

#### 显存不足？

```yaml
# 在 config/train_config.yaml 中调整：
training:
  per_device_train_batch_size: 2  # 减小batch size
  gradient_accumulation_steps: 8  # 增加梯度累积
  gradient_checkpointing: true    # 开启梯度检查点
```

#### 训练太慢？

```yaml
lora:
  r: 32  # 减小LoRA秩
```

#### 过拟合？

```yaml
training:
  num_train_epochs: 2  # 减少训练轮数
data:
  validation_split: 0.1  # 增加验证集比例
```

#### 欠拟合？

```yaml
lora:
  r: 64  # 增加LoRA秩
training:
  num_train_epochs: 5  # 增加训练轮数
  learning_rate: 3.0e-4  # 提高学习率
```

### 租卡训练

如果使用AutoDL、恒源云等云GPU平台：

```bash
# 上传代码和数据
# 在云平台终端运行：
git clone <你的仓库>
cd myllm
pip install -r requirements.txt
python scripts/train.py
```

## 🔧 高级功能

### 自定义数据格式

如果您的数据格式特殊，可以修改 `scripts/prepare_data.py` 中的 `DataProcessor` 类：

```python
def format_custom(self, item: Dict[str, str]) -> str:
    """自定义格式转换"""
    # 实现您的转换逻辑
    pass
```

### 使用wandb跟踪实验

```yaml
# 在 config/train_config.yaml 中修改：
other:
  report_to: "wandb"  # 改为wandb
```

然后登录wandb：
```bash
wandb login
```

### 多GPU训练

```bash
# 使用accelerate启动
accelerate config  # 首次配置
accelerate launch scripts/train.py
```

## ⚠️ 常见问题

### 1. CUDA Out of Memory

**解决方案**：
- 减小 `per_device_train_batch_size`
- 增加 `gradient_accumulation_steps`
- 减小 `max_length`
- 确保开启 `gradient_checkpointing`

### 2. 模型下载慢

**解决方案**：
```bash
# 设置HuggingFace镜像
export HF_ENDPOINT=https://hf-mirror.com  # Linux/Mac
set HF_ENDPOINT=https://hf-mirror.com     # Windows
```

### 3. 训练loss不下降

**检查项**：
- 数据格式是否正确
- 学习率是否合适
- 数据量是否足够
- 查看验证集loss判断是否过拟合

### 4. 生成效果不好

**优化建议**：
- 增加训练数据量和质量
- 调整 `system_prompt` 强化角色设定
- 尝试不同的推理参数（temperature、top_p）
- 增加训练轮数

## 📊 性能参考

**4060 笔记本 (8GB显存)**：
- 配置：batch_size=4, gradient_accumulation=4, r=64
- 显存占用：约7GB
- 训练速度：约2-3分钟/100步
- 可训练模型：Qwen-1.8B-Chat

**A100 40GB**：
- 可以使用更大的batch size和LoRA秩
- 或训练更大的模型（Qwen-7B/14B）

## 📝 更新日志

- 2024-10 初始版本，支持Qwen-1.8B-Chat微调

## 🤝 贡献

欢迎提Issue和PR！

## 📄 许可证

详见 LICENSE 文件

---

**Powered by Qwen-1.8B-Chat & LoRA**

如有问题，请查看文档或提Issue。
