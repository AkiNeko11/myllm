#!/bin/bash

# 角色语言风格微调 - 快速启动脚本（Linux/Mac）

echo "========================================"
echo "角色语言风格微调 - 快速启动脚本"
echo "========================================"
echo ""

show_menu() {
    echo "请选择操作："
    echo "1. 安装依赖"
    echo "2. 处理数据"
    echo "3. 开始训练"
    echo "4. 测试模型"
    echo "5. 查看训练日志（TensorBoard）"
    echo "6. 退出"
    echo ""
}

install_deps() {
    echo ""
    echo "[1/3] 安装依赖..."
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    echo ""
    echo "✓ 依赖安装完成！"
    read -p "按Enter继续..."
}

prepare_data() {
    echo ""
    echo "[2/3] 处理数据..."
    python scripts/prepare_data.py
    echo ""
    echo "✓ 数据处理完成！"
    read -p "按Enter继续..."
}

train_model() {
    echo ""
    echo "[3/3] 开始训练..."
    echo "提示：训练过程中可以按Ctrl+C中断"
    python scripts/train.py
    echo ""
    echo "✓ 训练完成！"
    read -p "按Enter继续..."
}

inference() {
    echo ""
    read -p "请输入模型路径（例如：checkpoints/角色名_Qwen1.5-1.8B-Chat/final_model）: " model_path
    echo ""
    echo "启动交互式对话..."
    python scripts/inference.py --model_path "$model_path" --load_in_4bit
    read -p "按Enter继续..."
}

tensorboard_view() {
    echo ""
    echo "启动TensorBoard..."
    echo "打开浏览器访问: http://localhost:6006"
    tensorboard --logdir logs
    read -p "按Enter继续..."
}

while true; do
    show_menu
    read -p "请输入选项 (1-6): " choice
    
    case $choice in
        1) install_deps ;;
        2) prepare_data ;;
        3) train_model ;;
        4) inference ;;
        5) tensorboard_view ;;
        6) echo ""; echo "再见！"; exit 0 ;;
        *) echo "无效选项，请重新选择" ;;
    esac
done

