#!/bin/bash

echo "=== 🚀 M2 Ultra 大模型训练环境 ==="
echo "正在启动..."

# 切换到脚本目录
cd "$(dirname "$0")"

# 检查conda环境
if ! conda env list | grep -q "llm_large"; then
    echo "❌ 未找到llm_large环境，正在创建..."
    source ~/miniforge3/etc/profile.d/conda.sh
    conda create -n llm_large python=3.10 -y
fi

# 激活环境
source ~/miniforge3/etc/profile.d/conda.sh
conda activate llm_large

echo "✅ 环境已激活"

# 显示菜单
echo ""
echo "=== 选择操作 ==="
echo "1. 安装基础环境"
echo "2. 量化大模型"
echo "3. 训练大模型"
echo "4. 监控内存使用"
echo "5. 退出"

read -p "请选择 (1-5): " choice

case $choice in
    1)
        echo "开始安装基础环境..."
        bash 安装大模型.sh
        ;;
    2)
        echo "开始量化大模型..."
        python3 量化大模型.py
        ;;
    3)
        echo "开始训练大模型..."
        python3 训练大模型.py
        ;;
    4)
        echo "监控内存使用..."
        python3 -c "
import torch
import psutil
import time

while True:
    memory = psutil.virtual_memory()
    print(f'系统内存: {memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB ({memory.percent}%)')
    
    if torch.backends.mps.is_available():
        allocated = torch.mps.get_allocated_memory() / 1024**3
        available = torch.mps.get_available_memory() / 1024**3
        print(f'MPS内存: {allocated:.1f}GB / {allocated+available:.1f}GB')
    
    time.sleep(5)
"
        ;;
    5)
        echo "退出..."
        exit 0
        ;;
    *)
        echo "❌ 无效选择"
        ;;
esac

echo ""
echo "按任意键退出..."
read -n 1
