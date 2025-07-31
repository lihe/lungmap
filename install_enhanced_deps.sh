#!/bin/bash
# 增强训练功能依赖安装脚本

echo "🚀 安装CPR-TaG-Net增强训练功能依赖"
echo "==========================================="

# 检查Python环境
echo "🔍 检查Python环境..."
python3 --version || { echo "❌ Python3未安装"; exit 1; }

# 安装基础科学计算包
echo "📦 安装基础依赖包..."
pip3 install --upgrade pip

echo "   安装PyTorch..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "   安装科学计算包..."
pip3 install numpy scipy matplotlib seaborn

echo "   安装机器学习包..."
pip3 install scikit-learn pandas

echo "   安装进度条和工具包..."
pip3 install tqdm pathlib

echo "   安装图形处理包..."
pip3 install pillow

# 安装已有的requirements.txt中的包
if [ -f "requirements.txt" ]; then
    echo "📋 安装项目依赖..."
    pip3 install -r requirements.txt
fi

echo ""
echo "✅ 依赖安装完成!"
echo ""
echo "🧪 运行测试验证增强功能:"
echo "   python3 test_enhanced_features.py"
echo ""
echo "🚀 启动增强训练:"
echo "   python3 run_optimized_training.py"
