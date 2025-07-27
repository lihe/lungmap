#!/bin/bash

# CPR-TaG-Net 数据预处理和训练脚本

echo "=== CPR-TaG-Net 批量数据处理 ==="

# 检查Python环境
echo "检查Python环境..."
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import nibabel; print('NiBabel: ✓')" || echo "请安装: pip install nibabel"
python -c "import nrrd; print('PyNRRD: ✓')" || echo "请安装: pip install pynrrd"
python -c "import skimage; print('scikit-image: ✓')" || echo "请安装: pip install scikit-image"

echo ""
echo "=== 步骤 1: 批量数据预处理 ==="
echo "请确保您已在 data/batch_preprocess.py 中设置了正确的数据路径"
echo "开始处理25组CT和分割数据..."

cd /Users/leslie/IT/CPR_TaG_Net
python data/batch_preprocess.py

if [ $? -eq 0 ]; then
    echo "✅ 数据预处理完成!"
    echo ""
    echo "=== 步骤 2: 开始训练 ==="
    echo "启动CPR-TaG-Net训练..."
    python train_eval.py
else
    echo "❌ 数据预处理失败，请检查错误信息"
    exit 1
fi

echo ""
echo "=== 训练完成 ==="
echo "检查结果文件："
echo "- 模型检查点: ./checkpoints/"
echo "- 训练日志: ./logs/"
