#!/bin/bash

# CPR-TaG-Net 血管分类训练启动脚本
echo "🚀 CPR-TaG-Net 血管分类训练启动器"
echo "=================================================="

# 检查是否在正确的目录
if [ ! -f "train.py" ]; then
    echo "❌ 请在 lungmap 项目根目录下运行此脚本"
    exit 1
fi

# 激活conda环境
echo "🔧 激活conda环境..."
eval "$(conda shell.bash hook)"
conda activate lungmap

# 检查环境
echo "🔍 检查环境..."
python -c "
import torch
import torch_geometric
print('✅ PyTorch版本:', torch.__version__)
print('✅ PyTorch Geometric版本:', torch_geometric.__version__)
print('✅ CUDA可用:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✅ GPU设备:', torch.cuda.get_device_name(0))
    print('✅ GPU显存:', torch.cuda.get_device_properties(0).total_memory // 1024**3, 'GB')
"

if [ $? -ne 0 ]; then
    echo "❌ 环境检查失败，请检查conda环境"
    exit 1
fi

# 检查数据
echo "📁 检查数据..."
if [ ! -d "data/processed" ]; then
    echo "❌ 数据目录不存在: data/processed"
    exit 1
fi

npz_count=$(ls data/processed/*.npz 2>/dev/null | wc -l)
echo "📊 找到 $npz_count 个预处理数据文件"

if [ $npz_count -eq 0 ]; then
    echo "❌ 没有找到预处理数据文件 (.npz)"
    exit 1
fi

# 显示训练选项
echo ""
echo "🔧 训练模式选择:"
echo "1. 快速测试模式 (小数据集, 10轮)"
echo "2. 标准训练模式 (中等数据集, 50轮)"
echo "3. 完整训练模式 (大数据集, 100轮)"
echo "4. 血管感知训练模式 (推荐, 带血管层次信息)"
echo "5. 🔬 验证改进模式 (统一损失+动态验证+交叉验证)"
echo "6. 🚀 全功能训练模式 (所有改进功能启用)"

read -p "请选择训练模式 [1-6]: " choice

case $choice in
    1)
        echo "🔧 启动快速测试模式..."
        python train.py \
            --epochs 10 \
            --max_nodes 500 \
            --node_batch_size 200 \
            --save_freq 2 \
            --enable_visualization \
            --save_training_curves
        ;;
    2)
        echo "🔧 启动标准训练模式..."
        python train.py \
            --epochs 50 \
            --max_nodes 2000 \
            --node_batch_size 300 \
            --save_freq 5 \
            --enable_vessel_aware \
            --enable_visualization \
            --save_training_curves \
            --save_confusion_matrix
        ;;
    3)
        echo "🔧 启动完整训练模式..."
        python train.py \
            --epochs 100 \
            --enable_large_cases \
            --max_nodes_per_case 8000 \
            --node_batch_size 500 \
            --save_freq 10 \
            --enable_vessel_aware \
            --enable_visualization \
            --save_training_curves \
            --save_confusion_matrix \
            --enable_graph_completion
        ;;
    4)
        echo "🔧 启动血管感知训练模式 (推荐)..."
        python train.py \
            --epochs 50 \
            --enable_large_cases \
            --max_nodes_per_case 6000 \
            --node_batch_size 400 \
            --save_freq 5 \
            --enable_vessel_aware \
            --vessel_consistency_weight 0.1 \
            --spatial_consistency_weight 0.05 \
            --enable_visualization \
            --save_training_curves \
            --save_confusion_matrix
        ;;
    5)
        echo "🔬 启动验证改进模式 (统一损失+动态验证+交叉验证)..."
        python train.py \
            --epochs 30 \
            --max_nodes 2000 \
            --node_batch_size 300 \
            --save_freq 5 \
            --enable_vessel_aware \
            --vessel_consistency_weight 0.1 \
            --spatial_consistency_weight 0.05 \
            --dynamic_split_interval 5 \
            --enable_cross_validation \
            --cv_folds 3 \
            --enable_leave_one_out \
            --save_training_curves \
            --save_confusion_matrix
        ;;
    6)
        echo "🚀 启动全功能训练模式 (所有改进功能启用)..."
        python train.py \
            --epochs 50 \
            --enable_large_cases \
            --max_nodes_per_case 8000 \
            --node_batch_size 500 \
            --save_freq 5 \
            --enable_vessel_aware \
            --vessel_consistency_weight 0.1 \
            --spatial_consistency_weight 0.05 \
            --dynamic_split_interval 10 \
            --enable_cross_validation \
            --cv_folds 5 \
            --enable_leave_one_out \
            --enable_graph_completion \
            --enable_visualization \
            --save_training_curves \
            --save_confusion_matrix
        ;;
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac

echo ""
echo "🎉 训练完成!"
echo "📝 查看训练日志: outputs/logs/"
echo "💾 查看模型检查点: outputs/checkpoints/"
echo "📊 查看可视化结果: outputs/visualizations/"
echo ""
echo "🔬 验证改进功能说明:"
echo "  - 统一损失函数: 训练和验证都使用层级损失"
echo "  - 动态验证集: 定期重新分割数据防止过拟合"
echo "  - K-fold交叉验证: 更可靠的模型评估"
echo "  - 留一法验证: 小数据集的严格验证"
echo "  - 综合验证分析: 对比多种验证方法的结果"
