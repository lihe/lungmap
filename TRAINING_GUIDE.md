# CPR-TaG-Net 血管分类训练指南

## 🚀 快速开始

### 方法1: 使用bash脚本 (推荐)
```bash
# 在lungmap项目根目录下运行
./train.sh
```

### 方法2: 直接使用Python命令

#### 激活conda环境
```bash
conda activate lungmap
```

#### 快速测试 (10轮, 小数据集)
```bash
python train.py \
    --epochs 10 \
    --max_nodes 500 \
    --node_batch_size 200 \
    --enable_visualization \
    --save_training_curves
```

#### 血管感知训练 (推荐, 50轮)
```bash
python train.py \
    --epochs 50 \
    --enable_large_cases \
    --max_nodes_per_case 6000 \
    --node_batch_size 400 \
    --enable_vessel_aware \
    --vessel_consistency_weight 0.1 \
    --spatial_consistency_weight 0.05 \
    --enable_visualization \
    --save_training_curves \
    --save_confusion_matrix
```

#### 完整训练 (100轮, 需要大显存)
```bash
python train.py \
    --epochs 100 \
    --enable_large_cases \
    --max_nodes_per_case 8000 \
    --node_batch_size 500 \
    --enable_vessel_aware \
    --enable_visualization \
    --save_training_curves \
    --save_confusion_matrix \
    --enable_graph_completion
```

## 📊 训练状态监控

### 检查训练状态
```bash
./check_training.py
# 或者
python check_training.py
```

### GPU监控
```bash
nvidia-smi
# 或者持续监控
watch -n 1 nvidia-smi
```

### 查看日志
```bash
# 查看最新训练日志
ls -la outputs/logs/
tail -f outputs/logs/cpr_tagnet_training_*/training_*.txt
```

## 🔧 重要参数说明

### 显存相关
- `--enable_large_cases`: 启用大案例训练 (需要>16GB显存)
- `--max_nodes_per_case`: 单案例最大节点数
- `--node_batch_size`: 节点批大小

### 血管感知训练 (核心改进)
- `--enable_vessel_aware`: 启用血管感知训练 (推荐)
- `--vessel_consistency_weight`: 血管一致性损失权重 (默认0.1)
- `--spatial_consistency_weight`: 空间连续性损失权重 (默认0.05)

### 可视化功能
- `--enable_visualization`: 启用训练可视化
- `--save_training_curves`: 保存训练曲线
- `--save_confusion_matrix`: 保存混淆矩阵
- `--enable_graph_completion`: 启用图形补全

## 📁 输出目录结构

```
outputs/
├── checkpoints/           # 模型检查点
│   ├── best.pth          # 最佳模型
│   └── latest.pth        # 最新模型
├── logs/                 # 训练日志
│   └── cpr_tagnet_training_*/
│       └── training_*.txt
└── visualizations/       # 可视化结果
    ├── training_curve_*.png
    ├── confusion_matrix_*.png
    └── analysis_report_*.txt
```

## 🔍 训练进度监控

### 实时查看训练进度
```bash
# 方法1: 查看最新日志
tail -f outputs/logs/cpr_tagnet_training_*/training_*.txt

# 方法2: 使用我们的监控脚本
python check_training.py
```

### 训练指标含义
- **Train Loss**: 训练损失（应该逐渐下降）
- **Train Accuracy**: 训练准确率（应该逐渐上升）
- **Val Loss**: 验证损失
- **Val Accuracy**: 验证准确率（最重要的指标）

## ⚠️ 常见问题

### 1. 显存不足 (CUDA out of memory)
- 减小 `--node_batch_size`
- 减小 `--max_nodes_per_case`
- 关闭 `--enable_large_cases`

### 2. 训练速度慢
- 检查GPU使用率: `nvidia-smi`
- 增大 `--node_batch_size` (如果显存允许)
- 确保使用 `--enable_vessel_aware` 提升训练效率

### 3. 精度不提升
- 检查学习率是否合适
- 启用血管感知训练: `--enable_vessel_aware`
- 增加训练轮数
- 检查数据质量

### 4. 程序崩溃
- 检查conda环境: `conda activate lungmap`
- 检查依赖: `pip list | grep torch`
- 查看错误日志: `tail outputs/logs/*/training_*.txt`

## 🩸 血管感知训练特色

本训练代码实现了专门针对血管分类的改进：

1. **血管层次感知**: 利用解剖学血管层次结构 (MPA → LPA/RPA → 分支)
2. **血管连接保持**: 保持血管间的空间连续性
3. **血管上下文注入**: 注入血管类型和位置信息
4. **层次化损失函数**: 结合血管一致性和空间连续性损失

推荐使用血管感知训练模式获得最佳效果！

## 📈 预期训练时间

| 模式 | 数据量 | 轮数 | 预计时间 (RTX 3090) |
|------|--------|------|-------------------|
| 快速测试 | 小 | 10 | 30分钟 |
| 标准训练 | 中 | 50 | 2-3小时 |
| 完整训练 | 大 | 100 | 5-8小时 |
| 血管感知 | 中-大 | 50 | 3-4小时 |
