# CPR-TaG-Net 完整训练Pipeline使用指南

## 📋 项目概述

本项目实现了从肺动脉分割标签到血管分类的完整pipeline，包括：
1. 标签过滤和预处理
2. 中心线提取和图构建
3. CPR-TaG-Net训练和推理

## 🏗️ 项目结构

```
/Users/leslie/IT/feixueguanditu/test/
├── train/                          # CT图像目录 (.nii文件)
├── label/                          # 原始分割标签
├── label_filtered/                 # 过滤后的标签 (已完成)
├── keep.txt                        # 保留标签列表
├── CPR_TaG_Net/                    # CPR-TaG-Net模型代码
│   └── models/
│       ├── cpr_tagnet.py          # 主模型
│       └── modules/               # 模型组件
├── vessel_preprocessing.py         # 数据预处理脚本
├── data_loader.py                 # 数据加载器
├── train_pipeline.py              # 完整训练pipeline
└── processed_data/                # 处理后的训练数据 (将生成)
```

## 🔧 环境要求

### 必需依赖
```bash
pip install SimpleITK numpy scipy scikit-image scikit-learn networkx
pip install torch torchvision torch-geometric  # PyTorch相关
pip install tqdm tensorboard  # 训练相关
```

### 可选依赖（增强功能）
```bash
pip install vtk  # 3D可视化
pip install matplotlib seaborn  # 绘图
```

## 🚀 快速开始

### 1. 数据预处理 (必需)

首先需要安装依赖并运行预处理：

```bash
# 安装必需的依赖包
pip install SimpleITK numpy scipy scikit-image scikit-learn networkx

# 运行数据预处理（仅预处理，不训练）
python train_pipeline.py --preprocess_only
```

预处理过程包括：
- ✅ 从过滤后的标签中提取血管中心线
- ✅ 构建血管拓扑图结构
- ✅ 采样节点周围的3D图像块
- ✅ 计算54维几何特征
- ✅ 生成训练数据

### 2. 模型训练 (需要PyTorch)

如果已安装PyTorch环境：

```bash
# 安装PyTorch依赖
pip install torch torchvision torch-geometric tqdm tensorboard

# 运行完整训练pipeline
python train_pipeline.py --epochs 50 --batch_size 2
```

## 📊 数据预处理详解

### 输入数据
- **CT图像**: `train/` 目录下的 `.nii` 文件
- **分割标签**: `label_filtered/` 目录下的 `.seg.nrrd` 文件 (24个文件)
- **标签映射**: 15个前三级肺动脉血管标签

### 处理流程

#### 1. 中心线提取
```python
# 对每个血管标签：
# 1. 形态学清理 -> 2. 3D骨架化 -> 3. 中心线排序 -> 4. 半径估计
vessel_mask -> skeleton_3d -> ordered_centerline -> radii
```

#### 2. 几何特征计算 (54维)
```
坐标 (3) + 半径 (1) + 方向 (3) + 曲率 (1) + 扭转 (1) + 
局部统计 (6) + 距离特征 (6) + 分叉特征 (3) + 位置编码 (30) = 54维
```

#### 3. 图构建
```python
# 节点：中心线采样点
# 边：血管内连接 + 解剖学连接
# 特征：54维几何特征 + 32³图像块
```

### 输出数据
每个病例生成一个 `.npz` 文件，包含：
- `node_features`: [N, 54] 节点几何特征
- `node_positions`: [N, 3] 节点3D坐标
- `edge_index`: [2, E] 图连接关系
- `image_cubes`: [N, 32, 32, 32] 节点图像块
- `node_classes`: [N] 节点类别标签 (0-17)

## 🧠 CPR-TaG-Net模型详解

### 模型架构
```
图像路径: 3D CNN -> 全局池化 -> 投影 [N, 256]
图结构路径: SA1 -> SA2 -> FP1 -> FP2 [N, 64]
融合分类: Concat -> MLP -> 分类 [N, 18]
```

### 类别定义 (18类)
```python
0: MPA (主肺动脉)
1-2: LPA, RPA (左右肺动脉)
3-6: 左侧二级血管 (Linternal, Lupper, Lmedium, Ldown)
7-10: 右侧二级血管 (Rinternal, Rupper, Rmedium, RDown)
11-12: 左侧三级血管 (L1+2, L1+3)
13-14: 右侧三级血管 (R1+2, R1+3)
15-17: 背景、不确定、连接点
```

## 🔧 自定义配置

### 修改类别数量
```python
# 在 train_pipeline.py 中修改
parser.add_argument('--num_classes', type=int, default=18)

# 在 vessel_preprocessing.py 中修改 vessel_hierarchy
```

### 修改特征维度
```python
# 在 vessel_preprocessing.py 的 _compute_geometric_features 中
# 调整特征计算逻辑，确保总维度为node_feature_dim

# 在 CPR_TaG_Net 中修改
CPRTaGNet(node_feature_dim=54)  # 改为新的特征维度
```

### 修改图像块大小
```python
# 在 vessel_preprocessing.py 中
VesselPreprocessor(cube_size=32)  # 改为其他尺寸如64

# 相应地在CPR-TaG-Net中调整CNN3D输入
```

## 📈 训练监控

### Tensorboard
```bash
# 在另一个终端运行
tensorboard --logdir outputs/logs
```

### 检查点
- `checkpoints/checkpoint_latest.pth`: 最新模型
- `checkpoints/checkpoint_best.pth`: 最佳验证准确率模型

## 🚨 常见问题

### 1. 内存不足
```python
# 减少批次大小
python train_pipeline.py --batch_size 1

# 减少最大节点数
# 在 data_loader.py 中修改 max_nodes=500
```

### 2. GPU内存不足
```python
# 使用CPU训练（较慢）
# 模型会自动检测并使用可用设备

# 或减少图像块大小
VesselPreprocessor(cube_size=16)
```

### 3. 数据不平衡
```python
# 使用类别权重（已自动计算）
# 在 data_loader.py 的 get_class_weights() 中调整策略
```

## 🎯 结果分析

### 评估指标
- 整体准确率
- 各类别精确率/召回率
- 混淆矩阵
- 各血管类型的分类性能

### 可视化
- 血管图结构可视化
- 预测结果对比
- 特征空间分析

## 🔄 完整workflow

```bash
# 1. 确保数据准备完成
ls train/        # 应该看到 .nii 文件
ls label_filtered/  # 应该看到 .seg.nrrd 文件

# 2. 运行数据预处理
python train_pipeline.py --preprocess_only

# 3. 检查预处理结果
ls processed_data/  # 应该看到 _processed.npz 文件

# 4. 运行训练（如果有PyTorch环境）
python train_pipeline.py --epochs 50

# 5. 监控训练
tensorboard --logdir outputs/logs
```

## 📝 下一步

1. **模型优化**: 调整网络架构、超参数
2. **数据增强**: 更多几何变换、图像增强
3. **后处理**: 连通性检查、解剖学约束
4. **评估**: 在验证集上详细分析
5. **部署**: 转换为推理模式，集成到应用中

这个pipeline为您提供了从原始医学图像到血管分类的完整解决方案！
