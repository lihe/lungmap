# CPR-TaG-Net: 肺血管地图构建网络

## 项目概述

CPR-TaG-Net 是一个用于肺血管地图构建的深度学习网络，结合了 CPR-GCN 的图像条件引导和 TaG-Net 的拓扑感知采样技术。该网络专门针对肺部血管结构的分类和分析任务设计。

## 网络特点

### 🏗️ 双路径架构设计
- **图结构路径**: 处理血管的几何拓扑特征
- **图像条件路径**: 提取血管的视觉纹理特征
- **多模态融合**: 结合几何和影像信息进行精确分类

### 🧠 核心技术创新
- **拓扑保持采样 (TPS)**: 保留血管分叉点和端点的关键拓扑结构
- **拓扑感知分组 (TFG)**: 基于图连接关系进行特征分组
- **多尺度特征学习**: U-Net风格的编码-解码结构

## 网络架构

```
输入数据
├── 图结构路径 (节点特征, 位置, 边连接)
│   ├── SA1 (TopoSAModule) → [N→M1, 54→128]
│   ├── SA2 (TopoSAModule) → [M1→M2, 128→256] 
│   ├── FP1 (TopoFPModule) → [M2→M1, 256+128→128]
│   └── FP2 (TopoFPModule) → [M1→N, 128+54→64]
│
├── 图像条件路径 (3D图像块)
│   ├── CNN3D → [N,1,D,H,W] → [N,64]
│   └── Linear投影 → [N,64] → [N,256]
│
└── 特征融合 + 分类
    ├── Concat → [N,64+256] → [N,320]
    └── MLP分类器 → [N,18]
```

## 项目结构

```
CPR_TaG_Net/
├── src/
│   ├── models/
│   │   └── CPR_TaG_Net/
│   │       ├── models/
│   │       │   ├── cpr_tagnet.py        # 主模型文件
│   │       │   └── modules/
│   │       │       ├── sa_module.py     # 采样聚合模块
│   │       │       ├── fp_module.py     # 特征传播模块
│   │       │       └── cnn3d_module.py  # 3D CNN模块
│   │       └── configs/
│   └── training/
│       └── vessel_data_loader.py        # 数据加载器
├── docs/                                # 文档目录
├── configs/                             # 配置文件
├── requirements.txt                     # 依赖配置
├── train.py                            # 训练脚本
├── simple_train.py                     # 简化训练脚本
└── run_optimized_training.py           # 优化训练脚本
```

## 环境要求

### 硬件要求
- **GPU**: NVIDIA RTX 4090 (24GB显存推荐)
- **内存**: 32GB RAM 推荐
- **存储**: 至少50GB可用空间

### 软件依赖
```
Python >= 3.8
PyTorch >= 1.12.0
PyTorch Geometric >= 2.0.0
CUDA >= 11.6
```

详细依赖列表请参考 [`requirements.txt`](requirements.txt)

## 使用方法

### 1. 环境配置
```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备
项目需要预处理的血管数据文件（.npz格式），包含：
- 节点特征 (54维)
- 3D坐标位置
- 图像立方体
- 边连接关系
- 分类标签 (18类血管分支)

### 3. 训练模型
```bash
# 24GB显存优化训练
python run_optimized_training.py

# 或使用基础训练脚本
python train.py

# 或使用简化训练脚本
python simple_train.py
```

### 4. 训练参数
- **最大节点数**: 6000 (24GB显存优化)
- **批处理大小**: 300节点/批
- **学习率**: 0.0005
- **训练轮数**: 25
- **优化器**: Adam with weight decay

## 模型性能

### 训练结果
- **参数量**: 513,615个参数
- **训练精度**: 84.83% (最终轮)
- **最佳验证精度**: 67.37%
- **支持案例**: 18个案例 (6000节点限制)

### 技术指标
- **内存优化**: 24GB显存充分利用
- **训练速度**: ~7分钟/轮 (18案例)
- **收敛稳定**: 无tensor维度错误

## 技术特点

### 医学价值
1. **术前规划**: 精确的血管分支定位
2. **介入导航**: 血管路径的自动规划  
3. **病理诊断**: 基于血管树结构的异常检测

### 技术优势
1. **多模态融合**: 结合几何和影像特征
2. **拓扑感知**: 保留血管关键结构
3. **内存优化**: 支持大规模血管数据
4. **临床导向**: 符合医学诊断需求

## 文档说明

- [`DEPENDENCY_ANALYSIS.md`](DEPENDENCY_ANALYSIS.md): 依赖分析文档
- [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md): 项目总结文档
- [`docs/README_Pipeline.md`](docs/README_Pipeline.md): 数据处理流程

## 贡献指南

欢迎提交Issue和Pull Request！

### 开发环境设置
1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交修改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 联系方式

- 项目负责人: lihe
- GitHub: [lihe/CPR_TaG_Net](https://github.com/lihe/CPR_TaG_Net)

## 致谢

感谢以下技术和论文的启发：
- CPR-GCN: 图像条件引导的图卷积网络
- TaG-Net: 拓扑感知的图神经网络
- PointNet++: 深度学习点云处理

---

*最后更新: 2025年7月27日*
