#!/usr/bin/env python3
"""
血管分类模型训练启动脚本 - 整合数据加载和CPR-TaG-Net训练
"""

import os
import sys
import argparse
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from txt_logger import create_txt_logger  # 替换TensorBoard

# 添加模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'training'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'models', 'CPR_TaG_Net'))

from vessel_data_loader import load_processed_data, create_data_splits, create_dataloaders
from models.cpr_tagnet import CPRTaGNet

class VesselTrainer:
    """血管分类训练器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        
        # 创建输出目录
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        
        # 创建可视化目录（如果启用增强功能）
        if args.enable_visualization or args.save_confusion_matrix or args.save_training_curves:
            os.makedirs(args.visualization_dir, exist_ok=True)
            print(f"📊 可视化目录: {args.visualization_dir}")
        
        # 设置TXT日志记录器
        self.logger = create_txt_logger(args.log_dir, "cpr_tagnet_training")
        
        # 记录配置信息
        config_dict = {
            "model": "CPR-TaG-Net",
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "node_batch_size": args.node_batch_size,
            "weight_decay": args.weight_decay,
            "max_nodes_per_case": args.max_nodes_per_case,
            "enable_large_cases": args.enable_large_cases,
            "device": str(self.device),
            "enhanced_features": {
                "graph_completion": getattr(args, 'enable_graph_completion', False),
                "visualization": getattr(args, 'enable_visualization', False),
                "confusion_matrix": getattr(args, 'save_confusion_matrix', False),
                "training_curves": getattr(args, 'save_training_curves', False)
            }
        }
        self.logger.log_config(config_dict)
        
        # 🔧 血管感知训练改进：添加正确的血管层次信息（包括变异情况）
        self.vessel_hierarchy = {
            # 一级：主肺动脉
            'MPA': {'level': 0, 'parent': None, 'expected_class_range': [0, 1, 2, 3]},
            
            # 二级：左右肺动脉
            'LPA': {'level': 1, 'parent': 'MPA', 'expected_class_range': [1, 2, 3]},
            'RPA': {'level': 1, 'parent': 'MPA', 'expected_class_range': [1, 2, 3]},
            
            # 三级：上叶、段间、内侧、中叶、下叶分支（包括变异）
            'Lupper': {'level': 2, 'parent': 'LPA', 'expected_class_range': [4, 5, 6, 7]},
            'Rupper': {'level': 2, 'parent': 'RPA', 'expected_class_range': [4, 5, 6, 7]},
            'L1+2': {'level': 2, 'parent': 'LPA', 'expected_class_range': [4, 5, 6, 7]},      # 左上叶变异
            'R1+2': {'level': 2, 'parent': 'RPA', 'expected_class_range': [4, 5, 6, 7]},      # 右上叶变异
            'L1+3': {'level': 2, 'parent': 'LPA', 'expected_class_range': [4, 5, 6, 7]},      # 左上叶变异
            'R1+3': {'level': 2, 'parent': 'RPA', 'expected_class_range': [4, 5, 6, 7]},      # 右上叶变异
            'Linternal': {'level': 2, 'parent': 'LPA', 'expected_class_range': [8, 9, 10, 11]},
            'Rinternal': {'level': 2, 'parent': 'RPA', 'expected_class_range': [8, 9, 10, 11]},
            'Lmedium': {'level': 2, 'parent': 'LPA', 'expected_class_range': [12]},         # 左中叶（变异）
            'Rmedium': {'level': 2, 'parent': 'RPA', 'expected_class_range': [12]},         # 右中叶
            'Ldown': {'level': 2, 'parent': 'LPA', 'expected_class_range': [13, 14]},
            'RDown': {'level': 2, 'parent': 'RPA', 'expected_class_range': [13, 14]}
        }
        
        # 血管类型嵌入（更新维度以适应更多血管类型）
        self.vessel_type_embedding = nn.Embedding(len(self.vessel_hierarchy) + 1, 32).to(self.device)  # +1 for unknown vessels
        
        # 🔧 血管感知训练权重配置
        self.vessel_consistency_weight = getattr(args, 'vessel_consistency_weight', 0.1)
        self.spatial_consistency_weight = getattr(args, 'spatial_consistency_weight', 0.05)
        self.enable_vessel_aware = getattr(args, 'enable_vessel_aware', True)
        
        # 初始化增强训练工具
        self.enhanced_trainer = None
        if any([args.enable_graph_completion, args.enable_visualization, 
                args.save_confusion_matrix, args.save_training_curves]):
            try:
                from enhanced_training_utils import create_enhanced_trainer
                self.enhanced_trainer = create_enhanced_trainer(args.visualization_dir)
                print("🧠 增强训练功能已启用")
            except ImportError as e:
                print(f"⚠️ 无法导入增强训练工具: {e}")
                print("   部分可视化功能可能不可用")
        
        # 训练历史记录（用于可视化）
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        
        # 训练开始时间
        self.start_time = time.time()
        
        # 初始化模型
        self._setup_model()
        
        # 加载数据
        self._setup_data()
        
        # 设置训练参数
        self._setup_training()
    
    def _setup_model(self):
        """初始化CPR-TaG-Net模型"""
        print("🔧 Setting up CPR-TaG-Net model...")
        
        # 🔧 计算增强后的特征维度
        # 原始特征(54) + 血管类型嵌入(32) + 层次编码(3) + 血管内位置(1) = 90
        enhanced_feature_dim = 54 + 32 + 3 + 1
        
        # CPR-TaG-Net 配置参数
        model_config = {
            'num_classes': 15,  # 实际数据中有0-14共15个类别（包括背景类0）
            'node_feature_dim': enhanced_feature_dim,  # 增强后的节点特征维度
            'image_channels': 1,  # 图像通道数
        }
        
        self.model = CPRTaGNet(**model_config).to(self.device)
        print(f"✅ CPR-TaG-Net initialized with enhanced features ({enhanced_feature_dim}D)")
        print(f"✅ Model has {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"🩸 血管层次结构: {len(self.vessel_hierarchy)} 种血管类型")
        
    def _setup_data(self):
        """设置数据加载器 - 24GB显存优化版本"""
        print("📊 Loading and preparing data...")
        
        # 加载预处理数据
        data_list = load_processed_data(self.args.data_dir)
        print(f"✅ Loaded {len(data_list)} cases")
        
        # 根据显存情况筛选数据
        if self.args.enable_large_cases:
            # 24GB显存：使用大部分数据
            filtered_data = []
            for data in data_list:
                nodes = len(data['node_features'])
                if nodes < self.args.max_nodes_per_case:
                    filtered_data.append(data)
                    print(f"  ✅ {data['case_id']}: {nodes} nodes")
                else:
                    print(f"  ⚠️  {data['case_id']}: {nodes} nodes (跳过，超过{self.args.max_nodes_per_case})")
            
            print(f"📊 Using {len(filtered_data)} cases (24GB显存优化)")
        else:
            # 保守模式：只使用小数据集
            filtered_data = []
            for data in data_list:
                if len(data['node_features']) < self.args.max_nodes:
                    filtered_data.append(data)
                    print(f"  ✅ {data['case_id']}: {len(data['node_features'])} nodes")
            
            print(f"📊 Selected {len(filtered_data)} small cases for training")
        
        if len(filtered_data) == 0:
            print("❌ No suitable cases found, using first 3 cases")
            filtered_data = data_list[:3]
        
        # 创建数据分割
        train_data, val_data, test_data = create_data_splits(
            filtered_data, 
            train_ratio=0.7, 
            val_ratio=0.15, 
            test_ratio=0.15,
            random_state=42
        )
        
        # 保存为类属性
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        
        print(f"✅ Data splits - Train: {len(self.train_data)}, Val: {len(self.val_data)}, Test: {len(self.test_data)}")
        
        # 显存使用预估
        total_train_nodes = sum(len(d['node_features']) for d in self.train_data)
        estimated_memory_gb = total_train_nodes * 0.25 / 1024  # 每节点约0.25MB
        print(f"💾 Estimated training memory: {estimated_memory_gb:.1f}GB")
        
    def _setup_training(self):
        """设置训练参数"""
        print("🔧 Setting up training parameters...")
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.args.step_size,
            gamma=self.args.gamma
        )
        
        print("✅ Training setup completed")
    
    def train_on_case(self, case_data, epoch, case_idx):
        """改进的血管感知训练方法 - 充分利用血管连接前置信息"""
        self.model.train()
        
        case_id = case_data['case_id']
        vessel_ranges = case_data['vessel_node_ranges']
        
        # 准备数据
        node_features = torch.FloatTensor(case_data['node_features']).to(self.device)
        node_positions = torch.FloatTensor(case_data['node_positions']).to(self.device)
        edge_index = torch.LongTensor(case_data['edge_index']).to(self.device)
        image_cubes = torch.FloatTensor(case_data['image_cubes']).to(self.device)
        node_classes = torch.LongTensor(case_data['node_classes']).to(self.device)
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        # 🔧 关键改进1: 按血管层次顺序训练
        vessel_order = self._get_hierarchical_vessel_order(vessel_ranges)
        
        for vessel_name in vessel_order:
            if vessel_name not in vessel_ranges:
                continue
                
            start, end = vessel_ranges[vessel_name]
            vessel_node_indices = torch.arange(start, end + 1, device=self.device)
            
            # 🔧 关键改进2: 血管内批处理，保持连续性
            vessel_batches = self._create_vessel_batches(vessel_node_indices, max_batch_size=200)
            
            for batch_idx, batch_indices in enumerate(vessel_batches):
                try:
                    # 准备批次数据
                    batch_node_features = node_features[batch_indices]
                    batch_node_positions = node_positions[batch_indices]
                    batch_image_cubes = image_cubes[batch_indices]
                    batch_node_classes = node_classes[batch_indices]
                    
                    # 🔧 关键改进3: 注入血管先验信息
                    enhanced_features = self._inject_vessel_context(
                        batch_node_features, vessel_name, batch_indices, vessel_ranges
                    )
                    
                    # 🔧 关键改进4: 获取完整边连接（血管内+血管间）
                    batch_edge_index = self._get_complete_vessel_edges(
                        edge_index, batch_indices, vessel_ranges, vessel_name
                    )
                    
                    # 前向传播
                    self.optimizer.zero_grad()
                    outputs = self.model(
                        enhanced_features,
                        batch_node_positions,
                        batch_edge_index,
                        batch_image_cubes
                    )
                    
                    # 🔧 关键改进5: 层次化损失函数
                    loss = self._compute_hierarchical_loss(
                        outputs, batch_node_classes, vessel_name, batch_indices
                    )
                    
                    # 反向传播
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    
                    # 统计
                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total_correct += predicted.eq(batch_node_classes).sum().item()
                    total_samples += batch_node_classes.size(0)
                    
                    # 记录到日志
                    if batch_idx % 10 == 0:
                        global_step = epoch * 1000 + case_idx * 100 + batch_idx
                        self.logger.add_scalar('Train/BatchLoss', loss.item(), global_step)
                    
                    # 内存清理
                    del enhanced_features, batch_edge_index, outputs
                    torch.cuda.empty_cache()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        print(f"⚠️  {case_id} OOM in vessel {vessel_name}, skipping batch {batch_idx}")
                        continue
                    else:
                        print(f"⚠️  Error in {case_id}, vessel {vessel_name}: {e}")
                        continue
        
        # 清理
        del node_features, node_positions, edge_index, image_cubes, node_classes
        torch.cuda.empty_cache()
        
        avg_loss = total_loss / max(1, len(vessel_order))
        accuracy = 100.0 * total_correct / max(1, total_samples)
        
        return avg_loss, accuracy, total_samples
    
    def _get_hierarchical_vessel_order(self, vessel_ranges):
        """获取血管层次训练顺序"""
        available_vessels = list(vessel_ranges.keys())
        
        # 按层次排序：主干 → 分支
        ordered_vessels = []
        for level in range(4):  # 0-3级
            level_vessels = [
                vessel for vessel in available_vessels 
                if vessel in self.vessel_hierarchy and 
                self.vessel_hierarchy[vessel]['level'] == level
            ]
            # 同级内按字母排序保证一致性
            ordered_vessels.extend(sorted(level_vessels))
        
        # 添加未在层次中的血管
        remaining = [v for v in available_vessels if v not in ordered_vessels]
        ordered_vessels.extend(sorted(remaining))
        
        return ordered_vessels
    
    def _create_vessel_batches(self, vessel_indices, max_batch_size=200):
        """创建血管内批次，保持空间连续性"""
        if len(vessel_indices) <= max_batch_size:
            return [vessel_indices]
        
        # 按空间位置排序，保持连续性
        batches = []
        for i in range(0, len(vessel_indices), max_batch_size):
            end_idx = min(i + max_batch_size, len(vessel_indices))
            batches.append(vessel_indices[i:end_idx])
        
        return batches
    
    def _inject_vessel_context(self, node_features, vessel_name, batch_indices, vessel_ranges):
        """注入血管上下文信息"""
        batch_size = node_features.shape[0]
        
        # 1. 血管类型嵌入
        vessel_id = self._get_vessel_type_id(vessel_name)
        vessel_embedding = self.vessel_type_embedding(torch.tensor(vessel_id, device=self.device))
        vessel_emb_expanded = vessel_embedding.unsqueeze(0).expand(batch_size, -1)
        
        # 2. 层次位置编码
        hierarchy_encoding = self._compute_hierarchy_encoding(vessel_name, batch_size)
        
        # 3. 血管内位置编码
        intra_vessel_encoding = self._compute_intra_vessel_position(
            batch_indices, vessel_ranges[vessel_name], batch_size
        )
        
        # 4. 特征融合
        enhanced_features = torch.cat([
            node_features,
            vessel_emb_expanded,
            hierarchy_encoding,
            intra_vessel_encoding
        ], dim=1)
        
        return enhanced_features
    
    def _get_complete_vessel_edges(self, edge_index, batch_indices, vessel_ranges, current_vessel):
        """获取完整的血管边连接信息"""
        device = edge_index.device
        
        if edge_index.shape[1] == 0:
            return torch.zeros((2, 0), dtype=torch.long, device=device)
        
        # 1. 批内边（血管内连接）
        src_in_batch = torch.isin(edge_index[0], batch_indices)
        dst_in_batch = torch.isin(edge_index[1], batch_indices)
        intra_batch_mask = src_in_batch & dst_in_batch
        intra_edges = edge_index[:, intra_batch_mask]
        
        # 2. 血管间连接（关键改进！）
        inter_vessel_edges = self._get_inter_vessel_connections(
            edge_index, batch_indices, vessel_ranges, current_vessel
        )
        
        # 3. 合并边信息
        if inter_vessel_edges.shape[1] > 0:
            all_edges = torch.cat([intra_edges, inter_vessel_edges], dim=1)
        else:
            all_edges = intra_edges
        
        # 4. 重新索引
        if all_edges.shape[1] > 0:
            return self._reindex_edges_for_batch(all_edges, batch_indices)
        else:
            return torch.zeros((2, 0), dtype=torch.long, device=device)
    
    def _get_inter_vessel_connections(self, edge_index, batch_indices, vessel_ranges, current_vessel):
        """获取血管间的关键连接"""
        device = edge_index.device
        
        # 获取当前血管的父血管和子血管
        parent_vessel = self.vessel_hierarchy.get(current_vessel, {}).get('parent')
        child_vessels = [
            v for v, info in self.vessel_hierarchy.items() 
            if info.get('parent') == current_vessel
        ]
        
        relevant_vessels = []
        if parent_vessel and parent_vessel in vessel_ranges:
            relevant_vessels.append(parent_vessel)
        for child in child_vessels:
            if child in vessel_ranges:
                relevant_vessels.append(child)
        
        if not relevant_vessels:
            return torch.zeros((2, 0), dtype=torch.long, device=device)
        
        # 收集相关血管的节点
        relevant_nodes = []
        for vessel in relevant_vessels:
            start, end = vessel_ranges[vessel]
            relevant_nodes.extend(range(start, end + 1))
        
        if not relevant_nodes:
            return torch.zeros((2, 0), dtype=torch.long, device=device)
        
        relevant_nodes_tensor = torch.tensor(relevant_nodes, device=device)
        
        # 找到跨血管的边连接
        src_in_batch = torch.isin(edge_index[0], batch_indices)
        dst_in_relevant = torch.isin(edge_index[1], relevant_nodes_tensor)
        src_in_relevant = torch.isin(edge_index[0], relevant_nodes_tensor)
        dst_in_batch = torch.isin(edge_index[1], batch_indices)
        
        # 双向连接：batch->relevant 或 relevant->batch
        inter_mask = (src_in_batch & dst_in_relevant) | (src_in_relevant & dst_in_batch)
        
        return edge_index[:, inter_mask]
    
    def _compute_hierarchical_loss(self, outputs, targets, vessel_name, batch_indices):
        """计算层次化损失函数"""
        # 1. 基础交叉熵损失
        ce_loss = F.cross_entropy(outputs, targets)
        
        # 2. 血管类型一致性损失
        vessel_consistency_loss = self._compute_vessel_consistency_loss(
            outputs, targets, vessel_name
        )
        
        # 3. 空间连续性损失
        spatial_consistency_loss = self._compute_spatial_consistency_loss(
            outputs, batch_indices
        )
        
        # 权重组合
        total_loss = (ce_loss + 
                     self.vessel_consistency_weight * vessel_consistency_loss + 
                     self.spatial_consistency_weight * spatial_consistency_loss)
        
        return total_loss
    
    def _get_vessel_type_id(self, vessel_name):
        """获取血管类型ID"""
        vessel_names = list(self.vessel_hierarchy.keys())
        if vessel_name in vessel_names:
            return vessel_names.index(vessel_name)
        else:
            return len(vessel_names)  # 未知血管类型
    
    def _compute_hierarchy_encoding(self, vessel_name, batch_size):
        """计算层次位置编码"""
        if vessel_name in self.vessel_hierarchy:
            level = self.vessel_hierarchy[vessel_name]['level']
        else:
            level = -1  # 未知层次
        
        # 简单的位置编码（3级：0-主干，1-左右分支，2-末端分支）
        encoding = torch.zeros(batch_size, 3, device=self.device)  # 3层
        if 0 <= level < 3:
            encoding[:, level] = 1.0
        
        return encoding
    
    def _compute_intra_vessel_position(self, batch_indices, vessel_range, batch_size):
        """计算血管内位置编码"""
        start, end = vessel_range
        vessel_length = end - start + 1
        
        # 归一化位置
        positions = (batch_indices - start).float() / max(1, vessel_length - 1)
        position_encoding = positions.unsqueeze(1)  # [batch_size, 1]
        
        return position_encoding
    
    def _compute_vessel_consistency_loss(self, outputs, targets, vessel_name):
        """血管类型一致性损失"""
        if vessel_name not in self.vessel_hierarchy:
            return torch.tensor(0.0, device=outputs.device)
        
        expected_classes = self.vessel_hierarchy[vessel_name]['expected_class_range']
        
        # 预测概率
        probs = F.softmax(outputs, dim=1)
        
        # 期望类别的概率和
        expected_prob = probs[:, expected_classes].sum(dim=1)
        
        # 一致性损失：鼓励预测在期望范围内
        consistency_loss = -torch.log(expected_prob + 1e-8).mean()
        
        return consistency_loss
    
    def _compute_spatial_consistency_loss(self, outputs, batch_indices):
        """空间连续性损失（相邻节点预测应该相似）"""
        if len(batch_indices) < 2:
            return torch.tensor(0.0, device=outputs.device)
        
        # 相邻节点的预测应该相似
        pred_probs = F.softmax(outputs, dim=1)
        
        # 计算相邻预测的差异
        neighbor_diff = torch.abs(pred_probs[1:] - pred_probs[:-1]).sum(dim=1)
        
        # 连续性损失
        continuity_loss = neighbor_diff.mean()
        
        return continuity_loss
    
    def _reindex_edges_for_batch(self, edges, batch_indices):
        """为批次重新索引边"""
        device = edges.device
        
        # 创建索引映射
        max_idx = max(edges.max().item(), batch_indices.max().item())
        old_to_new = torch.full((max_idx + 1,), -1, device=device, dtype=torch.long)
        old_to_new[batch_indices] = torch.arange(len(batch_indices), device=device)
        
        # 重新索引
        new_edges = edges.clone()
        new_edges[0] = old_to_new[edges[0]]
        new_edges[1] = old_to_new[edges[1]]
        
        # 过滤无效边
        valid_mask = (new_edges[0] >= 0) & (new_edges[1] >= 0)
        valid_edges = new_edges[:, valid_mask]
        
        return valid_edges
    
    def _prepare_batch_edges(self, edge_index, batch_indices, batch_size):
        """为批处理准备边连接 - 保留原方法作为备用"""
        device = edge_index.device
        
        # 如果没有边，返回空边索引
        if edge_index.shape[1] == 0:
            return torch.zeros((2, 0), dtype=torch.long, device=device)
        
        # 创建索引映射
        max_idx = max(edge_index.max().item(), batch_indices.max().item())
        old_to_new = torch.full((max_idx + 1,), -1, device=device)
        old_to_new[batch_indices] = torch.arange(len(batch_indices), device=device)
        
        # 找到批内的边
        src_in_batch = torch.isin(edge_index[0], batch_indices)
        dst_in_batch = torch.isin(edge_index[1], batch_indices)
        edge_mask = src_in_batch & dst_in_batch
        
        if edge_mask.sum() == 0:
            # 没有批内边，返回空边索引
            return torch.zeros((2, 0), dtype=torch.long, device=device)
        
        # 获取批内边并重新索引
        batch_edges = edge_index[:, edge_mask]
        
        # 安全地重新索引
        try:
            batch_edges[0] = old_to_new[batch_edges[0]]
            batch_edges[1] = old_to_new[batch_edges[1]]
            
            # 检查是否有无效的索引
            if (batch_edges < 0).any() or (batch_edges >= len(batch_indices)).any():
                print(f"⚠️  Invalid edge indices detected, returning empty edge index")
                return torch.zeros((2, 0), dtype=torch.long, device=device)
            
            return batch_edges
            
        except Exception as e:
            print(f"⚠️  Error in edge preparation: {e}, returning empty edge index")
            return torch.zeros((2, 0), dtype=torch.long, device=device)

    def train_epoch(self, epoch):
        """训练一个epoch - 基于case的批处理"""
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{self.args.epochs}")
        print(f"{'='*50}")
        
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0
        
        # 随机打乱训练数据
        import random
        train_indices = list(range(len(self.train_data)))
        random.shuffle(train_indices)
        
        pbar = tqdm(train_indices, desc='Training Cases')
        for i, case_idx in enumerate(pbar):
            case_data = self.train_data[case_idx]
            case_id = case_data['case_id']
            num_nodes = len(case_data['node_features'])
            
            try:
                loss, acc, samples = self.train_on_case(case_data, epoch, i)
                
                epoch_loss += loss * samples
                epoch_correct += acc * samples / 100.0
                epoch_samples += samples
                
                current_acc = 100.0 * epoch_correct / epoch_samples if epoch_samples > 0 else 0.0
                
                pbar.set_postfix({
                    'Case': case_id,
                    'Nodes': num_nodes,
                    'Loss': f'{loss:.3f}',
                    'Acc': f'{current_acc:.1f}%',
                    'GPU': f'{torch.cuda.memory_allocated()/1024**3:.1f}GB'
                })
                
            except Exception as e:
                print(f"❌ Error training on {case_id}: {e}")
                continue
        
        avg_loss = epoch_loss / epoch_samples if epoch_samples > 0 else 0.0
        avg_acc = 100.0 * epoch_correct / epoch_samples if epoch_samples > 0 else 0.0
        
        # 记录到日志
        self.logger.add_scalar('Train/EpochLoss', avg_loss, epoch)
        self.logger.add_scalar('Train/EpochAccuracy', avg_acc, epoch)
        
        return avg_loss, avg_acc
    
    def validate_on_case(self, case_data):
        """改进的血管感知验证方法"""
        self.model.eval()
        
        case_id = case_data['case_id']
        vessel_ranges = case_data['vessel_node_ranges']
        
        with torch.no_grad():
            # 准备数据
            node_features = torch.FloatTensor(case_data['node_features']).to(self.device)
            node_positions = torch.FloatTensor(case_data['node_positions']).to(self.device)
            edge_index = torch.LongTensor(case_data['edge_index']).to(self.device)
            image_cubes = torch.FloatTensor(case_data['image_cubes']).to(self.device)
            node_classes = torch.LongTensor(case_data['node_classes']).to(self.device)
            
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            
            # 🔧 验证时也使用血管感知的方法
            vessel_order = self._get_hierarchical_vessel_order(vessel_ranges)
            
            for vessel_name in vessel_order:
                if vessel_name not in vessel_ranges:
                    continue
                    
                start, end = vessel_ranges[vessel_name]
                vessel_node_indices = torch.arange(start, end + 1, device=self.device)
                
                vessel_batches = self._create_vessel_batches(vessel_node_indices, max_batch_size=300)
                
                for batch_idx, batch_indices in enumerate(vessel_batches):
                    try:
                        # 准备批次数据
                        batch_node_features = node_features[batch_indices]
                        batch_node_positions = node_positions[batch_indices]
                        batch_image_cubes = image_cubes[batch_indices]
                        batch_node_classes = node_classes[batch_indices]
                        
                        # 注入血管上下文信息
                        enhanced_features = self._inject_vessel_context(
                            batch_node_features, vessel_name, batch_indices, vessel_ranges
                        )
                        
                        # 获取完整边连接
                        batch_edge_index = self._get_complete_vessel_edges(
                            edge_index, batch_indices, vessel_ranges, vessel_name
                        )
                        
                        # 前向传播
                        outputs = self.model(
                            enhanced_features,
                            batch_node_positions,
                            batch_edge_index,
                            batch_image_cubes
                        )
                        
                        # 计算损失（验证时只用基础损失）
                        loss = F.cross_entropy(outputs, batch_node_classes)
                        
                        # 统计
                        total_loss += loss.item()
                        _, predicted = outputs.max(1)
                        total_correct += predicted.eq(batch_node_classes).sum().item()
                        total_samples += batch_node_classes.size(0)
                        
                        # 清理内存
                        del enhanced_features, batch_edge_index, outputs
                        torch.cuda.empty_cache()
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            torch.cuda.empty_cache()
                            print(f"⚠️  Validation {case_id} OOM in vessel {vessel_name}, skipping batch {batch_idx}")
                            continue
                        else:
                            print(f"⚠️  Validation error in {case_id}, vessel {vessel_name}: {e}")
                            continue
            
            # 清理
            del node_features, node_positions, edge_index, image_cubes, node_classes
            torch.cuda.empty_cache()
            
            avg_loss = total_loss / max(1, len(vessel_order))
            accuracy = 100.0 * total_correct / max(1, total_samples)
            
            return avg_loss, accuracy, total_samples

    def validate(self, epoch):
        """验证一个epoch"""
        if not self.val_data:
            return 0.0, 0.0
        
        print("\n🔍 Validating...")
        
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0
        
        pbar = tqdm(self.val_data, desc='Validation Cases')
        for case_data in pbar:
            case_id = case_data['case_id']
            num_nodes = len(case_data['node_features'])
            
            try:
                loss, acc, samples = self.validate_on_case(case_data)
                
                epoch_loss += loss * samples
                epoch_correct += acc * samples / 100.0
                epoch_samples += samples
                
                current_acc = 100.0 * epoch_correct / epoch_samples if epoch_samples > 0 else 0.0
                
                pbar.set_postfix({
                    'Case': case_id,
                    'Nodes': num_nodes,
                    'Loss': f'{loss:.3f}',
                    'Acc': f'{current_acc:.1f}%'
                })
                
            except Exception as e:
                print(f"❌ Error validating on {case_id}: {e}")
                continue
        
        avg_loss = epoch_loss / epoch_samples if epoch_samples > 0 else 0.0
        avg_acc = 100.0 * epoch_correct / epoch_samples if epoch_samples > 0 else 0.0
        
        # 记录到日志
        self.logger.add_scalar('Val/EpochLoss', avg_loss, epoch)
        self.logger.add_scalar('Val/EpochAccuracy', avg_acc, epoch)
        
        return avg_loss, avg_acc
    
    def _generate_confusion_matrix(self, epoch):
        """生成验证集的混淆矩阵"""
        if not self.val_data or not self.enhanced_trainer:
            return
        
        print("📊 收集验证数据用于混淆矩阵...")
        all_preds = []
        all_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for case_data in self.val_data:
                try:
                    # 获取案例数据
                    node_features = torch.tensor(case_data['node_features'], dtype=torch.float32).to(self.device)
                    node_positions = torch.tensor(case_data['node_positions'], dtype=torch.float32).to(self.device)
                    edge_index = torch.tensor(case_data['edge_index'], dtype=torch.long).to(self.device)
                    image_cubes = torch.tensor(case_data['image_cubes'], dtype=torch.float32).to(self.device)  # [N, 32, 32, 32]
                    labels = torch.tensor(case_data['node_classes'], dtype=torch.long).to(self.device)
                    
                    # 模型预测 - 使用CPR-TaG-Net的完整参数
                    logits = self.model(node_features, node_positions, edge_index, image_cubes)
                    
                    preds = torch.argmax(logits, dim=1)
                    
                    # 图形补全（如果启用）
                    if self.args.enable_graph_completion and node_positions is not None:
                        try:
                            refined_preds, _ = self.enhanced_trainer.complete_graph(
                                node_positions, preds, distance_threshold=5.0
                            )
                            preds = refined_preds
                        except Exception as e:
                            print(f"   ⚠️ 图形补全失败: {e}")
                    
                    all_preds.append(preds.cpu())
                    all_labels.append(labels.cpu())
                    
                except Exception as e:
                    print(f"   ⚠️ 处理案例 {case_data.get('case_id', 'unknown')} 失败: {e}")
                    continue
        
        if all_preds:
            # 合并所有预测和标签
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            
            # 生成混淆矩阵
            self.enhanced_trainer.plot_confusion_matrix(all_labels, all_preds, epoch)
            
            # 分析预测质量
            analysis = self.enhanced_trainer.analyze_prediction_quality(all_labels, all_preds)
            self.enhanced_trainer.save_analysis_report(analysis, epoch)
        else:
            print("   ⚠️ 无法收集到验证数据")
    
    def save_checkpoint(self, epoch, train_loss, val_loss, val_acc, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
        }
        
        # 保存最新检查点
        checkpoint_path = os.path.join(self.args.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.args.checkpoint_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            print(f"💾 Best model saved at epoch {epoch+1}")
    
    def train(self):
        """开始训练"""
        print("🚀 Starting CPR-TaG-Net training...")
        
        # 记录训练开始信息
        model_info = f"CPR-TaG-Net ({sum(p.numel() for p in self.model.parameters()):,} 参数)"
        data_info = f"训练集: {len(self.train_data)} 案例, 验证集: {len(self.val_data) if self.val_data else 0} 案例"
        config_info = f"Epochs: {self.args.epochs}, Learning Rate: {self.args.learning_rate}, Batch Size: {self.args.node_batch_size}"
        
        self.logger.log_training_start(model_info, data_info, config_info)
        
        # 显示增强功能状态
        if self.enhanced_trainer:
            print("🧠 增强功能已启用:")
            enhanced_features = []
            if self.args.enable_graph_completion:
                enhanced_features.append("图形补全")
                print("  ✅ 图形补全")
            if self.args.enable_visualization:
                enhanced_features.append("训练可视化")
                print("  ✅ 训练可视化")
            if self.args.save_confusion_matrix:
                enhanced_features.append("混淆矩阵")
                print("  ✅ 混淆矩阵")
            if self.args.save_training_curves:
                enhanced_features.append("训练曲线")
                print("  ✅ 训练曲线")
            
            self.logger.log_message(f"增强功能已启用: {', '.join(enhanced_features)}")
        
        best_val_acc = 0.0
        best_epoch = 0
        
        for epoch in range(self.args.epochs):
            epoch_start_time = time.time()
            
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_acc = self.validate(epoch)
            
            # 记录训练历史（用于可视化）
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            if val_loss is not None:
                self.val_losses.append(val_loss)
                self.val_accs.append(val_acc)
            
            # 更新学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()
            new_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录学习率变化
            self.logger.add_scalar('Train/LearningRate', new_lr, epoch)
            
            # 计算epoch用时
            epoch_time = time.time() - epoch_start_time
            
            # 打印结果
            print(f"\n📊 Epoch {epoch+1} Results:")
            print(f"  Training   - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
            if self.val_data:
                print(f"  Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
            print(f"  Learning Rate: {new_lr:.6f}")
            print(f"  Epoch Time: {epoch_time:.1f}s")
            
            # 记录详细的epoch信息到日志
            extra_info = f"Epoch用时: {epoch_time:.1f}s"
            if current_lr != new_lr:
                extra_info += f", 学习率变化: {current_lr:.6f} -> {new_lr:.6f}"
            
            self.logger.log_epoch_summary(
                epoch + 1, train_loss, train_acc, 
                val_loss if val_loss is not None else 0.0, 
                val_acc if val_acc is not None else 0.0, 
                new_lr, extra_info
            )
            
            # 增强功能：生成训练进度可视化
            if self.enhanced_trainer and self.args.save_training_curves and (epoch + 1) % 5 == 0:
                try:
                    self.enhanced_trainer.visualize_training_progress(
                        self.train_losses, self.train_accs, 
                        self.val_losses if self.val_losses else None,
                        self.val_accs if self.val_accs else None,
                        epoch + 1
                    )
                    self.logger.log_message(f"生成训练进度可视化 (Epoch {epoch + 1})")
                except Exception as e:
                    print(f"   ⚠️ 训练进度可视化失败: {e}")
                    self.logger.log_message(f"训练进度可视化失败: {e}", "WARNING")
            
            # 保存检查点
            is_best = val_acc > best_val_acc if val_acc is not None else False
            if is_best:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                self.logger.log_message(f"新的最佳验证准确率: {best_val_acc:.4f}% (Epoch {best_epoch})")
            
            if epoch % self.args.save_freq == 0 or is_best or epoch == self.args.epochs - 1:
                self.save_checkpoint(epoch, train_loss, val_loss, val_acc, is_best)
                
                # # 增强功能：在最佳模型时生成混淆矩阵
                # if is_best and self.enhanced_trainer and self.args.save_confusion_matrix:
                #     try:
                #         print("   📊 生成最佳模型混淆矩阵...")
                #         self.logger.log_message("生成最佳模型混淆矩阵")
                #         self._generate_confusion_matrix(epoch + 1)
                #     except Exception as e:
                #         print(f"   ⚠️ 混淆矩阵生成失败: {e}")
                #         self.logger.log_message(f"混淆矩阵生成失败: {e}", "WARNING")
        
        # 计算总训练时间
        total_time = (time.time() - self.start_time) / 60  # 转换为分钟
        
        print(f"\n🎉 Training completed! Best validation accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
        print(f"⏱️ Total training time: {total_time:.2f} minutes")
        
        # 最终可视化
        if self.enhanced_trainer:
            if self.args.save_training_curves:
                try:
                    print("📈 生成最终训练曲线...")
                    self.enhanced_trainer.visualize_training_progress(
                        self.train_losses, self.train_accs, 
                        self.val_losses if self.val_losses else None,
                        self.val_accs if self.val_accs else None
                    )
                    self.logger.log_message("生成最终训练曲线")
                except Exception as e:
                    print(f"⚠️ 最终训练曲线生成失败: {e}")
                    self.logger.log_message(f"最终训练曲线生成失败: {e}", "WARNING")
        
        # 记录训练结束
        self.logger.log_training_end(best_epoch, best_val_acc, total_time)
        self.logger.close()
        
        print(f"📝 详细日志已保存到: {self.logger.get_experiment_dir()}")

def main():
    parser = argparse.ArgumentParser(description='CPR-TaG-Net血管分类模型训练')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='/home/lihe/classify/lungmap/data/processed',
                       help='预处理数据目录')
    parser.add_argument('--max_nodes', type=int, default=1000, 
                       help='最大节点数限制（过滤大案例）')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--node_batch_size', type=int, default=500, help='节点批大小')  # 24GB显存可以更大
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--step_size', type=int, default=20, help='学习率衰减步长')
    parser.add_argument('--gamma', type=float, default=0.5, help='学习率衰减因子')
    
    # 24GB显存优化参数
    parser.add_argument('--max_nodes_per_case', type=int, default=8000, 
                       help='单案例最大节点数（24GB显存优化）')
    parser.add_argument('--enable_large_cases', action='store_true', 
                       help='启用大案例训练（需要24GB显存）')
    
    # 系统参数
    parser.add_argument('--checkpoint_dir', type=str, default='/home/lihe/classify/lungmap/outputs/checkpoints',
                       help='检查点保存目录')
    parser.add_argument('--log_dir', type=str, default='/home/lihe/classify/lungmap/outputs/logs',
                       help='日志保存目录')
    parser.add_argument('--save_freq', type=int, default=5, help='检查点保存频率')
    
    # 增强功能参数
    parser.add_argument('--enable_graph_completion', action='store_true',
                       help='启用图形补全功能')
    parser.add_argument('--enable_visualization', action='store_true',
                       help='启用训练可视化功能')
    parser.add_argument('--save_confusion_matrix', action='store_true',
                       help='保存混淆矩阵')
    parser.add_argument('--save_training_curves', action='store_true',
                       help='保存训练曲线')
    parser.add_argument('--visualization_dir', type=str, default='/home/lihe/classify/lungmap/outputs/visualizations',
                       help='可视化结果保存目录')
    
    # 🔧 血管感知训练参数
    parser.add_argument('--enable_vessel_aware', action='store_true', default=True,
                       help='启用血管感知训练（推荐）')
    parser.add_argument('--vessel_consistency_weight', type=float, default=0.1,
                       help='血管一致性损失权重')
    parser.add_argument('--spatial_consistency_weight', type=float, default=0.05,
                       help='空间连续性损失权重')
    
    args = parser.parse_args()
    
    # 打印配置
    print("🔧 CPR-TaG-Net Training Configuration:")
    print("🧠 血管感知训练改进版 - 充分利用血管连接前置信息")
    print(f"  Data directory: {args.data_dir}")
    if args.enable_large_cases:
        print(f"  24GB显存模式: 启用大案例训练")
        print(f"  Max nodes per case: {args.max_nodes_per_case}")
    else:
        print(f"  保守模式: Max nodes per case: {args.max_nodes}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Node batch size: {args.node_batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # 🔧 血管感知训练配置
    if args.enable_vessel_aware:
        print(f"  🩸 血管感知训练: 启用")
        print(f"    - 血管层次顺序训练")
        print(f"    - 血管间连接保持")
        print(f"    - 血管先验信息注入")
        print(f"    - 层次化损失函数")
        print(f"    - 血管一致性权重: {args.vessel_consistency_weight}")
        print(f"    - 空间连续性权重: {args.spatial_consistency_weight}")
    else:
        print(f"  ⚠️  血管感知训练: 禁用（不推荐）")
    
    # 增强功能配置
    enhanced_features = []
    if args.enable_graph_completion:
        enhanced_features.append("图形补全")
    if args.enable_visualization:
        enhanced_features.append("训练可视化")
    if args.save_confusion_matrix:
        enhanced_features.append("混淆矩阵")
    if args.save_training_curves:
        enhanced_features.append("训练曲线")
    
    if enhanced_features:
        print(f"  🧠 增强功能: {', '.join(enhanced_features)}")
        print(f"  📊 可视化目录: {args.visualization_dir}")
    else:
        print(f"  基础训练模式（无增强功能）")
    
    # GPU显存检查
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU Memory: {gpu_memory:.1f}GB")
        if gpu_memory >= 20 and not args.enable_large_cases:
            print(f"  💡 检测到大显存GPU，建议使用 --enable_large_cases 训练更多数据")
    
    # 开始训练
    trainer = VesselTrainer(args)
    trainer.train()

if __name__ == "__main__":
    main()
