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
        
        # CPR-TaG-Net 配置参数
        model_config = {
            'num_classes': 15,  # 根据您的数据，有15个类别（0-14）
            'node_feature_dim': 54,  # 节点特征维度
            'image_channels': 1,  # 图像通道数
        }
        
        self.model = CPRTaGNet(**model_config).to(self.device)
        print(f"✅ CPR-TaG-Net initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
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
        """在单个case上训练（分批处理节点）- 适配CPR-TaG-Net，自适应批大小"""
        self.model.train()
        
        case_id = case_data['case_id']
        
        # 准备数据
        node_features = torch.FloatTensor(case_data['node_features']).to(self.device)
        node_positions = torch.FloatTensor(case_data['node_positions']).to(self.device)
        edge_index = torch.LongTensor(case_data['edge_index']).to(self.device)
        image_cubes = torch.FloatTensor(case_data['image_cubes']).unsqueeze(1).to(self.device)  # [N, 1, 32, 32, 32]
        node_classes = torch.LongTensor(case_data['node_classes']).to(self.device)
        
        num_nodes = node_features.shape[0]
        
        # 自适应批大小：根据节点数量调整
        if num_nodes > 5000:
            batch_size = min(200, num_nodes)  # 大案例用小批
        elif num_nodes > 2000:
            batch_size = min(300, num_nodes)  # 中案例用中批
        else:
            batch_size = min(self.args.node_batch_size, num_nodes)  # 小案例用大批
            
        num_batches = (num_nodes + batch_size - 1) // batch_size
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        successful_batches = 0
        
        # 随机打乱节点顺序
        indices = torch.randperm(num_nodes, device=self.device)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_nodes)
            batch_indices = indices[start_idx:end_idx]
            
            # 🔧 改进内存管理：预先清理
            if batch_idx > 0:
                torch.cuda.empty_cache()
            
            # 批数据准备 - 添加错误检查
            try:
                batch_node_features = node_features[batch_indices]
                batch_node_positions = node_positions[batch_indices]
                batch_image_cubes = image_cubes[batch_indices]
                batch_node_classes = node_classes[batch_indices]
                
                # 数据完整性检查
                if batch_node_features.shape[0] == 0:
                    print(f"⚠️  Empty batch at {batch_idx}, skipping...")
                    continue
                    
                # 检查特征维度
                if batch_node_features.shape[1] not in [54, 64]:  # 允许的特征维度
                    print(f"⚠️  Unexpected feature dimension: {batch_node_features.shape}")
                    
            except Exception as e:
                print(f"⚠️  Error preparing batch {batch_idx}: {e}")
                continue
            
            # 为CPR-TaG-Net准备边连接 - 这里简化处理，只保留批内连接
            batch_edge_index = self._prepare_batch_edges(edge_index, batch_indices, batch_size)
            
            try:
                # 前向传播
                self.optimizer.zero_grad()
                
                outputs = self.model(
                    batch_node_features, 
                    batch_node_positions, 
                    batch_edge_index, 
                    batch_image_cubes
                )
                
                loss = self.criterion(outputs, batch_node_classes)
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # 统计
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total_correct += predicted.eq(batch_node_classes).sum().item()
                total_samples += batch_node_classes.size(0)
                successful_batches += 1
                
                # 记录到日志
                if batch_idx % 10 == 0:  # 每10个batch记录一次
                    global_step = epoch * 1000 + batch_idx  # 估算全局步数
                    self.logger.add_scalar('Train/BatchLoss', loss.item(), global_step)
                
                # 清理内存
                del batch_node_features, batch_node_positions, batch_image_cubes, batch_node_classes, outputs
                del batch_edge_index
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # 🔧 改进的OOM处理
                    torch.cuda.empty_cache()
                    if batch_size > 32:  # 降低最小批大小阈值
                        batch_size = max(32, batch_size // 2)
                        print(f"⚠️  {case_id} OOM, reducing batch size to {batch_size}")
                        # 强制内存清理
                        if 'batch_node_features' in locals():
                            del batch_node_features, batch_node_positions, batch_image_cubes, batch_node_classes
                        if 'batch_edge_index' in locals():
                            del batch_edge_index
                        if 'outputs' in locals():
                            del outputs
                        torch.cuda.empty_cache()
                        continue
                    else:
                        print(f"⚠️  {case_id} batch {batch_idx} OOM (min batch size reached), skipping...")
                        # 跳过当前批次，继续下一个
                        torch.cuda.empty_cache()
                        continue
                elif "size of tensor" in str(e) or "mat1 and mat2 shapes cannot be multiplied" in str(e):
                    print(f"⚠️  {case_id} tensor dimension mismatch: {e}")
                    print(f"    Node features: {batch_node_features.shape if 'batch_node_features' in locals() else 'N/A'}")
                    print(f"    Image cubes: {batch_image_cubes.shape if 'batch_image_cubes' in locals() else 'N/A'}")
                    # 清理内存并继续
                    torch.cuda.empty_cache()
                    continue
                else:
                    # 其他错误，重新抛出
                    print(f"⚠️  Unexpected error in {case_id}: {e}")
                    raise e
        
        # 清理case级别的数据
        del node_features, node_positions, edge_index, image_cubes, node_classes
        torch.cuda.empty_cache()
        
        avg_loss = total_loss / successful_batches if successful_batches > 0 else 0.0
        accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0
        
        return avg_loss, accuracy, total_samples
    
    def _prepare_batch_edges(self, edge_index, batch_indices, batch_size):
        """为批处理准备边连接 - 改进版本"""
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
        """在单个case上验证 - 适配CPR-TaG-Net，自适应批大小"""
        self.model.eval()
        
        case_id = case_data['case_id']
        
        with torch.no_grad():
            # 准备数据
            node_features = torch.FloatTensor(case_data['node_features']).to(self.device)
            node_positions = torch.FloatTensor(case_data['node_positions']).to(self.device)
            edge_index = torch.LongTensor(case_data['edge_index']).to(self.device)
            image_cubes = torch.FloatTensor(case_data['image_cubes']).unsqueeze(1).to(self.device)
            node_classes = torch.LongTensor(case_data['node_classes']).to(self.device)
            
            num_nodes = node_features.shape[0]
            
            # 自适应批大小
            if num_nodes > 5000:
                batch_size = min(200, num_nodes)
            elif num_nodes > 2000:
                batch_size = min(300, num_nodes)
            else:
                batch_size = min(self.args.node_batch_size, num_nodes)
                
            num_batches = (num_nodes + batch_size - 1) // batch_size
            
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            successful_batches = 0
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, num_nodes)
                
                # 批数据
                batch_indices = torch.arange(start_idx, end_idx, device=self.device)
                batch_node_features = node_features[batch_indices]
                batch_node_positions = node_positions[batch_indices]
                batch_image_cubes = image_cubes[batch_indices]
                batch_node_classes = node_classes[batch_indices]
                
                # 为CPR-TaG-Net准备边连接
                batch_edge_index = self._prepare_batch_edges(edge_index, batch_indices, batch_size)
                
                try:
                    outputs = self.model(
                        batch_node_features, 
                        batch_node_positions, 
                        batch_edge_index, 
                        batch_image_cubes
                    )
                    
                    loss = self.criterion(outputs, batch_node_classes)
                    
                    # 统计
                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total_correct += predicted.eq(batch_node_classes).sum().item()
                    total_samples += batch_node_classes.size(0)
                    successful_batches += 1
                    
                    # 清理内存
                    del batch_node_features, batch_node_positions, batch_image_cubes, batch_node_classes, outputs
                    del batch_edge_index
                    torch.cuda.empty_cache()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        if batch_size > 50:
                            batch_size = batch_size // 2
                            print(f"⚠️  Validation {case_id} OOM, reducing batch size to {batch_size}")
                            continue
                        else:
                            print(f"⚠️  Validation {case_id} batch {batch_idx} OOM, skipping...")
                            continue
                    elif "size of tensor" in str(e):
                        print(f"⚠️  Validation {case_id} tensor dimension mismatch: {e}")
                        continue
                    else:
                        raise e
            
            # 清理case级别的数据
            del node_features, node_positions, edge_index, image_cubes, node_classes
            torch.cuda.empty_cache()
            
            avg_loss = total_loss / successful_batches if successful_batches > 0 else 0.0
            accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0
            
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
                    image_cubes = torch.tensor(case_data['image_cubes'], dtype=torch.float32).unsqueeze(1).to(self.device)  # [N, 1, 32, 32, 32]
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
                
                # 增强功能：在最佳模型时生成混淆矩阵
                if is_best and self.enhanced_trainer and self.args.save_confusion_matrix:
                    try:
                        print("   📊 生成最佳模型混淆矩阵...")
                        self.logger.log_message("生成最佳模型混淆矩阵")
                        self._generate_confusion_matrix(epoch + 1)
                    except Exception as e:
                        print(f"   ⚠️ 混淆矩阵生成失败: {e}")
                        self.logger.log_message(f"混淆矩阵生成失败: {e}", "WARNING")
        
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
    
    args = parser.parse_args()
    
    # 打印配置
    print("🔧 CPR-TaG-Net Training Configuration:")
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
