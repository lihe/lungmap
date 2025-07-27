#!/usr/bin/env python3
"""
CPR-TaG-Net 血管分析训练管道
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import argparse
import sys

# 导入数据加载器
from vessel_data_loader import load_processed_data, create_data_splits, create_dataloaders, save_data_splits, load_data_splits

# 导入CPR-TaG-Net模型
sys.path.append('.')
try:
    from data_loader import VesselGraphDataset as OriginalDataset
    print("✅ Found original data_loader.py")
except ImportError:
    print("⚠️  Original data_loader.py not found, using vessel_data_loader")

class VesselTrainer:
    """血管图像分析训练器"""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建实验目录
        self.exp_dir = os.path.join('experiments', config['exp_name'])
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(os.path.join(self.exp_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.exp_dir, 'logs'), exist_ok=True)
        
        # 初始化tensorboard
        self.writer = SummaryWriter(os.path.join(self.exp_dir, 'logs'))
        
        # 保存配置
        with open(os.path.join(self.exp_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"🔧 Experiment directory: {self.exp_dir}")
    
    def setup_model(self):
        """设置模型"""
        # 简化的模型定义，如果找不到原始模型
        try:
            # 尝试导入原始CPR-TaG-Net
            from train_eval import build_model
            self.model = build_model().to(self.device)
            print("✅ Using original CPR-TaG-Net model")
        except ImportError:
            # 如果导入失败，创建一个简化的替代模型
            print("⚠️  Original model not found, creating simplified model")
            self.model = self._create_simplified_model().to(self.device)
        
        # 打印模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"🔧 Model Setup:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Device: {self.device}")
    
    def _create_simplified_model(self):
        """创建简化的模型用于测试"""
        class SimplifiedVesselNet(nn.Module):
            def __init__(self, num_classes=18, node_feature_dim=54):
                super().__init__()
                
                # 图像编码器 (简化版3D CNN)
                self.image_encoder = nn.Sequential(
                    nn.Conv3d(1, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool3d((4, 4, 4)),
                    nn.Conv3d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool3d((1, 1, 1)),
                    nn.Flatten()
                )
                
                # 节点特征编码器
                self.node_encoder = nn.Sequential(
                    nn.Linear(node_feature_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 256)
                )
                
                # 分类器
                self.classifier = nn.Sequential(
                    nn.Linear(64 + 256, 512),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(256, num_classes)
                )
            
            def forward(self, node_features, node_positions, edge_index, image_cubes):
                # 图像特征提取
                batch_size = image_cubes.size(0)
                image_feat = self.image_encoder(image_cubes)  # [N, 64]
                
                # 节点特征编码
                node_feat = self.node_encoder(node_features)  # [N, 256]
                
                # 特征融合
                combined_feat = torch.cat([image_feat, node_feat], dim=1)  # [N, 320]
                
                # 分类
                logits = self.classifier(combined_feat)  # [N, num_classes]
                
                return logits
        
        return SimplifiedVesselNet()
    
    def setup_optimizer_scheduler(self):
        """设置优化器和调度器"""
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config['lr_step_size'],
            gamma=self.config['lr_gamma']
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        
        print(f"🔧 Optimizer: Adam (lr={self.config['learning_rate']})")
    
    def setup_data(self):
        """设置数据"""
        print("🔄 Loading data...")
        
        # 检查是否已有数据分割
        if os.path.exists('data_splits/train_data.pkl'):
            print("Found existing data splits, loading...")
            train_data, val_data, test_data = load_data_splits('data_splits')
        else:
            print("Creating new data splits...")
            data_list = load_processed_data('processed_data')
            if len(data_list) == 0:
                raise ValueError("No processed data found! Please run vessel_preprocessing.py first.")
            
            train_data, val_data, test_data = create_data_splits(data_list)
            save_data_splits(train_data, val_data, test_data, 'data_splits')
        
        # 创建数据加载器
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            train_data, val_data, test_data,
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers']
        )
        
        print(f"📊 Data loaded:")
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Val batches: {len(self.val_loader) if self.val_loader else 0}")
        print(f"  Test batches: {len(self.test_loader) if self.test_loader else 0}")
    
    def train_epoch(self, epoch: int):
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        pbar = tqdm(self.train_loader, desc=f'Train Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # 移动数据到设备
                node_features = batch['node_features'].to(self.device)
                node_positions = batch['node_positions'].to(self.device)
                edge_index = batch['edge_index'].to(self.device)
                image_cubes = batch['image_cubes'].to(self.device)
                node_classes = batch['node_classes'].to(self.device)
                
                # 前向传播
                self.optimizer.zero_grad()
                
                logits = self.model(node_features, node_positions, edge_index, image_cubes)
                loss = self.criterion(logits, node_classes)
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # 统计
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct_predictions += (predictions == node_classes).sum().item()
                total_predictions += node_classes.size(0)
                
                # 更新进度条
                current_acc = correct_predictions / total_predictions
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.4f}'
                })
                
                # 记录到tensorboard
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
                self.writer.add_scalar('Train/BatchAcc', current_acc, global_step)
                
            except Exception as e:
                print(f"❌ Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0
        avg_acc = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        return avg_loss, avg_acc
    
    def validate(self, epoch: int):
        """验证"""
        if not self.val_loader:
            return 0.0, 0.0
            
        self.model.eval()
        
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Val Epoch {epoch}')
            
            for batch in pbar:
                try:
                    # 移动数据到设备
                    node_features = batch['node_features'].to(self.device)
                    node_positions = batch['node_positions'].to(self.device)
                    edge_index = batch['edge_index'].to(self.device)
                    image_cubes = batch['image_cubes'].to(self.device)
                    node_classes = batch['node_classes'].to(self.device)
                    
                    logits = self.model(node_features, node_positions, edge_index, image_cubes)
                    loss = self.criterion(logits, node_classes)
                    
                    total_loss += loss.item()
                    predictions = torch.argmax(logits, dim=1)
                    correct_predictions += (predictions == node_classes).sum().item()
                    total_predictions += node_classes.size(0)
                    
                    pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{correct_predictions/total_predictions:.4f}'
                    })
                    
                except Exception as e:
                    print(f"❌ Error in validation: {e}")
                    continue
        
        avg_loss = total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0
        avg_acc = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        return avg_loss, avg_acc
    
    def save_checkpoint(self, epoch: int, train_loss: float, val_loss: float, val_acc: float, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'config': self.config
        }
        
        # 保存最新检查点
        torch.save(checkpoint, os.path.join(self.exp_dir, 'checkpoints', 'latest.pth'))
        
        # 保存最佳检查点
        if is_best:
            torch.save(checkpoint, os.path.join(self.exp_dir, 'checkpoints', 'best.pth'))
        
        # 定期保存
        if epoch % self.config['save_interval'] == 0:
            torch.save(checkpoint, os.path.join(self.exp_dir, 'checkpoints', f'epoch_{epoch}.pth'))
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint['epoch'], checkpoint.get('val_acc', 0.0)
    
    def train(self):
        """完整训练流程"""
        print("🚀 Starting training...")
        
        best_val_acc = 0.0
        start_epoch = 1
        
        # 如果有检查点，加载它
        latest_checkpoint = os.path.join(self.exp_dir, 'checkpoints', 'latest.pth')
        if os.path.exists(latest_checkpoint) and self.config['resume']:
            print("Resuming from checkpoint...")
            start_epoch, best_val_acc = self.load_checkpoint(latest_checkpoint)
            start_epoch += 1
        
        for epoch in range(start_epoch, self.config['epochs'] + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{self.config['epochs']}")
            print(f"{'='*50}")
            
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_acc = self.validate(epoch)
            
            # 调度器步进
            self.scheduler.step()
            
            # 记录到tensorboard
            self.writer.add_scalar('Epoch/TrainLoss', train_loss, epoch)
            self.writer.add_scalar('Epoch/TrainAcc', train_acc, epoch)
            self.writer.add_scalar('Epoch/ValLoss', val_loss, epoch)
            self.writer.add_scalar('Epoch/ValAcc', val_acc, epoch)
            self.writer.add_scalar('Epoch/LR', self.scheduler.get_last_lr()[0], epoch)
            
            # 打印结果
            print(f"\nEpoch {epoch} Results:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"  Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # 保存检查点
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                print(f"  🎉 New best validation accuracy: {best_val_acc:.4f}")
            
            self.save_checkpoint(epoch, train_loss, val_loss, val_acc, is_best)
        
        print(f"\n🎉 Training completed!")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        
        self.writer.close()

def get_default_config():
    """获取默认配置"""
    return {
        'exp_name': f'vessel_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'epochs': 50,
        'batch_size': 1,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'lr_step_size': 20,
        'lr_gamma': 0.5,
        'num_workers': 2,
        'save_interval': 10,
        'resume': False,
    }

def main():
    parser = argparse.ArgumentParser(description='Train CPR-TaG-Net for vessel analysis')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--exp_name', type=str, help='Experiment name')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # 获取配置
    config = get_default_config()
    
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config.update(json.load(f))
    
    # 命令行参数覆盖
    if args.exp_name:
        config['exp_name'] = args.exp_name
    if args.epochs:
        config['epochs'] = args.epochs
    if args.lr:
        config['learning_rate'] = args.lr
    if args.resume:
        config['resume'] = True
    
    print("🔧 Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 创建训练器并开始训练
    trainer = VesselTrainer(config)
    trainer.setup_model()
    trainer.setup_optimizer_scheduler()
    trainer.setup_data()
    trainer.train()

if __name__ == "__main__":
    main()
