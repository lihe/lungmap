"""
训练脚本：整合数据预处理、数据加载和CPR-TaG-Net训练
"""
import os
import sys
import argparse
import time
from tqdm import tqdm

# 添加CPR_TaG_Net目录到路径
sys.path.append('/Users/leslie/IT/feixueguanditu/test/CPR_TaG_Net')

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.tensorboard import SummaryWriter
    from models.cpr_tagnet import CPRTaGNet
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Some functions will be limited.")

from vessel_preprocessing import VesselPreprocessor
if TORCH_AVAILABLE:
    from data_loader import VesselDataModule

class VesselTrainer:
    """血管分类训练器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if TORCH_AVAILABLE else None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.data_module = None
        self.writer = None
        
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        
        if TORCH_AVAILABLE:
            self._setup_logging()
            self._setup_model()
            self._setup_data()
            self._setup_training()
    
    def _setup_logging(self):
        """设置日志"""
        log_dir = os.path.join(self.args.output_dir, 'logs')
        self.writer = SummaryWriter(log_dir)
        print(f"Tensorboard logs will be saved to: {log_dir}")
    
    def _setup_model(self):
        """设置模型"""
        self.model = CPRTaGNet(
            num_classes=self.args.num_classes,
            node_feature_dim=54,
            image_channels=1
        )
        
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
        
        self.model.to(self.device)
        
        # 打印模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    def _setup_data(self):
        """设置数据"""
        self.data_module = VesselDataModule(
            data_dir=self.args.data_dir,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            train_ratio=0.8
        )
        self.data_module.setup()
        
        # 获取类别权重
        class_weights = self.data_module.get_class_weights()
        self.class_weights = class_weights.to(self.device)
        print(f"Class weights: {self.class_weights}")
    
    def _setup_training(self):
        """设置训练组件"""
        # 损失函数
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        
        # 优化器
        if self.args.optimizer == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay
            )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.args.epochs,
            eta_min=self.args.learning_rate * 0.01
        )
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        train_loader = self.data_module.train_dataloader()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            if batch is None:
                continue
            
            batch = batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            logits = self.model(
                x_node=batch.x,
                pos=batch.pos,
                edge_index=batch.edge_index,
                image_cubes=batch.image_cubes
            )
            
            # 计算损失
            loss = self.criterion(logits, batch.y)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += batch.y.size(0)
            correct += (predicted == batch.y).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
            
            # 记录到tensorboard
            global_step = epoch * len(train_loader) + batch_idx
            self.writer.add_scalar('Train/Loss', loss.item(), global_step)
            self.writer.add_scalar('Train/Accuracy', 100.*correct/total, global_step)
            
            if batch_idx % self.args.log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)}] '
                      f'Loss: {loss.item():.6f} Acc: {100.*correct/total:.2f}%')
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, epoch):
        """验证"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        val_loader = self.data_module.val_dataloader()
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                if batch is None:
                    continue
                
                batch = batch.to(self.device)
                
                logits = self.model(
                    x_node=batch.x,
                    pos=batch.pos,
                    edge_index=batch.edge_index,
                    image_cubes=batch.image_cubes
                )
                
                loss = self.criterion(logits, batch.y)
                
                total_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += batch.y.size(0)
                correct += (predicted == batch.y).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        # 记录到tensorboard
        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        self.writer.add_scalar('Val/Accuracy', accuracy, epoch)
        
        print(f'Validation: Average loss: {avg_loss:.4f}, '
              f'Accuracy: {correct}/{total} ({accuracy:.2f}%)')
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch, val_loss, val_acc, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'args': self.args
        }
        
        # 保存最新检查点
        checkpoint_path = os.path.join(self.args.checkpoint_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.args.checkpoint_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best model with validation accuracy: {val_acc:.2f}%")
    
    def train(self):
        """主训练循环"""
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.args.epochs}")
        
        best_val_acc = 0
        
        for epoch in range(1, self.args.epochs + 1):
            print(f"\nEpoch {epoch}/{self.args.epochs}")
            print("-" * 50)
            
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_acc = self.validate(epoch)
            
            # 更新学习率
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            self.writer.add_scalar('Train/LearningRate', current_lr, epoch)
            
            # 保存检查点
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
            
            self.save_checkpoint(epoch, val_loss, val_acc, is_best)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Best Val Acc: {best_val_acc:.2f}%")
        
        print("\nTraining completed!")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        
        self.writer.close()


def run_preprocessing(args):
    """运行数据预处理"""
    print("Starting data preprocessing...")
    
    preprocessor = VesselPreprocessor(
        ct_dir=args.ct_dir,
        label_dir=args.label_dir,
        output_dir=args.data_dir,
        cube_size=32
    )
    
    results = preprocessor.process_all_cases()
    
    print(f"Preprocessing completed! Processed {len(results)} cases.")
    return len(results) > 0


def main():
    parser = argparse.ArgumentParser(description='CPR-TaG-Net Training Pipeline')
    
    # 数据路径
    parser.add_argument('--ct_dir', type=str, default='train',
                       help='CT images directory')
    parser.add_argument('--label_dir', type=str, default='label_filtered',
                       help='Filtered labels directory')
    parser.add_argument('--data_dir', type=str, default='processed_data',
                       help='Processed data directory')
    
    # 输出路径
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Checkpoint directory')
    
    # 模型参数
    parser.add_argument('--num_classes', type=int, default=18,
                       help='Number of vessel classes')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adam', 'adamw'], help='Optimizer')
    
    # 系统参数
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Logging interval')
    
    # 控制参数
    parser.add_argument('--preprocess_only', action='store_true',
                       help='Only run preprocessing')
    parser.add_argument('--skip_preprocessing', action='store_true',
                       help='Skip preprocessing step')
    
    args = parser.parse_args()
    
    print("CPR-TaG-Net Training Pipeline")
    print("=" * 50)
    print(f"CT directory: {args.ct_dir}")
    print(f"Label directory: {args.label_dir}")
    print(f"Data directory: {args.data_dir}")
    print(f"PyTorch available: {TORCH_AVAILABLE}")
    
    # 步骤1：数据预处理
    if not args.skip_preprocessing:
        preprocessing_success = run_preprocessing(args)
        if not preprocessing_success:
            print("Preprocessing failed. Exiting.")
            return
    
    if args.preprocess_only:
        print("Preprocessing completed. Exiting as requested.")
        return
    
    # 步骤2：训练模型
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Cannot proceed with training.")
        return
    
    # 检查处理后的数据是否存在
    if not os.path.exists(args.data_dir) or not os.listdir(args.data_dir):
        print(f"No processed data found in {args.data_dir}. Please run preprocessing first.")
        return
    
    trainer = VesselTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
