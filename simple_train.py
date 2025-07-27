#!/usr/bin/env python3
"""
简化的血管分类训练脚本 - 处理GPU内存不足的问题
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# 添加模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'training'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'models', 'CPR_TaG_Net'))

from vessel_data_loader import load_processed_data, create_data_splits, create_dataloaders

class SimpleTrainer:
    """简化的训练器 - 处理内存限制"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        
        # 加载数据
        self._load_small_data()
        
    def _load_small_data(self):
        """加载较小的数据进行测试"""
        print("📊 Loading small dataset for testing...")
        
        # 加载所有数据
        data_list = load_processed_data('/home/lihe/classify/lungmap/data/processed')
        
        # 筛选出节点数较少的数据（节点数 < 1000）
        small_data = []
        for data in data_list:
            if len(data['node_features']) < 1000:
                small_data.append(data)
                print(f"  ✅ {data['case_id']}: {len(data['node_features'])} nodes")
        
        print(f"📊 Selected {len(small_data)} small cases for training")
        
        if len(small_data) == 0:
            print("❌ No small cases found, using first 3 cases")
            small_data = data_list[:3]
        
        # 简单分割
        if len(small_data) >= 3:
            self.train_data = small_data[:-2]
            self.val_data = small_data[-2:]
        else:
            self.train_data = small_data
            self.val_data = []
        
        print(f"📊 Train cases: {len(self.train_data)}, Val cases: {len(self.val_data)}")
        
    def create_simple_model(self):
        """创建一个简化的模型用于测试"""
        print("🔧 Creating simplified model...")
        
        class SimpleCNN3D(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv3d(1, 16, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool3d((1, 1, 1)),
                    nn.Flatten()
                )
                
            def forward(self, x):
                return self.conv(x)
        
        class SimpleVesselModel(nn.Module):
            def __init__(self, num_classes=15):
                super().__init__()
                self.num_classes = num_classes
                
                # 简化的图像编码器
                self.image_encoder = SimpleCNN3D()
                
                # 节点特征处理
                self.node_proj = nn.Linear(54, 64)
                
                # 分类器
                self.classifier = nn.Sequential(
                    nn.Linear(64 + 16, 128),  # node_features + image_features
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, num_classes)
                )
                
            def forward(self, node_features, node_positions, edge_index, image_cubes):
                # 处理节点特征
                node_feat = self.node_proj(node_features)  # [N, 64]
                
                # 处理图像特征
                image_feat = self.image_encoder(image_cubes)  # [N, 16]
                
                # 融合特征
                combined_feat = torch.cat([node_feat, image_feat], dim=1)  # [N, 80]
                
                # 分类
                logits = self.classifier(combined_feat)  # [N, num_classes]
                
                return logits
        
        model = SimpleVesselModel().to(self.device)
        print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        return model
    
    def train_on_case(self, model, optimizer, criterion, case_data):
        """在单个case上训练"""
        model.train()
        
        # 准备数据
        node_features = torch.FloatTensor(case_data['node_features']).to(self.device)
        node_positions = torch.FloatTensor(case_data['node_positions']).to(self.device)
        edge_index = torch.LongTensor(case_data['edge_index']).to(self.device)
        image_cubes = torch.FloatTensor(case_data['image_cubes']).unsqueeze(1).to(self.device)  # [N, 1, 32, 32, 32]
        node_classes = torch.LongTensor(case_data['node_classes']).to(self.device)
        
        # 分批处理节点（每次处理100个节点）
        batch_size = 100
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        num_nodes = node_features.shape[0]
        num_batches = (num_nodes + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_nodes)
            
            # 批数据
            batch_node_features = node_features[start_idx:end_idx]
            batch_node_positions = node_positions[start_idx:end_idx]
            batch_image_cubes = image_cubes[start_idx:end_idx]
            batch_node_classes = node_classes[start_idx:end_idx]
            
            # 简化的edge_index（这里不使用图连接）
            batch_edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
            
            try:
                # 前向传播
                optimizer.zero_grad()
                outputs = model(batch_node_features, batch_node_positions, batch_edge_index, batch_image_cubes)
                loss = criterion(outputs, batch_node_classes)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                # 统计
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total_correct += predicted.eq(batch_node_classes).sum().item()
                total_samples += batch_node_classes.size(0)
                
                # 清理内存
                del batch_node_features, batch_node_positions, batch_image_cubes, batch_node_classes, outputs
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"⚠️  Batch {i} OOM, skipping...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0
        
        return avg_loss, accuracy
    
    def test_simple_training(self):
        """测试简化训练流程"""
        print("🚀 Starting simple training test...")
        
        # 创建模型
        model = self.create_simple_model()
        
        # 优化器和损失函数
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # 训练几个epoch
        for epoch in range(3):
            print(f"\n{'='*30}")
            print(f"Epoch {epoch+1}/3")
            print(f"{'='*30}")
            
            total_loss = 0.0
            total_acc = 0.0
            
            for i, case_data in enumerate(self.train_data):
                print(f"Training on case {case_data['case_id']} ({len(case_data['node_features'])} nodes)...")
                
                try:
                    loss, acc = self.train_on_case(model, optimizer, criterion, case_data)
                    total_loss += loss
                    total_acc += acc
                    
                    print(f"  Loss: {loss:.4f}, Accuracy: {acc:.2f}%")
                    
                except Exception as e:
                    print(f"  ❌ Error: {e}")
                    continue
            
            avg_loss = total_loss / len(self.train_data)
            avg_acc = total_acc / len(self.train_data)
            print(f"\nEpoch {epoch+1} - Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_acc:.2f}%")
        
        print("\n✅ Simple training test completed!")

def main():
    trainer = SimpleTrainer()
    trainer.test_simple_training()

if __name__ == "__main__":
    main()
