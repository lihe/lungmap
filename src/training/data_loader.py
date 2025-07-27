import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from typing import List, Dict, Tuple, Optional
import random

class VesselGraphDataset(Dataset):
    """血管图数据集，用于CPR-TaG-Net训练"""
    
    def __init__(self, 
                 data_dir: str,
                 split: str = 'train',
                 train_ratio: float = 0.8,
                 augment: bool = True,
                 max_nodes: int = 1000):
        """
        Args:
            data_dir: 处理后数据的目录
            split: 'train', 'val', 'test'
            train_ratio: 训练集比例
            augment: 是否进行数据增强
            max_nodes: 最大节点数量（用于批处理）
        """
        self.data_dir = data_dir
        self.split = split
        self.augment = augment
        self.max_nodes = max_nodes
        
        # 加载所有数据文件
        self.data_files = [f for f in os.listdir(data_dir) if f.endswith('_processed.npz')]
        self.data_files.sort()
        
        # 划分训练/验证集
        n_train = int(len(self.data_files) * train_ratio)
        
        if split == 'train':
            self.data_files = self.data_files[:n_train]
        elif split == 'val':
            self.data_files = self.data_files[n_train:]
        elif split == 'test':
            # 如果有单独的测试集，可以在这里处理
            pass
        
        print(f"Loaded {len(self.data_files)} files for {split} split")
        
        # 预载所有数据到内存（如果内存足够）
        self.data_cache = {}
        self._preload_data()
    
    def _preload_data(self):
        """预载数据到内存"""
        print("Preloading data to memory...")
        
        for i, file_name in enumerate(self.data_files):
            try:
                file_path = os.path.join(self.data_dir, file_name)
                data = np.load(file_path, allow_pickle=True)
                
                # 检查数据完整性
                required_keys = ['node_features', 'node_positions', 'edge_index', 
                               'image_cubes', 'node_classes']
                
                if all(key in data for key in required_keys):
                    self.data_cache[i] = data
                else:
                    print(f"Warning: Missing keys in {file_name}")
                    
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
        
        print(f"Successfully preloaded {len(self.data_cache)} files")
    
    def __len__(self):
        return len(self.data_cache)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        if idx not in self.data_cache:
            return None
        
        data = self.data_cache[idx]
        
        # 提取数据
        node_features = torch.FloatTensor(data['node_features'])  # [N, 54]
        node_positions = torch.FloatTensor(data['node_positions'])  # [N, 3]
        edge_index = torch.LongTensor(data['edge_index'])  # [2, E]
        image_cubes = torch.FloatTensor(data['image_cubes'])  # [N, D, H, W]
        node_classes = torch.LongTensor(data['node_classes'])  # [N]
        
        # 数据增强
        if self.augment and self.split == 'train':
            node_features, node_positions, image_cubes = self._apply_augmentation(
                node_features, node_positions, image_cubes
            )
        
        # 限制节点数量（避免GPU内存不足）
        if len(node_features) > self.max_nodes:
            indices = self._sample_nodes(len(node_features), self.max_nodes)
            node_features = node_features[indices]
            node_positions = node_positions[indices]
            image_cubes = image_cubes[indices]
            node_classes = node_classes[indices]
            
            # 更新边索引
            edge_index = self._update_edge_index(edge_index, indices)
        
        # 添加批次维度到image_cubes
        image_cubes = image_cubes.unsqueeze(1)  # [N, 1, D, H, W]
        
        # 创建PyTorch Geometric数据对象
        graph_data = Data(
            x=node_features,
            pos=node_positions,
            edge_index=edge_index,
            y=node_classes,
            image_cubes=image_cubes,
            case_id=data.get('case_id', f'case_{idx}'),
            num_nodes=len(node_features)
        )
        
        return graph_data
    
    def _apply_augmentation(self, node_features, node_positions, image_cubes):
        """应用数据增强"""
        
        # 1. 随机旋转
        if random.random() < 0.5:
            node_features, node_positions, image_cubes = self._random_rotation(
                node_features, node_positions, image_cubes
            )
        
        # 2. 随机缩放
        if random.random() < 0.3:
            scale_factor = random.uniform(0.9, 1.1)
            node_positions = node_positions * scale_factor
            # 更新坐标相关的特征
            node_features[:, :3] = node_positions  # 假设前3维是坐标
        
        # 3. 随机噪声
        if random.random() < 0.3:
            noise_std = 0.01
            noise = torch.randn_like(image_cubes) * noise_std
            image_cubes = image_cubes + noise
        
        # 4. 随机强度变换
        if random.random() < 0.3:
            intensity_factor = random.uniform(0.8, 1.2)
            image_cubes = image_cubes * intensity_factor
        
        return node_features, node_positions, image_cubes
    
    def _random_rotation(self, node_features, node_positions, image_cubes):
        """随机旋转增强"""
        # 简化版本：只在xy平面旋转
        angle = random.uniform(-np.pi/6, np.pi/6)  # ±30度
        
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        # 旋转矩阵（只旋转xy平面）
        rotation_matrix = torch.FloatTensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        
        # 旋转位置
        node_positions = torch.matmul(node_positions, rotation_matrix.T)
        
        # 更新特征中的坐标和方向
        node_features[:, :3] = node_positions  # 坐标
        if node_features.shape[1] > 7:  # 如果有方向特征
            directions = node_features[:, 4:7]  # 假设4-6是方向向量
            rotated_directions = torch.matmul(directions, rotation_matrix.T)
            node_features[:, 4:7] = rotated_directions
        
        return node_features, node_positions, image_cubes
    
    def _sample_nodes(self, total_nodes: int, max_nodes: int) -> torch.LongTensor:
        """采样节点（保持连接性）"""
        if total_nodes <= max_nodes:
            return torch.arange(total_nodes)
        
        # 简单策略：等间隔采样
        step = total_nodes / max_nodes
        indices = [int(i * step) for i in range(max_nodes)]
        return torch.LongTensor(indices)
    
    def _update_edge_index(self, edge_index: torch.LongTensor, keep_indices: torch.LongTensor) -> torch.LongTensor:
        """更新边索引"""
        if edge_index.numel() == 0:
            return edge_index
        
        try:
            # 创建旧索引到新索引的映射
            old_to_new = {}
            for new_idx, old_idx in enumerate(keep_indices):
                old_to_new[old_idx.item()] = new_idx
            
            # 过滤有效的边
            valid_edges = []
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[0, i].item(), edge_index[1, i].item()
                if src in old_to_new and dst in old_to_new:
                    valid_edges.append([old_to_new[src], old_to_new[dst]])
            
            if len(valid_edges) == 0:
                # 如果没有有效边，返回空边索引
                return torch.empty((2, 0), dtype=torch.long)
            
            return torch.LongTensor(valid_edges).T
            
        except Exception as e:
            print(f"⚠️  Edge index update error: {e}")
            # 返回空边索引作为降级处理
            return torch.empty((2, 0), dtype=torch.long)
        
        # 过滤和更新边
        valid_edges = []
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src in old_to_new and dst in old_to_new:
                valid_edges.append([old_to_new[src], old_to_new[dst]])
        
        if valid_edges:
            return torch.LongTensor(valid_edges).T
        else:
            return torch.LongTensor([[], []])


def collate_vessel_graphs(batch):
    """自定义批处理函数"""
    # 过滤None值
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return None
    
    # 使用PyTorch Geometric的Batch
    return Batch.from_data_list(batch)


class VesselDataModule:
    """数据模块，管理数据加载"""
    
    def __init__(self, 
                 data_dir: str,
                 batch_size: int = 4,
                 num_workers: int = 4,
                 train_ratio: float = 0.8):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
    
    def setup(self):
        """设置数据集"""
        self.train_dataset = VesselGraphDataset(
            self.data_dir, 
            split='train', 
            train_ratio=self.train_ratio,
            augment=True
        )
        
        self.val_dataset = VesselGraphDataset(
            self.data_dir, 
            split='val', 
            train_ratio=self.train_ratio,
            augment=False
        )
    
    def train_dataloader(self):
        """训练数据加载器"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_vessel_graphs,
            pin_memory=True
        )
    
    def val_dataloader(self):
        """验证数据加载器"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_vessel_graphs,
            pin_memory=True
        )
    
    def get_class_weights(self) -> torch.FloatTensor:
        """计算类别权重（用于处理类别不平衡）"""
        if not hasattr(self, 'train_dataset'):
            self.setup()
        
        all_labels = []
        for i in range(len(self.train_dataset)):
            data = self.train_dataset[i]
            if data is not None:
                all_labels.extend(data.y.tolist())
        
        # 计算类别频率
        from collections import Counter
        label_counts = Counter(all_labels)
        
        # 计算权重（逆频率）
        total_samples = len(all_labels)
        num_classes = max(all_labels) + 1
        
        weights = []
        for i in range(num_classes):
            count = label_counts.get(i, 1)
            weight = total_samples / (num_classes * count)
            weights.append(weight)
        
        return torch.FloatTensor(weights)


def test_dataloader():
    """测试数据加载器"""
    data_module = VesselDataModule(
        data_dir="processed_data",
        batch_size=2,
        num_workers=0
    )
    
    data_module.setup()
    
    train_loader = data_module.train_dataloader()
    
    print("Testing dataloader...")
    for i, batch in enumerate(train_loader):
        if batch is None:
            continue
            
        print(f"Batch {i}:")
        print(f"  - Batch size: {batch.num_graphs}")
        print(f"  - Total nodes: {batch.num_nodes}")
        print(f"  - Node features shape: {batch.x.shape}")
        print(f"  - Image cubes shape: {batch.image_cubes.shape}")
        print(f"  - Edge index shape: {batch.edge_index.shape}")
        print(f"  - Labels shape: {batch.y.shape}")
        
        if i >= 2:  # 只测试前3个批次
            break
    
    # 计算类别权重
    class_weights = data_module.get_class_weights()
    print(f"\nClass weights: {class_weights}")


if __name__ == "__main__":
    test_dataloader()
