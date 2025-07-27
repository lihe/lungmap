#!/usr/bin/env python3
"""
血管图数据加载器 - 适配VesselPreprocessor生成的npz格式数据
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import pickle
from sklearn.model_selection import train_test_split

class VesselGraphDataset(Dataset):
    """血管图数据集 - 适配预处理后的npz格式"""
    
    def __init__(self, data_list: List[Dict], transform=None):
        """
        Args:
            data_list: 包含预处理数据的字典列表
            transform: 数据变换（可选）
        """
        self.data_list = data_list
        self.transform = transform
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = self.data_list[idx]
        
        # 转换为PyTorch张量
        sample = {
            'case_id': data['case_id'],
            'node_features': torch.FloatTensor(data['node_features']),  # [N, 54]
            'node_positions': torch.FloatTensor(data['node_positions']),  # [N, 3]
            'edge_index': torch.LongTensor(data['edge_index']),  # [2, E]
            'image_cubes': torch.FloatTensor(data['image_cubes']).unsqueeze(1),  # [N, 1, 32, 32, 32]
            'node_classes': torch.LongTensor(data['node_classes']),  # [N]
            'vessel_node_ranges': data['vessel_node_ranges'],
            'node_to_vessel': data['node_to_vessel']
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample

def collate_fn(batch):
    """自定义批处理函数 - 处理不同大小的图"""
    # 因为每个图的节点数不同，需要特殊处理
    if len(batch) == 1:
        return batch[0]
    
    # 对于批处理，我们简单返回第一个样本
    # 在实际训练中，CPR-TaG-Net通常使用batch_size=1
    return batch[0]

def load_processed_data(processed_data_dir: str) -> List[Dict]:
    """加载所有预处理后的数据"""
    data_list = []
    
    if not os.path.exists(processed_data_dir):
        print(f"❌ Directory {processed_data_dir} does not exist!")
        return data_list
    
    for filename in os.listdir(processed_data_dir):
        if filename.endswith('_processed.npz'):
            filepath = os.path.join(processed_data_dir, filename)
            
            try:
                # 加载npz文件
                loaded_data = np.load(filepath, allow_pickle=True)
                
                # 转换为字典格式
                data_dict = {
                    'case_id': str(loaded_data['case_id']),
                    'node_features': loaded_data['node_features'],
                    'node_positions': loaded_data['node_positions'], 
                    'edge_index': loaded_data['edge_index'],
                    'image_cubes': loaded_data['image_cubes'],
                    'node_classes': loaded_data['node_classes'],
                    'vessel_node_ranges': loaded_data['vessel_node_ranges'].item(),
                    'node_to_vessel': loaded_data['node_to_vessel']
                }
                
                # 验证数据完整性
                if len(data_dict['node_features']) > 0:
                    data_list.append(data_dict)
                    print(f"✅ Loaded {filename}: {len(data_dict['node_features'])} nodes, {len(data_dict['vessel_node_ranges'])} vessels")
                    
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")
                continue
    
    print(f"\n📊 Successfully loaded {len(data_list)} cases")
    return data_list

def create_data_splits(data_list: List[Dict], 
                      train_ratio: float = 0.7,
                      val_ratio: float = 0.15,
                      test_ratio: float = 0.15,
                      random_state: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """创建训练/验证/测试数据分割"""
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"
    
    if len(data_list) < 3:
        print("⚠️  数据量太少，无法进行三分割，将使用简单分割")
        if len(data_list) == 1:
            return data_list, [], []
        elif len(data_list) == 2:
            return [data_list[0]], [data_list[1]], []
    
    # 首先分出训练集和临时集
    train_data, temp_data = train_test_split(
        data_list, 
        test_size=(1 - train_ratio),
        random_state=random_state,
        shuffle=True
    )
    
    # 从临时集中分出验证集和测试集
    if len(temp_data) >= 2:
        val_size = val_ratio / (val_ratio + test_ratio)
        val_data, test_data = train_test_split(
            temp_data,
            test_size=(1 - val_size),
            random_state=random_state,
            shuffle=True
        )
    else:
        val_data = temp_data
        test_data = []
    
    print(f"📊 Data splits:")
    print(f"  Train: {len(train_data)} cases")
    print(f"  Val: {len(val_data)} cases") 
    print(f"  Test: {len(test_data)} cases")
    
    return train_data, val_data, test_data

def create_dataloaders(train_data: List[Dict], 
                      val_data: List[Dict], 
                      test_data: List[Dict],
                      batch_size: int = 1,
                      num_workers: int = 2) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建数据加载器"""
    
    train_dataset = VesselGraphDataset(train_data)
    val_dataset = VesselGraphDataset(val_data) if val_data else None
    test_dataset = VesselGraphDataset(test_data) if test_data else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # 验证时使用batch_size=1
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    ) if val_dataset else None
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    ) if test_dataset else None
    
    return train_loader, val_loader, test_loader

def save_data_splits(train_data: List[Dict], val_data: List[Dict], test_data: List[Dict], save_dir: str):
    """保存数据分割结果"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存为pickle格式（与原代码兼容）
    with open(os.path.join(save_dir, 'train_data.pkl'), 'wb') as f:
        pickle.dump(train_data, f)
    
    with open(os.path.join(save_dir, 'val_data.pkl'), 'wb') as f:
        pickle.dump(val_data, f)
        
    with open(os.path.join(save_dir, 'test_data.pkl'), 'wb') as f:
        pickle.dump(test_data, f)
    
    print(f"💾 Data splits saved to {save_dir}")

def load_data_splits(save_dir: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """加载保存的数据分割"""
    with open(os.path.join(save_dir, 'train_data.pkl'), 'rb') as f:
        train_data = pickle.load(f)
    
    with open(os.path.join(save_dir, 'val_data.pkl'), 'rb') as f:
        val_data = pickle.load(f)
        
    with open(os.path.join(save_dir, 'test_data.pkl'), 'rb') as f:
        test_data = pickle.load(f)
        
    return train_data, val_data, test_data

def main():
    """测试数据加载器"""
    print("🔄 Loading processed data...")
    data_list = load_processed_data('/home/lihe/classify/lungmap/data/processed')
    
    if len(data_list) == 0:
        print("❌ No data found! Please run vessel_preprocessing.py first.")
        return
    
    print("\n📊 Creating data splits...")
    train_data, val_data, test_data = create_data_splits(data_list)
    
    print("\n💾 Saving data splits...")
    save_data_splits(train_data, val_data, test_data, '/home/lihe/classify/lungmap/data/splits')
    
    print("\n🔧 Creating data loaders...")
    train_loader, val_loader, test_loader = create_dataloaders(train_data, val_data, test_data)
    
    print("\n✅ Testing data loader...")
    for i, batch in enumerate(train_loader):
        print(f"Batch {i+1}:")
        print(f"  Case ID: {batch['case_id']}")
        print(f"  Node features: {batch['node_features'].shape}")
        print(f"  Node positions: {batch['node_positions'].shape}")
        print(f"  Edge index: {batch['edge_index'].shape}")
        print(f"  Image cubes: {batch['image_cubes'].shape}")
        print(f"  Node classes: {batch['node_classes'].shape}")
        print(f"  Unique classes: {torch.unique(batch['node_classes']).tolist()}")
        print()
        
        if i >= 2:  # 只测试前3个批次
            break
    
    print("🎉 Data loader setup completed!")

if __name__ == "__main__":
    main()
