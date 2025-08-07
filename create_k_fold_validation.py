#!/usr/bin/env python3
"""
K折交叉验证脚本 - 验证模型的真实性能
"""

import os
import json
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import glob
from collections import defaultdict
import argparse

def load_processed_data():
    """加载所有处理过的数据"""
    data_dir = "data/processed"
    all_data = []
    all_labels = []
    file_names = []
    
    for file_path in glob.glob(os.path.join(data_dir, "*_processed.npz")):
        file_name = os.path.basename(file_path).replace("_processed.npz", "")
        try:
            data = np.load(file_path, allow_pickle=True)
            
            # 获取数据
            node_features = data['node_features']
            edge_index = data['edge_index']
            node_labels = data['node_classes']
            
            # 转换为PyG数据格式
            graph_data = Data(
                x=torch.FloatTensor(node_features),
                edge_index=torch.LongTensor(edge_index),
                y=torch.LongTensor(node_labels)
            )
            
            all_data.append(graph_data)
            all_labels.extend(node_labels)
            file_names.append(file_name)
            
        except Exception as e:
            print(f"❌ 加载文件失败 {file_path}: {e}")
            continue
    
    print(f"✅ 成功加载 {len(all_data)} 个案例")
    return all_data, all_labels, file_names

def create_simple_model(input_dim, num_classes):
    """创建简单的基线模型"""
    from torch_geometric.nn import GCNConv, global_mean_pool
    import torch.nn as nn
    
    class SimpleGCN(torch.nn.Module):
        def __init__(self, input_dim, hidden_dim, num_classes):
            super(SimpleGCN, self).__init__()
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.classifier = nn.Linear(hidden_dim, num_classes)
            self.dropout = nn.Dropout(0.5)
            
        def forward(self, x, edge_index, batch=None):
            x = F.relu(self.conv1(x, edge_index))
            x = self.dropout(x)
            x = F.relu(self.conv2(x, edge_index))
            x = self.dropout(x)
            x = self.classifier(x)
            return x
    
    return SimpleGCN(input_dim, 64, num_classes)

def k_fold_validation(k=5):
    """执行K折交叉验证"""
    print(f"🔄 开始{k}折交叉验证...")
    
    # 加载数据
    all_data, all_labels, file_names = load_processed_data()
    
    if len(all_data) == 0:
        print("❌ 没有找到有效的数据文件")
        return
    
    # 创建K折分割
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    
    fold_results = []
    all_predictions = []
    all_true_labels = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(all_data)):
        print(f"\n📊 训练折 {fold+1}/{k}")
        print(f"   训练案例: {len(train_idx)}, 验证案例: {len(val_idx)}")
        
        # 分割数据
        train_data = [all_data[i] for i in train_idx]
        val_data = [all_data[i] for i in val_idx]
        
        # 创建数据加载器
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
        
        # 创建模型
        input_dim = all_data[0].x.shape[1]
        num_classes = len(set(all_labels))
        model = create_simple_model(input_dim, num_classes).to(device)
        
        # 训练配置
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        # 训练模型
        model.train()
        for epoch in range(50):  # 减少训练轮数
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")
        
        # 验证模型
        model.eval()
        fold_predictions = []
        fold_true_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                pred = out.argmax(dim=1)
                
                fold_predictions.extend(pred.cpu().numpy())
                fold_true_labels.extend(batch.y.cpu().numpy())
        
        # 计算折的性能
        fold_accuracy = accuracy_score(fold_true_labels, fold_predictions)
        print(f"   折 {fold+1} 准确率: {fold_accuracy:.4f}")
        
        fold_results.append({
            'fold': fold + 1,
            'accuracy': fold_accuracy,
            'train_cases': [file_names[i] for i in train_idx],
            'val_cases': [file_names[i] for i in val_idx],
            'predictions': fold_predictions,
            'true_labels': fold_true_labels
        })
        
        all_predictions.extend(fold_predictions)
        all_true_labels.extend(fold_true_labels)
    
    # 计算总体结果
    overall_accuracy = accuracy_score(all_true_labels, all_predictions)
    fold_accuracies = [r['accuracy'] for r in fold_results]
    
    print(f"\n📊 K折交叉验证结果:")
    print(f"   平均准确率: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
    print(f"   最高准确率: {np.max(fold_accuracies):.4f}")
    print(f"   最低准确率: {np.min(fold_accuracies):.4f}")
    print(f"   总体准确率: {overall_accuracy:.4f}")
    
    # 生成详细报告
    print(f"\n📋 分类报告:")
    print(classification_report(all_true_labels, all_predictions))
    
    # 保存结果
    results = {
        'k_folds': k,
        'fold_results': fold_results,
        'summary': {
            'mean_accuracy': np.mean(fold_accuracies),
            'std_accuracy': np.std(fold_accuracies),
            'max_accuracy': np.max(fold_accuracies),
            'min_accuracy': np.min(fold_accuracies),
            'overall_accuracy': overall_accuracy
        },
        'classification_report': classification_report(all_true_labels, all_predictions, output_dict=True),
        'confusion_matrix': confusion_matrix(all_true_labels, all_predictions).tolist()
    }
    
    with open(f'k_fold_validation_results_{k}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ 结果已保存到 k_fold_validation_results_{k}.json")
    
    return results

def leave_one_out_validation():
    """留一法交叉验证"""
    print("🔄 开始留一法交叉验证...")
    
    all_data, all_labels, file_names = load_processed_data()
    
    if len(all_data) == 0:
        print("❌ 没有找到有效的数据文件")
        return
    
    results = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for i in range(len(all_data)):
        print(f"📊 测试案例 {i+1}/{len(all_data)}: {file_names[i]}")
        
        # 分割数据
        test_data = [all_data[i]]
        train_data = [all_data[j] for j in range(len(all_data)) if j != i]
        
        # 创建数据加载器
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
        
        # 创建模型
        input_dim = all_data[0].x.shape[1]
        num_classes = len(set(all_labels))
        model = create_simple_model(input_dim, num_classes).to(device)
        
        # 训练配置
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        # 训练模型
        model.train()
        for epoch in range(30):  # 减少训练轮数
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        # 测试模型
        model.eval()
        test_predictions = []
        test_true_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                pred = out.argmax(dim=1)
                
                test_predictions.extend(pred.cpu().numpy())
                test_true_labels.extend(batch.y.cpu().numpy())
        
        # 计算准确率
        accuracy = accuracy_score(test_true_labels, test_predictions)
        print(f"   案例 {file_names[i]} 准确率: {accuracy:.4f}")
        
        results.append({
            'case': file_names[i],
            'accuracy': accuracy,
            'predictions': test_predictions,
            'true_labels': test_true_labels
        })
    
    # 计算总体结果
    accuracies = [r['accuracy'] for r in results]
    mean_accuracy = np.mean(accuracies)
    
    print(f"\n📊 留一法交叉验证结果:")
    print(f"   平均准确率: {mean_accuracy:.4f} ± {np.std(accuracies):.4f}")
    print(f"   最高准确率: {np.max(accuracies):.4f}")
    print(f"   最低准确率: {np.min(accuracies):.4f}")
    
    # 保存结果
    final_results = {
        'method': 'leave_one_out',
        'case_results': results,
        'summary': {
            'mean_accuracy': mean_accuracy,
            'std_accuracy': np.std(accuracies),
            'max_accuracy': np.max(accuracies),
            'min_accuracy': np.min(accuracies)
        }
    }
    
    with open('leave_one_out_validation_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"✅ 结果已保存到 leave_one_out_validation_results.json")
    
    return final_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='K折交叉验证')
    parser.add_argument('--k', type=int, default=5, help='K折数量')
    parser.add_argument('--method', choices=['kfold', 'loo', 'both'], default='both', help='验证方法')
    
    args = parser.parse_args()
    
    if args.method in ['kfold', 'both']:
        k_fold_validation(args.k)
    
    if args.method in ['loo', 'both']:
        leave_one_out_validation()
