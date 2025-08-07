#!/usr/bin/env python3
"""
详细的NPZ文件内容分析工具
"""
import numpy as np
import os
from collections import Counter

def analyze_vessel_hierarchy():
    """分析血管层次结构"""
    print("🧬 血管层次结构分析:")
    vessel_hierarchy = {
        0: 'Background',
        1: 'MPA (主肺动脉)',
        2: 'LPA (左肺动脉)', 
        3: 'RPA (右肺动脉)',
        4: 'Lupper (左上叶动脉)',
        5: 'Rupper (右上叶动脉)',
        6: 'L1+2 (左上下叶间动脉)',
        7: 'R1+2 (右上下叶间动脉)',
        8: 'L1+3 (左1+3段动脉)',
        9: 'R1+3 (右1+3段动脉)',
        10: 'Lmedium (左中叶动脉)',
        11: 'Rmedium (右中叶动脉)',
        12: 'Linternal (左内侧动脉)',
        13: 'Rinternal (右内侧动脉)',
        14: 'Ldown (左下叶动脉)',
        15: 'RDown (右下叶动脉)'
    }
    
    for class_id, vessel_name in vessel_hierarchy.items():
        print(f"   类别 {class_id:2d}: {vessel_name}")
    print()

def analyze_npz_detailed(filepath):
    """详细分析NPZ文件内容"""
    if not os.path.exists(filepath):
        print(f"文件不存在: {filepath}")
        return
    
    print(f"=== 详细NPZ分析: {os.path.basename(filepath)} ===")
    print(f"文件大小: {os.path.getsize(filepath)/1024/1024:.2f} MB")
    
    try:
        data = np.load(filepath, allow_pickle=True)
        
        # 基本信息
        case_id = data['case_id'].item()
        node_features = data['node_features']
        node_positions = data['node_positions']
        edge_index = data['edge_index']
        image_cubes = data['image_cubes']
        node_classes = data['node_classes']
        vessel_ranges = data['vessel_node_ranges'].item()
        node_to_vessel = data['node_to_vessel']
        
        print(f"📋 病例ID: {case_id}")
        print(f"🔗 图结构: {node_features.shape[0]} 个节点, {edge_index.shape[1]} 条边")
        print(f"📊 特征维度: {node_features.shape[1]} 维特征向量")
        print(f"🎯 节点类别分布:")
        
        # 类别分布分析
        class_counts = Counter(node_classes)
        for class_id in sorted(class_counts.keys()):
            count = class_counts[class_id]
            percentage = count / len(node_classes) * 100
            print(f"   类别 {class_id:2d}: {count:2d} 个节点 ({percentage:5.1f}%)")
        
        print(f"\n🩸 血管分布:")
        for vessel_name, (start, end) in vessel_ranges.items():
            node_count = end - start + 1
            vessel_classes = node_classes[start:end+1]
            unique_classes = len(set(vessel_classes))
            print(f"   {vessel_name:12s}: 节点 {start:2d}-{end:2d} ({node_count:2d}个) - {unique_classes}种类别")
        
        # 特征统计
        print(f"\n📈 特征统计:")
        print(f"   特征均值范围: {node_features.mean(axis=0).min():.3f} ~ {node_features.mean(axis=0).max():.3f}")
        print(f"   特征标准差范围: {node_features.std(axis=0).min():.3f} ~ {node_features.std(axis=0).max():.3f}")
        
        # 图像块分析
        print(f"\n🖼️  图像块分析:")
        print(f"   总体积: {np.prod(image_cubes.shape)} 个体素")
        print(f"   强度范围: {image_cubes.min():.3f} ~ {image_cubes.max():.3f}")
        print(f"   平均强度: {image_cubes.mean():.3f} ± {image_cubes.std():.3f}")
        
        # 空间分布
        print(f"\n📍 空间分布:")
        print(f"   X 范围: {node_positions[:, 0].min():.1f} ~ {node_positions[:, 0].max():.1f}")
        print(f"   Y 范围: {node_positions[:, 1].min():.1f} ~ {node_positions[:, 1].max():.1f}")
        print(f"   Z 范围: {node_positions[:, 2].min():.1f} ~ {node_positions[:, 2].max():.1f}")
        
        # 连通性分析
        print(f"\n🔗 连通性分析:")
        degree_counts = np.bincount(np.concatenate([edge_index[0], edge_index[1]]))
        print(f"   度数范围: {degree_counts.min()} ~ {degree_counts.max()}")
        print(f"   平均度数: {degree_counts.mean():.2f}")
        
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
    
    print("=" * 60)

def analyze_multiple_files():
    """分析多个文件的统计信息"""
    npz_dir = "/home/lihe/classify/lungmap/data/processed"
    if not os.path.exists(npz_dir):
        print(f"目录不存在: {npz_dir}")
        return
    
    npz_files = sorted([f for f in os.listdir(npz_dir) if f.endswith('.npz')])
    if not npz_files:
        print("未找到NPZ文件")
        return
    
    print(f"📁 找到 {len(npz_files)} 个NPZ文件")
    print()
    
    # 分析血管层次结构
    analyze_vessel_hierarchy()
    
    # 详细分析前3个文件
    for i, filename in enumerate(npz_files[:3]):
        filepath = os.path.join(npz_dir, filename)
        analyze_npz_detailed(filepath)
        if i < 2:
            print()
    
    # 统计所有文件的概览
    print("\n📊 整体数据集统计:")
    total_nodes = 0
    total_edges = 0
    total_vessels = 0
    file_sizes = []
    
    for filename in npz_files:
        filepath = os.path.join(npz_dir, filename)
        try:
            data = np.load(filepath, allow_pickle=True)
            total_nodes += data['node_features'].shape[0]
            total_edges += data['edge_index'].shape[1]
            total_vessels += len(data['vessel_node_ranges'].item())
            file_sizes.append(os.path.getsize(filepath) / 1024 / 1024)
        except Exception as e:
            print(f"读取 {filename} 失败: {e}")
    
    print(f"   总文件数: {len(npz_files)}")
    print(f"   总节点数: {total_nodes}")
    print(f"   总边数: {total_edges}")
    print(f"   总血管数: {total_vessels}")
    print(f"   平均节点/病例: {total_nodes/len(npz_files):.1f}")
    print(f"   平均血管/病例: {total_vessels/len(npz_files):.1f}")
    print(f"   文件大小范围: {min(file_sizes):.2f} - {max(file_sizes):.2f} MB")
    print(f"   总数据集大小: {sum(file_sizes):.2f} MB")

if __name__ == "__main__":
    analyze_multiple_files()
