#!/usr/bin/env python3
"""
分析大节点案例的详细信息
"""
import numpy as np
import os
from collections import Counter

def analyze_large_cases():
    """分析大节点案例"""
    npz_dir = "/home/lihe/classify/lungmap/data/processed"
    large_cases = ["4000010084_processed.npz", "4000010082_processed.npz", "3000020023_processed.npz"]
    
    print("🔍 分析大节点案例")
    print("=" * 60)
    
    for filename in large_cases:
        filepath = os.path.join(npz_dir, filename)
        if not os.path.exists(filepath):
            print(f"❌ 文件不存在: {filename}")
            continue
            
        print(f"\n📋 案例: {filename}")
        print("-" * 40)
        
        try:
            data = np.load(filepath, allow_pickle=True)
            
            case_id = data['case_id'].item()
            node_features = data['node_features']
            node_positions = data['node_positions']
            edge_index = data['edge_index']
            image_cubes = data['image_cubes']
            node_classes = data['node_classes']
            vessel_ranges = data['vessel_node_ranges'].item()
            node_to_vessel = data['node_to_vessel']
            
            print(f"🆔 病例ID: {case_id}")
            print(f"📊 节点数量: {len(node_features)}")
            print(f"🔗 边数量: {edge_index.shape[1]}")
            print(f"🩸 血管数量: {len(vessel_ranges)}")
            
            # 分析血管分布
            print(f"\n🩸 血管详细分布:")
            total_vessel_nodes = 0
            for vessel_name, (start, end) in vessel_ranges.items():
                node_count = end - start + 1
                total_vessel_nodes += node_count
                vessel_classes = node_classes[start:end+1]
                unique_classes = len(set(vessel_classes))
                avg_class = np.mean(vessel_classes)
                print(f"  {vessel_name:12s}: 节点 {start:3d}-{end:3d} ({node_count:3d}个) - 类别: {unique_classes} (平均: {avg_class:.1f})")
            
            print(f"  总血管节点数: {total_vessel_nodes}")
            
            # 分析节点类别分布
            print(f"\n🎯 节点类别分布:")
            class_counts = Counter(node_classes)
            for class_id in sorted(class_counts.keys()):
                count = class_counts[class_id]
                percentage = count / len(node_classes) * 100
                print(f"  类别 {class_id:2d}: {count:3d} 个节点 ({percentage:5.1f}%)")
            
            # 分析空间分布
            print(f"\n📍 空间分布:")
            print(f"  X 范围: {node_positions[:, 0].min():.1f} ~ {node_positions[:, 0].max():.1f} (跨度: {node_positions[:, 0].max() - node_positions[:, 0].min():.1f})")
            print(f"  Y 范围: {node_positions[:, 1].min():.1f} ~ {node_positions[:, 1].max():.1f} (跨度: {node_positions[:, 1].max() - node_positions[:, 1].min():.1f})")
            print(f"  Z 范围: {node_positions[:, 2].min():.1f} ~ {node_positions[:, 2].max():.1f} (跨度: {node_positions[:, 2].max() - node_positions[:, 2].min():.1f})")
            
            # 分析特征统计
            print(f"\n📈 特征统计:")
            print(f"  特征均值范围: {node_features.mean(axis=0).min():.3f} ~ {node_features.mean(axis=0).max():.3f}")
            print(f"  特征标准差范围: {node_features.std(axis=0).min():.3f} ~ {node_features.std(axis=0).max():.3f}")
            
            # 分析图像数据
            print(f"\n🖼️  图像块分析:")
            print(f"  形状: {image_cubes.shape}")
            print(f"  强度范围: {image_cubes.min():.3f} ~ {image_cubes.max():.3f}")
            print(f"  数据大小: {image_cubes.nbytes / 1024 / 1024:.1f} MB")
            
            # 文件大小
            file_size = os.path.getsize(filepath) / 1024 / 1024
            print(f"\n💾 文件大小: {file_size:.2f} MB")
            
            # 检查是否有重复的血管类型
            vessel_types = list(vessel_ranges.keys())
            print(f"\n🔍 血管类型: {vessel_types}")
            
            # 检查节点密度
            if len(vessel_ranges) > 0:
                avg_nodes_per_vessel = len(node_features) / len(vessel_ranges)
                print(f"📊 平均每个血管节点数: {avg_nodes_per_vessel:.1f}")
            
        except Exception as e:
            print(f"❌ 读取文件失败: {e}")
            continue
        
        print("=" * 60)

def compare_with_normal_cases():
    """与正常案例对比"""
    print(f"\n📊 与正常案例对比:")
    print("=" * 60)
    
    npz_dir = "/home/lihe/classify/lungmap/data/processed"
    all_files = [f for f in os.listdir(npz_dir) if f.endswith('.npz')]
    
    node_counts = []
    vessel_counts = []
    
    for filename in all_files:
        filepath = os.path.join(npz_dir, filename)
        try:
            data = np.load(filepath, allow_pickle=True)
            node_count = len(data['node_features'])
            vessel_count = len(data['vessel_node_ranges'].item())
            
            node_counts.append((filename, node_count))
            vessel_counts.append((filename, vessel_count))
            
        except Exception as e:
            continue
    
    # 按节点数排序
    node_counts.sort(key=lambda x: x[1], reverse=True)
    
    print("🔝 节点数最多的10个案例:")
    for i, (filename, count) in enumerate(node_counts[:10]):
        marker = "🔥" if count > 100 else "📊"
        print(f"  {i+1:2d}. {marker} {filename:25s}: {count:3d} 节点")
    
    print(f"\n📈 统计信息:")
    all_node_counts = [count for _, count in node_counts]
    print(f"  最小节点数: {min(all_node_counts)}")
    print(f"  最大节点数: {max(all_node_counts)}")
    print(f"  平均节点数: {np.mean(all_node_counts):.1f}")
    print(f"  中位数节点数: {np.median(all_node_counts):.1f}")
    print(f"  超过100节点的案例: {sum(1 for count in all_node_counts if count > 100)} / {len(all_node_counts)}")

if __name__ == "__main__":
    analyze_large_cases()
    compare_with_normal_cases()
