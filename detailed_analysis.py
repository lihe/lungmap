#!/usr/bin/env python3
"""
详细分析处理后数据的脚本
"""

import numpy as np
import sys
import os

def detailed_analysis(file_path):
    """详细分析处理后的数据"""
    print(f"详细分析: {os.path.basename(file_path)}")
    
    try:
        data = np.load(file_path, allow_pickle=True)
        print(f"文件大小: {os.path.getsize(file_path) / 1024:.1f} KB")
        
        # 基本信息
        case_id = data.get('case_id', 'Unknown')
        print(f"病例ID: {case_id}")
        
        # 节点信息
        if 'node_features' in data:
            node_features = data['node_features']
            print(f"节点数量: {len(node_features)}")
            print(f"节点特征维度: {node_features.shape[1] if len(node_features.shape) > 1 else 'N/A'}")
            
        if 'node_positions' in data:
            node_positions = data['node_positions']
            print(f"节点位置维度: {node_positions.shape}")
            
        if 'node_classes' in data:
            node_classes = data['node_classes']
            unique_classes = np.unique(node_classes)
            print(f"节点类别: {unique_classes}")
            
        # 边信息
        if 'edge_index' in data:
            edge_index = data['edge_index']
            print(f"边数量: {edge_index.shape[1] if len(edge_index.shape) > 1 else len(edge_index)}")
            
        # 血管范围信息
        if 'vessel_node_ranges' in data:
            vessel_ranges = data['vessel_node_ranges'].item()
            print(f"血管数量: {len(vessel_ranges)}")
            for vessel_name, (start, end) in vessel_ranges.items():
                print(f"  {vessel_name}: 节点 {start}-{end} ({end-start+1}个点)")
                
        # 图像立方体
        if 'image_cubes' in data:
            image_cubes = data['image_cubes']
            print(f"图像立方体数量: {len(image_cubes)}")
            if len(image_cubes) > 0:
                print(f"立方体尺寸: {image_cubes[0].shape}")
                
        print("=" * 60)
        
    except Exception as e:
        print(f"分析文件出错: {e}")
        import traceback
        traceback.print_exc()

def summary_statistics():
    """汇总统计信息"""
    processed_dir = "/home/lihe/classify/lungmap/data/processed"
    files = [f for f in os.listdir(processed_dir) if f.endswith('.npz')]
    
    total_files = len(files)
    total_nodes = 0
    total_edges = 0
    total_vessels = 0
    file_sizes = []
    
    print(f"\n{'='*20} 汇总统计 {'='*20}")
    
    for filename in files:
        file_path = os.path.join(processed_dir, filename)
        file_sizes.append(os.path.getsize(file_path) / 1024)  # KB
        
        try:
            data = np.load(file_path, allow_pickle=True)
            
            if 'node_features' in data:
                total_nodes += len(data['node_features'])
                
            if 'edge_index' in data:
                edge_index = data['edge_index']
                total_edges += edge_index.shape[1] if len(edge_index.shape) > 1 else len(edge_index)
                
            if 'vessel_node_ranges' in data:
                vessel_ranges = data['vessel_node_ranges'].item()
                total_vessels += len(vessel_ranges)
                
        except Exception as e:
            print(f"处理 {filename} 时出错: {e}")
            
    print(f"总文件数: {total_files}")
    print(f"总节点数: {total_nodes}")
    print(f"总边数: {total_edges}")
    print(f"总血管数: {total_vessels}")
    print(f"平均每个文件:")
    print(f"  节点数: {total_nodes/total_files:.1f}")
    print(f"  边数: {total_edges/total_files:.1f}")
    print(f"  血管数: {total_vessels/total_files:.1f}")
    print(f"  文件大小: {np.mean(file_sizes):.1f} KB")
    print(f"总数据大小: {sum(file_sizes)/1024:.1f} MB")

def main():
    """主函数"""
    processed_dir = "/home/lihe/classify/lungmap/data/processed"
    files = [f for f in os.listdir(processed_dir) if f.endswith('.npz')]
    
    print(f"发现 {len(files)} 个处理后的文件")
    
    # 详细分析前3个文件
    print(f"\n{'='*20} 详细分析示例 {'='*20}")
    for filename in files[:3]:
        file_path = os.path.join(processed_dir, filename)
        detailed_analysis(file_path)
        
    # 汇总统计
    summary_statistics()

if __name__ == "__main__":
    main()
