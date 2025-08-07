#!/usr/bin/env python3
"""
检查处理后数据质量的脚本
"""

import numpy as np
import sys
import os

def analyze_processed_data(file_path):
    """分析处理后的数据"""
    print(f"分析文件: {file_path}")
    
    try:
        data = np.load(file_path, allow_pickle=True)
        print(f"文件大小: {os.path.getsize(file_path) / 1024:.1f} KB")
        print(f"数据键: {list(data.keys())}")
        
        if 'nodes' in data:
            nodes = data['nodes']
            print(f"节点数: {len(nodes)}")
            print(f"节点特征维度: {nodes[0].shape if len(nodes) > 0 else 'N/A'}")
        
        if 'edges' in data:
            edges = data['edges']
            print(f"边数: {len(edges)}")
        
        if 'vessel_info' in data:
            vessel_info = data['vessel_info'].item()  # 转换为字典
            print(f"血管信息:")
            for vessel_name, info in vessel_info.items():
                coords = info['coords']
                radii = info['radii']
                additional_features = info.get('additional_features', {})
                
                print(f"  {vessel_name}:")
                print(f"    点数: {len(coords)}")
                print(f"    平均半径: {np.mean(radii):.2f}")
                print(f"    半径范围: {np.min(radii):.2f} - {np.max(radii):.2f}")
                
                if 'quality_metrics' in additional_features:
                    quality = additional_features['quality_metrics']
                    print(f"    质量分数: {quality['overall_score']:.3f}")
                    print(f"    长度保持: {quality['length_preservation']:.3f}")
                    print(f"    形状保持: {quality['shape_preservation']:.3f}")
                    print(f"    覆盖率: {quality['coverage']:.3f}")
                
                if 'processing_time' in additional_features:
                    print(f"    处理时间: {additional_features['processing_time']:.2f}s")
        
        if 'processing_stats' in data:
            stats = data['processing_stats'].item()
            print(f"处理统计:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
                
        print("=" * 50)
        
    except Exception as e:
        print(f"读取文件出错: {e}")

def main():
    """分析所有处理后的数据"""
    processed_dir = "/home/lihe/classify/lungmap/data/processed"
    
    files = [f for f in os.listdir(processed_dir) if f.endswith('.npz')]
    print(f"找到 {len(files)} 个处理后的文件\n")
    
    # 分析前3个文件作为示例
    for i, filename in enumerate(files[:3]):
        file_path = os.path.join(processed_dir, filename)
        analyze_processed_data(file_path)
        if i < 2:  # 不是最后一个
            print()

if __name__ == "__main__":
    main()
