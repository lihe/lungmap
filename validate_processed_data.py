#!/usr/bin/env python3
"""
验证处理后数据的质量
"""

import os
import numpy as np
import networkx as nx
from collections import Counter

def validate_processed_data(data_path: str):
    """验证处理后的数据质量"""
    print(f"🔍 验证数据: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"❌ 文件不存在: {data_path}")
        return False
    
    try:
        data = np.load(data_path, allow_pickle=True)
        
        # 1. 检查基础数据结构
        required_keys = ['node_features', 'node_positions', 'edge_index', 'image_cubes', 'node_classes']
        for key in required_keys:
            if key not in data:
                print(f"❌ 缺少关键字段: {key}")
                return False
        
        # 2. 检查数据维度一致性
        n_nodes = len(data['node_features'])
        print(f"📊 节点数量: {n_nodes}")
        
        if len(data['node_positions']) != n_nodes:
            print(f"❌ 节点位置数量不匹配: {len(data['node_positions'])} vs {n_nodes}")
            return False
        
        if len(data['node_classes']) != n_nodes:
            print(f"❌ 节点类别数量不匹配: {len(data['node_classes'])} vs {n_nodes}")
            return False
        
        if len(data['image_cubes']) != n_nodes:
            print(f"❌ 图像块数量不匹配: {len(data['image_cubes'])} vs {n_nodes}")
            return False
        
        # 3. 检查特征维度
        if data['node_features'].shape[1] != 54:
            print(f"❌ 节点特征维度错误: {data['node_features'].shape[1]} (期望: 54)")
            return False
        
        if data['node_positions'].shape[1] != 3:
            print(f"❌ 节点位置维度错误: {data['node_positions'].shape[1]} (期望: 3)")
            return False
        
        # 4. 检查图连通性
        edge_index = data['edge_index']
        n_edges = edge_index.shape[1] if edge_index.size > 0 else 0
        print(f"📊 边数量: {n_edges}")
        
        if n_edges > 0:
            # 构建NetworkX图进行连通性分析
            G = nx.Graph()
            G.add_nodes_from(range(n_nodes))
            
            for i in range(n_edges):
                src, tgt = edge_index[0, i], edge_index[1, i]
                if 0 <= src < n_nodes and 0 <= tgt < n_nodes:
                    G.add_edge(src, tgt)
            
            # 连通性分析
            components = list(nx.connected_components(G))
            print(f"📊 连通分量数: {len(components)}")
            print(f"📊 最大连通分量大小: {max(len(c) for c in components) if components else 0}")
            
            if len(components) == 1:
                print("✅ 图是连通的")
            else:
                print(f"⚠️ 图不连通，有 {len(components)} 个分量")
                for i, comp in enumerate(components):
                    print(f"   分量 {i+1}: {len(comp)} 个节点")
        else:
            print("⚠️ 没有边连接")
        
        # 5. 检查节点类别分布
        class_counts = Counter(data['node_classes'])
        print(f"📊 节点类别分布:")
        for cls, count in sorted(class_counts.items()):
            print(f"   类别 {cls}: {count} 个节点")
        
        # 6. 检查特征值分布
        features = data['node_features']
        print(f"📊 特征统计:")
        print(f"   值范围: [{features.min():.3f}, {features.max():.3f}]")
        print(f"   均值: {features.mean():.3f}")
        print(f"   标准差: {features.std():.3f}")
        print(f"   是否有NaN: {np.isnan(features).any()}")
        print(f"   是否有无穷大: {np.isinf(features).any()}")
        
        # 7. 检查图像块质量
        cubes = data['image_cubes']
        print(f"📊 图像块统计:")
        print(f"   形状: {cubes.shape}")
        print(f"   值范围: [{cubes.min():.3f}, {cubes.max():.3f}]")
        print(f"   均值: {cubes.mean():.3f}")
        print(f"   标准差: {cubes.std():.3f}")
        print(f"   是否有NaN: {np.isnan(cubes).any()}")
        
        # 8. 检查节点位置合理性
        positions = data['node_positions']
        print(f"📊 节点位置统计:")
        for dim, name in enumerate(['Z', 'Y', 'X']):
            pos_dim = positions[:, dim]
            print(f"   {name}轴范围: [{pos_dim.min():.1f}, {pos_dim.max():.1f}]")
        
        print("✅ 数据验证通过")
        return True
        
    except Exception as e:
        print(f"❌ 数据验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🧪 开始数据质量验证\n")
    
    # 验证测试数据
    test_data_path = "/home/lihe/classify/lungmap/data/processed_test/4000010002_processed.npz"
    success = validate_processed_data(test_data_path)
    
    if success:
        print("\n🎉 数据质量验证通过!")
    else:
        print("\n❌ 数据质量验证失败")

if __name__ == "__main__":
    main()
