#!/usr/bin/env python3
"""
验证预处理后的数据格式
"""

import os
import numpy as np
import glob

def verify_processed_data(data_dir: str = "processed_data"):
    """验证预处理数据的格式和内容"""
    
    # 找到所有处理后的文件
    processed_files = glob.glob(os.path.join(data_dir, "*_processed.npz"))
    
    print(f"🔍 Found {len(processed_files)} processed files")
    print("=" * 50)
    
    total_nodes = 0
    total_edges = 0
    total_vessels = 0
    
    for i, file_path in enumerate(processed_files[:5]):  # 只检查前5个文件
        case_name = os.path.basename(file_path).replace("_processed.npz", "")
        print(f"\n📁 Case {i+1}: {case_name}")
        
        try:
            # 加载数据
            data = np.load(file_path, allow_pickle=True)
            
            # 显示数据结构
            print(f"  📊 Data keys: {list(data.keys())}")
            
            # 检查图数据
            if 'graph_data' in data:
                graph_data = data['graph_data'].item()
                
                # 节点特征
                if 'node_features' in graph_data:
                    node_features = graph_data['node_features']
                    print(f"  🔗 Node features shape: {node_features.shape}")
                    print(f"  📏 Feature dimensions: {node_features.shape[1] if len(node_features.shape) > 1 else 'N/A'}")
                
                # 边信息
                if 'edge_index' in graph_data:
                    edge_index = graph_data['edge_index']
                    print(f"  🌐 Edge index shape: {edge_index.shape}")
                    print(f"  🔗 Number of edges: {edge_index.shape[1] if len(edge_index.shape) > 1 else 'N/A'}")
                
                # 血管信息
                if 'vessel_info' in graph_data:
                    vessel_info = graph_data['vessel_info']
                    print(f"  🫀 Number of vessels: {len(vessel_info)}")
                    
                    # 显示血管类型
                    vessel_types = [info['class_id'] for info in vessel_info.values()]
                    unique_types = set(vessel_types)
                    print(f"  🏷️  Vessel types: {sorted(unique_types)}")
                
                # 累计统计
                if 'node_features' in graph_data:
                    total_nodes += len(graph_data['node_features'])
                if 'edge_index' in graph_data:
                    total_edges += graph_data['edge_index'].shape[1] if len(graph_data['edge_index'].shape) > 1 else 0
                if 'vessel_info' in graph_data:
                    total_vessels += len(graph_data['vessel_info'])
        
        except Exception as e:
            print(f"  ❌ Error loading {case_name}: {e}")
    
    print("\n" + "=" * 50)
    print(f"📈 Summary Statistics:")
    print(f"  Total processed cases: {len(processed_files)}")
    print(f"  Total nodes: {total_nodes}")
    print(f"  Total edges: {total_edges}")
    print(f"  Total vessels: {total_vessels}")
    print(f"  Average nodes per case: {total_nodes/min(len(processed_files), 5):.0f}")
    print(f"  Average vessels per case: {total_vessels/min(len(processed_files), 5):.1f}")
    
    return len(processed_files)

if __name__ == "__main__":
    verify_processed_data()
