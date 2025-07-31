#!/usr/bin/env python3
"""
NPZ文件内容分析工具
"""
import numpy as np
import os

def analyze_npz_file(filepath):
    """分析NPZ文件内容"""
    if not os.path.exists(filepath):
        print(f"文件不存在: {filepath}")
        return
    
    print(f"=== NPZ文件分析: {os.path.basename(filepath)} ===")
    print(f"文件大小: {os.path.getsize(filepath)/1024/1024:.2f} MB")
    
    try:
        data = np.load(filepath, allow_pickle=True)
        print(f"包含的数组: {list(data.keys())}")
        print()
        
        for key in data.keys():
            arr = data[key]
            print(f"📊 {key}:")
            
            if hasattr(arr, 'shape'):
                print(f"   形状: {arr.shape}")
                print(f"   数据类型: {arr.dtype}")
                print(f"   内存大小: {arr.nbytes/1024:.1f} KB")
                
                if key == 'node_features' and len(arr.shape) == 2:
                    print(f"   特征维度: {arr.shape[0]} 个节点 × {arr.shape[1]} 个特征")
                elif key == 'node_positions' and len(arr.shape) == 2:
                    print(f"   位置维度: {arr.shape[0]} 个节点 × {arr.shape[1]}D 坐标")
                elif key == 'edge_index' and len(arr.shape) == 2:
                    print(f"   边连接: {arr.shape[1]} 条边")
                elif key == 'image_cubes' and len(arr.shape) == 4:
                    print(f"   图像块: {arr.shape[0]} 个节点 × {arr.shape[1]}×{arr.shape[2]}×{arr.shape[3]} 体素")
                elif key == 'node_classes':
                    unique_classes = np.unique(arr)
                    print(f"   类别分布: {len(unique_classes)} 个不同类别")
                    print(f"   类别范围: {unique_classes.min()} - {unique_classes.max()}")
                    
                if arr.size < 20 and key not in ['image_cubes']:
                    print(f"   内容预览: {arr}")
                    
            else:
                print(f"   类型: {type(arr)}")
                print(f"   内容: {arr}")
            print()
            
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")

if __name__ == "__main__":
    # 分析第一个NPZ文件
    npz_dir = "/home/lihe/classify/lungmap/data/processed"
    if os.path.exists(npz_dir):
        npz_files = [f for f in os.listdir(npz_dir) if f.endswith('.npz')]
        if npz_files:
            analyze_npz_file(os.path.join(npz_dir, npz_files[0]))
        else:
            print("未找到NPZ文件")
    else:
        print(f"目录不存在: {npz_dir}")
