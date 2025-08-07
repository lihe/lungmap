#!/usr/bin/env python3
"""
快速验证重新处理后的数据与修正后的血管层次结构一致性
"""

import numpy as np
import os
from glob import glob

# 正确的血管层次结构
VESSEL_HIERARCHY = {
    'MPA': {'level': 0, 'parent': None, 'class_id': 7},
    'LPA': {'level': 1, 'parent': 'MPA', 'class_id': 2},
    'RPA': {'level': 1, 'parent': 'MPA', 'class_id': 11},
    'Lupper': {'level': 2, 'parent': 'LPA', 'class_id': 6},
    'Rupper': {'level': 2, 'parent': 'RPA', 'class_id': 14},
    'L1+2': {'level': 2, 'parent': 'LPA', 'class_id': 0},
    'R1+2': {'level': 2, 'parent': 'RPA', 'class_id': 8},
    'L1+3': {'level': 2, 'parent': 'LPA', 'class_id': 1},
    'R1+3': {'level': 2, 'parent': 'RPA', 'class_id': 9},
    'Linternal': {'level': 2, 'parent': 'LPA', 'class_id': 4},
    'Rinternal': {'level': 2, 'parent': 'RPA', 'class_id': 12},
    'Lmedium': {'level': 2, 'parent': 'LPA', 'class_id': 5},
    'Rmedium': {'level': 2, 'parent': 'RPA', 'class_id': 13},
    'Ldown': {'level': 2, 'parent': 'LPA', 'class_id': 3},
    'RDown': {'level': 2, 'parent': 'RPA', 'class_id': 10}
}

# 解析血管名称到class_id的映射
vessel_to_class = {name: info['class_id'] for name, info in VESSEL_HIERARCHY.items()}
class_to_vessel = {info['class_id']: name for name, info in VESSEL_HIERARCHY.items()}

def validate_data_file(filepath):
    """验证单个数据文件"""
    try:
        data = np.load(filepath, allow_pickle=True)
        case_id = os.path.basename(filepath).replace('_processed.npz', '')
        
        # 基本信息
        n_nodes = data['node_features'].shape[0]
        n_edges = data['edge_index'].shape[1]
        vessel_ranges = data['vessel_node_ranges'].item()
        node_classes = data['node_classes']
        
        # 验证node_classes范围
        unique_classes = np.unique(node_classes)
        if np.max(unique_classes) >= 15:
            print(f"❌ {case_id}: 发现超出范围的类别 {unique_classes}")
            return False
            
        # 验证血管范围
        total_vessels = len(vessel_ranges)
        vessel_names = list(vessel_ranges.keys())
        
        # 检查血管名称是否都在预期的15类中
        invalid_vessels = [v for v in vessel_names if v not in VESSEL_HIERARCHY]
        if invalid_vessels:
            print(f"❌ {case_id}: 发现无效血管类型 {invalid_vessels}")
            return False
            
        # 检查血管层次结构
        levels = {}
        for vessel_name in vessel_names:
            levels[VESSEL_HIERARCHY[vessel_name]['level']] = levels.get(VESSEL_HIERARCHY[vessel_name]['level'], 0) + 1
        
        print(f"✅ {case_id}: {n_nodes}节点, {n_edges}边, {total_vessels}血管, 层次分布: {levels}")
        return True
        
    except Exception as e:
        print(f"❌ {case_id}: 处理失败 - {e}")
        return False

def main():
    print("🔍 验证重新处理后的数据...")
    print("="*60)
    
    # 找到所有处理后的文件
    processed_dir = "/home/lihe/classify/lungmap/data/processed"
    files = glob(os.path.join(processed_dir, "*_processed.npz"))
    
    if not files:
        print("❌ 未找到处理后的数据文件")
        return
    
    print(f"📁 找到 {len(files)} 个处理后的文件")
    print()
    
    valid_count = 0
    total_count = len(files)
    
    # 统计信息
    all_levels = {}
    all_vessels = set()
    total_nodes = 0
    total_edges = 0
    
    for filepath in sorted(files):
        if validate_data_file(filepath):
            valid_count += 1
            
            # 累计统计
            data = np.load(filepath, allow_pickle=True)
            vessel_ranges = data['vessel_node_ranges'].item()
            
            total_nodes += data['node_features'].shape[0]
            total_edges += data['edge_index'].shape[1]
            all_vessels.update(vessel_ranges.keys())
            
            # 层次统计
            for vessel_name in vessel_ranges.keys():
                level = VESSEL_HIERARCHY[vessel_name]['level']
                all_levels[level] = all_levels.get(level, 0) + 1
    
    print()
    print("="*60)
    print(f"📊 验证结果:")
    print(f"  有效文件: {valid_count}/{total_count}")
    print(f"  总节点数: {total_nodes:,}")
    print(f"  总边数: {total_edges:,}")
    print(f"  发现的血管类型: {len(all_vessels)}")
    print(f"  血管类型: {sorted(all_vessels)}")
    print(f"  层次分布: {all_levels}")
    
    if valid_count == total_count:
        print("✅ 所有文件验证通过！数据重新处理成功！")
        
        # 验证血管类别完整性
        expected_vessels = set(VESSEL_HIERARCHY.keys())
        found_vessels = all_vessels
        
        if found_vessels.issubset(expected_vessels):
            print("✅ 血管类型符合15类规范")
        else:
            unexpected = found_vessels - expected_vessels
            print(f"⚠️  发现意外的血管类型: {unexpected}")
            
        if len(found_vessels) == 15:
            print("✅ 血管类型数量正确")
        else:
            missing = expected_vessels - found_vessels
            print(f"⚠️  缺失的血管类型: {missing}")
            
    else:
        print(f"❌ {total_count - valid_count} 个文件验证失败")

if __name__ == "__main__":
    main()
