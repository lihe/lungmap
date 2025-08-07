#!/usr/bin/env python3
"""
调试血管类别映射
"""

from src.preprocessing.vessel_preprocessing import VesselPreprocessor

def debug_vessel_mapping():
    """调试血管类别映射"""
    preprocessor = VesselPreprocessor('.', '.', '/tmp/test', cube_size=32)
    
    print("🔍 血管层次结构:")
    for level, vessels in preprocessor.vessel_hierarchy.items():
        print(f"  {level}: {vessels}")
    
    print("\n🏷️ 标签到类别映射:")
    for vessel, class_id in preprocessor.label_to_class.items():
        print(f"  {vessel} -> {class_id}")
    
    print("\n🔗 解剖学邻接关系 (当前代码中):")
    anatomical_adjacency = {
        0: [1, 2],        # MPA -> LPA, RPA
        1: [3, 4],        # LPA -> Linternal, Lupper  
        2: [5, 6],        # RPA -> Rinternal, Rupper
        3: [7, 8],        # Linternal -> Lmedium, Ldown
        4: [9, 10],       # Lupper -> L1+2, L1+3
        5: [11, 12],      # Rinternal -> Rmedium, RDown
        6: [13, 14],      # Rupper -> R1+2, R1+3
    }
    
    for class_id, connected_classes in anatomical_adjacency.items():
        print(f"  类别 {class_id} -> {connected_classes}")
    
    print("\n❌ 问题分析:")
    print("   实际映射与解剖学邻接关系不一致!")
    
    # 构建正确的映射
    print("\n✅ 正确的解剖学邻接关系应该是:")
    reverse_mapping = {v: k for k, v in preprocessor.label_to_class.items()}
    
    correct_adjacency = {}
    
    # MPA (0) -> LPA (1), RPA (2)
    mpa_id = preprocessor.label_to_class.get('MPA')
    lpa_id = preprocessor.label_to_class.get('LPA')
    rpa_id = preprocessor.label_to_class.get('RPA')
    
    if mpa_id is not None:
        correct_adjacency[mpa_id] = [lpa_id, rpa_id]
        print(f"  MPA ({mpa_id}) -> LPA ({lpa_id}), RPA ({rpa_id})")
    
    # LPA连接
    linternal_id = preprocessor.label_to_class.get('Linternal')
    lupper_id = preprocessor.label_to_class.get('Lupper')
    lmedium_id = preprocessor.label_to_class.get('Lmedium')
    ldown_id = preprocessor.label_to_class.get('Ldown')
    
    if lpa_id is not None:
        connections = [linternal_id, lupper_id, lmedium_id, ldown_id]
        connections = [c for c in connections if c is not None]
        correct_adjacency[lpa_id] = connections
        print(f"  LPA ({lpa_id}) -> {connections}")
    
    # RPA连接
    rinternal_id = preprocessor.label_to_class.get('Rinternal')
    rupper_id = preprocessor.label_to_class.get('Rupper')
    rmedium_id = preprocessor.label_to_class.get('Rmedium')
    rdown_id = preprocessor.label_to_class.get('RDown')
    
    if rpa_id is not None:
        connections = [rinternal_id, rupper_id, rmedium_id, rdown_id]
        connections = [c for c in connections if c is not None]
        correct_adjacency[rpa_id] = connections
        print(f"  RPA ({rpa_id}) -> {connections}")
    
    return correct_adjacency

if __name__ == "__main__":
    debug_vessel_mapping()
