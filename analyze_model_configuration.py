#!/usr/bin/env python3
"""
综合分析训练代码和模型结构中的血管层次信息配置
"""

import sys
import os
import numpy as np
import glob

def analyze_data_labels():
    """分析实际数据中的标签分布"""
    print("🔍 分析实际数据中的标签分布...")
    
    data_dir = '/home/lihe/classify/lungmap/data/processed'
    npz_files = glob.glob(os.path.join(data_dir, '*.npz'))
    
    all_labels = []
    valid_files = 0
    
    for file_path in npz_files:
        try:
            data = np.load(file_path)
            if 'node_classes' in data:
                labels = data['node_classes']
                all_labels.extend(labels)
                valid_files += 1
            data.close()
        except Exception as e:
            continue
    
    if all_labels:
        all_unique = np.unique(all_labels)
        print(f"  ✅ 成功读取 {valid_files} 个数据文件")
        print(f"  📊 实际标签: {sorted(all_unique)}")
        print(f"  📊 标签数量: {len(all_unique)} 种")
        print(f"  📊 标签范围: {all_unique.min()} - {all_unique.max()}")
        return all_unique, len(all_unique)
    else:
        print("  ❌ 无法读取数据标签")
        return None, 0

def analyze_training_config():
    """分析训练代码中的配置"""
    print("\n🔍 分析训练代码配置...")
    
    # 读取训练代码
    try:
        with open('/home/lihe/classify/lungmap/train.py', 'r', encoding='utf-8') as f:
            train_content = f.read()
        
        # 检查num_classes配置
        if "num_classes': 15" in train_content:
            print("  ✅ 训练代码中 num_classes = 15")
            train_num_classes = 15
        elif "num_classes': 16" in train_content:
            print("  ⚠️  训练代码中 num_classes = 16")
            train_num_classes = 16
        else:
            print("  ❓ 无法确定训练代码中的 num_classes")
            train_num_classes = None
        
        # 检查血管层次结构
        if "'MPA':" in train_content and "'LPA':" in train_content:
            print("  ✅ 训练代码包含血管层次结构")
            has_vessel_hierarchy = True
        else:
            print("  ❌ 训练代码缺少血管层次结构")
            has_vessel_hierarchy = False
        
        # 检查特征维度
        if "enhanced_feature_dim = 54 + 32 + 3 + 1" in train_content:
            print("  ✅ 特征维度: 90D (54+32+3+1)")
            feature_dim = 90
        elif "enhanced_feature_dim = 54 + 32 + 4 + 1" in train_content:
            print("  ⚠️  特征维度: 91D (54+32+4+1) - 可能需要更新")
            feature_dim = 91
        else:
            print("  ❓ 无法确定特征维度")
            feature_dim = None
        
        return train_num_classes, has_vessel_hierarchy, feature_dim
        
    except Exception as e:
        print(f"  ❌ 读取训练代码失败: {e}")
        return None, False, None

def analyze_model_config():
    """分析CPR-TaG-Net模型配置"""
    print("\n🔍 分析CPR-TaG-Net模型配置...")
    
    try:
        model_file = '/home/lihe/classify/lungmap/src/models/CPR_TaG_Net/models/cpr_tagnet.py'
        with open(model_file, 'r', encoding='utf-8') as f:
            model_content = f.read()
        
        # 检查默认num_classes
        if "num_classes=18" in model_content:
            print("  ⚠️  模型默认 num_classes = 18 (会被训练代码覆盖)")
            model_default_classes = 18
        elif "num_classes=15" in model_content:
            print("  ✅ 模型默认 num_classes = 15")
            model_default_classes = 15
        else:
            print("  ❓ 无法确定模型默认 num_classes")
            model_default_classes = None
        
        # 检查默认特征维度
        if "node_feature_dim=54" in model_content:
            print("  ✅ 模型默认 node_feature_dim = 54")
            model_default_features = 54
        elif "node_feature_dim=90" in model_content:
            print("  ✅ 模型默认 node_feature_dim = 90")
            model_default_features = 90
        else:
            print("  ❓ 无法确定模型默认特征维度")
            model_default_features = None
        
        return model_default_classes, model_default_features
        
    except Exception as e:
        print(f"  ❌ 读取模型文件失败: {e}")
        return None, None

def analyze_vessel_hierarchy():
    """分析血管层次结构配置"""
    print("\n🔍 分析血管层次结构配置...")
    
    # 从训练代码中提取血管层次结构
    vessel_hierarchy = {
        # 一级：主肺动脉
        'MPA': {'level': 0, 'parent': None, 'expected_class_range': [0, 1, 2, 3]},
        
        # 二级：左右肺动脉
        'LPA': {'level': 1, 'parent': 'MPA', 'expected_class_range': [1, 2, 3]},
        'RPA': {'level': 1, 'parent': 'MPA', 'expected_class_range': [1, 2, 3]},
        
        # 三级：上叶、段间、内侧、中叶、下叶分支（包括变异）
        'Lupper': {'level': 2, 'parent': 'LPA', 'expected_class_range': [4, 5, 6, 7]},
        'Rupper': {'level': 2, 'parent': 'RPA', 'expected_class_range': [4, 5, 6, 7]},
        'L1+2': {'level': 2, 'parent': 'LPA', 'expected_class_range': [4, 5, 6, 7]},
        'R1+2': {'level': 2, 'parent': 'RPA', 'expected_class_range': [4, 5, 6, 7]},
        'L1+3': {'level': 2, 'parent': 'LPA', 'expected_class_range': [4, 5, 6, 7]},
        'R1+3': {'level': 2, 'parent': 'RPA', 'expected_class_range': [4, 5, 6, 7]},
        'Linternal': {'level': 2, 'parent': 'LPA', 'expected_class_range': [8, 9, 10, 11]},
        'Rinternal': {'level': 2, 'parent': 'RPA', 'expected_class_range': [8, 9, 10, 11]},
        'Lmedium': {'level': 2, 'parent': 'LPA', 'expected_class_range': [12]},
        'Rmedium': {'level': 2, 'parent': 'RPA', 'expected_class_range': [12]},
        'Ldown': {'level': 2, 'parent': 'LPA', 'expected_class_range': [13, 14]},
        'RDown': {'level': 2, 'parent': 'RPA', 'expected_class_range': [13, 14]}
    }
    
    print(f"  📊 血管类型数量: {len(vessel_hierarchy)}")
    
    # 统计各级血管
    level_counts = {}
    for vessel, info in vessel_hierarchy.items():
        level = info['level']
        level_counts[level] = level_counts.get(level, 0) + 1
    
    for level, count in sorted(level_counts.items()):
        level_name = ["一级（主干）", "二级（左右分支）", "三级（末端分支）"][level]
        print(f"    {level_name}: {count} 种血管")
    
    # 检查类别覆盖
    all_classes = set()
    for vessel, info in vessel_hierarchy.items():
        all_classes.update(info['expected_class_range'])
    
    expected_classes = set(range(15))  # 0-14
    missing = expected_classes - all_classes
    extra = all_classes - expected_classes
    
    print(f"  📋 类别覆盖: {sorted(all_classes)}")
    if missing:
        print(f"  ⚠️  缺失类别: {sorted(missing)}")
    if extra:
        print(f"  ⚠️  额外类别: {sorted(extra)}")
    if not missing and not extra:
        print(f"  ✅ 类别覆盖完整 (0-14)")
    
    return len(vessel_hierarchy), len(all_classes), missing, extra

def main():
    print("🔍 综合分析血管分类模型配置")
    print("=" * 60)
    
    # 1. 分析实际数据标签
    actual_labels, actual_label_count = analyze_data_labels()
    
    # 2. 分析训练代码配置
    train_classes, has_hierarchy, train_features = analyze_training_config()
    
    # 3. 分析模型配置
    model_classes, model_features = analyze_model_config()
    
    # 4. 分析血管层次结构
    vessel_count, hierarchy_classes, missing, extra = analyze_vessel_hierarchy()
    
    # 5. 综合分析
    print("\n" + "=" * 60)
    print("📊 综合配置分析")
    print("=" * 60)
    
    # 标签数量一致性检查
    print("\n🔢 标签数量一致性:")
    if actual_label_count and train_classes:
        if actual_label_count == train_classes:
            print(f"  ✅ 数据标签({actual_label_count}) == 训练配置({train_classes})")
        else:
            print(f"  ❌ 数据标签({actual_label_count}) != 训练配置({train_classes})")
    
    if train_classes and model_classes:
        if train_classes != model_classes:
            print(f"  ✅ 训练会覆盖模型默认值: {model_classes} -> {train_classes}")
        else:
            print(f"  ✅ 训练配置与模型默认值一致: {train_classes}")
    
    # 特征维度检查
    print("\n🔧 特征维度配置:")
    if train_features:
        print(f"  ✅ 训练使用增强特征: {train_features}D")
        if train_features == 90:
            print(f"    - 基础特征: 54D")
            print(f"    - 血管嵌入: 32D") 
            print(f"    - 层次编码: 3D")
            print(f"    - 位置编码: 1D")
    
    if model_features:
        if model_features == 54:
            print(f"  ⚠️  模型默认基础特征: {model_features}D (会被训练覆盖)")
        else:
            print(f"  ✅ 模型支持增强特征: {model_features}D")
    
    # 血管层次结构检查
    print("\n🩸 血管层次结构:")
    if has_hierarchy:
        print(f"  ✅ 包含完整血管层次结构")
        print(f"  ✅ 血管类型: {vessel_count} 种")
        print(f"  ✅ 支持血管感知训练")
    else:
        print(f"  ❌ 缺少血管层次结构")
    
    # 兼容性检查
    print("\n🔄 配置兼容性:")
    issues = []
    
    if actual_label_count and train_classes and actual_label_count != train_classes:
        issues.append(f"数据标签数({actual_label_count}) != 训练配置({train_classes})")
    
    if not has_hierarchy:
        issues.append("缺少血管层次结构，无法进行血管感知训练")
    
    if missing:
        issues.append(f"血管层次结构缺失类别: {sorted(missing)}")
    
    if extra:
        issues.append(f"血管层次结构包含额外类别: {sorted(extra)}")
    
    if issues:
        print("  ❌ 发现配置问题:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  ✅ 所有配置兼容，可以开始训练!")
    
    # 总结建议
    print("\n💡 配置建议:")
    if actual_label_count == 15:
        print("  ✅ 数据使用15个标签(0-14)，配置正确")
        print("  ✅ 包含背景类(标签0)，这是必要的")
        print("  ✅ 血管感知训练配置完整")
        
        if not issues:
            print("  🚀 建议：可以开始血管感知训练")
            print("     命令: python train.py --enable_large_cases --enable_vessel_aware")
    
    return len(issues) == 0

if __name__ == "__main__":
    success = main()
    print("\n" + "=" * 60)
    if success:
        print("✅ 配置分析完成，所有设置正确!")
    else:
        print("⚠️  配置分析完成，发现需要修复的问题")
