#!/usr/bin/env python3
"""
详细分析重复坐标和特征的bug
"""
import numpy as np
import os

def analyze_coordinate_duplication():
    """分析坐标重复的详细情况"""
    npz_dir = "/home/lihe/classify/lungmap/data/processed"
    problematic_cases = ["4000010084_processed.npz", "4000010082_processed.npz", "3000020023_processed.npz"]
    
    print("🔍 详细分析坐标重复Bug")
    print("=" * 70)
    
    for filename in problematic_cases:
        filepath = os.path.join(npz_dir, filename)
        if not os.path.exists(filepath):
            continue
            
        print(f"\n📋 分析: {filename}")
        print("-" * 50)
        
        data = np.load(filepath, allow_pickle=True)
        
        case_id = data['case_id'].item()
        node_positions = data['node_positions']  
        vessel_ranges = data['vessel_node_ranges'].item()
        node_features = data['node_features']
        
        print(f"总节点数: {len(node_positions)}")
        
        # 检查全局坐标重复
        unique_positions = np.unique(node_positions, axis=0)
        total_duplicates = len(node_positions) - len(unique_positions)
        print(f"重复坐标点: {total_duplicates} / {len(node_positions)} ({total_duplicates/len(node_positions)*100:.1f}%)")
        
        # 分血管检查
        for vessel_name, (start, end) in vessel_ranges.items():
            vessel_positions = node_positions[start:end+1]
            vessel_features = node_features[start:end+1]
            
            print(f"\n  {vessel_name} 血管:")
            print(f"    节点范围: {start}-{end} ({end-start+1}个)")
            
            # 检查坐标重复
            unique_vessel_pos = np.unique(vessel_positions, axis=0)
            vessel_duplicates = len(vessel_positions) - len(unique_vessel_pos)
            if vessel_duplicates > 0:
                print(f"    🔴 坐标重复: {vessel_duplicates} 个")
                
                # 找出重复的坐标
                from collections import Counter
                pos_strings = [f"{p[0]:.1f},{p[1]:.1f},{p[2]:.1f}" for p in vessel_positions]
                pos_counts = Counter(pos_strings)
                duplicated_coords = {coord: count for coord, count in pos_counts.items() if count > 1}
                
                for coord, count in duplicated_coords.items():
                    print(f"      重复坐标 {coord}: 出现 {count} 次")
            
            # 检查特征重复
            unique_vessel_features = np.unique(vessel_features, axis=0)
            feature_duplicates = len(vessel_features) - len(unique_vessel_features)
            if feature_duplicates > 0:
                print(f"    🔴 特征重复: {feature_duplicates} 个")
            
            # 检查坐标的分布模式
            x_coords = vessel_positions[:, 0]
            y_coords = vessel_positions[:, 1] 
            z_coords = vessel_positions[:, 2]
            
            # 检查是否有规律的间隔
            x_sorted = np.sort(x_coords)
            x_diffs = np.diff(x_sorted)
            x_diffs = x_diffs[x_diffs > 0.001]  # 忽略极小的差异
            
            if len(x_diffs) > 0:
                x_diff_std = np.std(x_diffs)
                x_diff_mean = np.mean(x_diffs)
                
                if x_diff_std < 0.1 and len(x_diffs) > 5:
                    print(f"    ⚠️  X坐标间隔过于规律: 平均间隔 {x_diff_mean:.2f} ± {x_diff_std:.3f}")
            
            # 检查坐标是否完全相同
            if len(unique_vessel_pos) == 1:
                print(f"    🔴 严重BUG: 所有坐标点完全相同 {unique_vessel_pos[0]}")
            
            # 检查特征的分布
            feature_means = np.mean(vessel_features, axis=0)
            feature_stds = np.std(vessel_features, axis=0)
            
            zero_std_features = np.sum(feature_stds < 1e-6)
            if zero_std_features > 0:
                print(f"    🔴 特征BUG: {zero_std_features} 个特征维度标准差为0 (特征完全相同)")
            
            low_variance_features = np.sum(feature_stds < 0.01)
            if low_variance_features > 10:
                print(f"    ⚠️  {low_variance_features} 个特征维度方差很小")

def analyze_preprocessing_pattern():
    """分析预处理的具体模式"""
    print(f"\n🔬 分析预处理模式")
    print("=" * 70)
    
    npz_dir = "/home/lihe/classify/lungmap/data/processed"
    problematic_cases = ["4000010084_processed.npz", "4000010082_processed.npz", "3000020023_processed.npz"]
    
    for filename in problematic_cases:
        filepath = os.path.join(npz_dir, filename)
        if not os.path.exists(filepath):
            continue
            
        print(f"\n📋 预处理模式分析: {filename}")
        print("-" * 50)
        
        data = np.load(filepath, allow_pickle=True)
        
        vessel_ranges = data['vessel_node_ranges'].item()
        node_positions = data['node_positions']
        
        # 检查节点数模式
        vessel_node_counts = {}
        for vessel_name, (start, end) in vessel_ranges.items():
            count = end - start + 1
            vessel_node_counts[vessel_name] = count
        
        all_counts = list(vessel_node_counts.values())
        if len(set(all_counts)) == 1:
            print(f"🔴 模式异常: 所有血管都有 {all_counts[0]} 个节点")
            
            # 分析这个数字是否有特殊含义
            target_count = all_counts[0]
            
            # 检查是否是简单的等分
            total_unique_positions = len(np.unique(node_positions, axis=0))
            if total_unique_positions * len(vessel_ranges) == len(node_positions):
                print(f"  🔍 可能的原因: 将 {total_unique_positions} 个唯一位置复制到 {len(vessel_ranges)} 个血管")
            
            # 检查是否是固定采样数
            common_sampling_nums = [50, 64, 90, 99, 100, 128, 256]
            if target_count in common_sampling_nums:
                print(f"  🔍 可能的原因: 固定采样到 {target_count} 个点")
            
            # 检查总数是否有规律
            total_nodes = len(node_positions)
            if total_nodes % len(vessel_ranges) == 0:
                nodes_per_vessel = total_nodes // len(vessel_ranges)
                print(f"  🔍 发现规律: 总节点数 {total_nodes} 被平均分配给 {len(vessel_ranges)} 个血管")
        
        # 检查坐标范围的相同性
        print(f"\n  坐标范围分析:")
        vessel_ranges_coords = {}
        for vessel_name, (start, end) in vessel_ranges.items():
            vessel_pos = node_positions[start:end+1]
            x_range = vessel_pos[:, 0].max() - vessel_pos[:, 0].min()
            y_range = vessel_pos[:, 1].max() - vessel_pos[:, 1].min()
            z_range = vessel_pos[:, 2].max() - vessel_pos[:, 2].min()
            vessel_ranges_coords[vessel_name] = (x_range, y_range, z_range)
            print(f"    {vessel_name}: X={x_range:.1f}, Y={y_range:.1f}, Z={z_range:.1f}")
        
        # 检查所有血管的坐标范围是否相同
        all_ranges = list(vessel_ranges_coords.values())
        if len(set([(round(r[0], 1), round(r[1], 1), round(r[2], 1)) for r in all_ranges])) == 1:
            print(f"  🔴 BUG: 所有血管的坐标范围完全相同!")
            print(f"       这表明所有血管可能共享同一组坐标点")

def suggest_fixes():
    """建议修复方案"""
    print(f"\n💡 Bug修复建议")
    print("=" * 70)
    
    print("根据分析结果，发现的主要Bug类型:")
    print()
    print("1. 🔴 所有血管节点数完全相同")
    print("   - 原因: 预处理算法可能错误地对所有血管应用了相同的采样策略")
    print("   - 影响: 违反了血管系统的自然层次结构")
    print("   - 修复: 根据血管类型和复杂度自适应采样")
    print()
    print("2. 🔴 大量重复坐标点") 
    print("   - 原因: 预处理过程中可能将相同的点集复制给了多个血管")
    print("   - 影响: 图结构不正确，训练时会产生错误的几何关系")
    print("   - 修复: 确保每个节点只属于一个血管，避免重复")
    print()
    print("3. 🔴 所有血管坐标范围相同")
    print("   - 原因: 可能所有血管共享同一个边界框内的采样点")
    print("   - 影响: 失去了血管间的空间区分度")
    print("   - 修复: 为每个血管独立提取其特定区域的点")
    print()
    print("建议的处理策略:")
    print("✅ 短期: 过滤掉这3个异常案例，用正常的21个案例进行训练")
    print("✅ 长期: 重新设计预处理算法，确保血管间的独立性和层次性")
    print("✅ 验证: 对所有案例运行一致性检查，确保数据质量")

if __name__ == "__main__":
    analyze_coordinate_duplication()
    analyze_preprocessing_pattern()
    suggest_fixes()
