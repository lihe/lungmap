#!/usr/bin/env python3
"""
检测血管预处理数据中的潜在bug
"""
import numpy as np
import os
from collections import Counter, defaultdict

def detect_preprocessing_bugs():
    """检测预处理过程中的潜在bug"""
    npz_dir = "/home/lihe/classify/lungmap/data/processed"
    all_files = [f for f in os.listdir(npz_dir) if f.endswith('.npz')]
    
    print("🐛 血管预处理Bug检测")
    print("=" * 70)
    
    suspicious_cases = []
    all_cases_info = []
    
    for filename in all_files:
        filepath = os.path.join(npz_dir, filename)
        
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
            
            case_info = {
                'filename': filename,
                'case_id': case_id,
                'total_nodes': len(node_features),
                'num_vessels': len(vessel_ranges),
                'vessel_ranges': vessel_ranges,
                'node_classes': node_classes,
                'node_positions': node_positions,
                'edges': edge_index.shape[1],
                'vessel_node_counts': {}
            }
            
            # 记录每个血管的节点数
            for vessel_name, (start, end) in vessel_ranges.items():
                node_count = end - start + 1
                case_info['vessel_node_counts'][vessel_name] = node_count
            
            all_cases_info.append(case_info)
            
        except Exception as e:
            print(f"❌ 读取 {filename} 失败: {e}")
            continue
    
    print(f"✅ 成功读取 {len(all_cases_info)} 个案例\n")
    
    # Bug检测1: 异常高的节点数
    print("🔍 Bug检测1: 异常高的节点数")
    print("-" * 50)
    node_counts = [info['total_nodes'] for info in all_cases_info]
    mean_nodes = np.mean(node_counts)
    std_nodes = np.std(node_counts)
    threshold = mean_nodes + 2 * std_nodes
    
    print(f"平均节点数: {mean_nodes:.1f}")
    print(f"标准差: {std_nodes:.1f}")
    print(f"异常阈值 (均值+2σ): {threshold:.1f}")
    
    for info in all_cases_info:
        if info['total_nodes'] > threshold:
            suspicious_cases.append((info['filename'], 'high_node_count', info['total_nodes']))
            print(f"⚠️  {info['filename']}: {info['total_nodes']} 节点 (异常高)")
    
    # Bug检测2: 血管节点数分布异常
    print(f"\n🔍 Bug检测2: 血管节点数分布异常")
    print("-" * 50)
    
    vessel_node_patterns = defaultdict(list)
    for info in all_cases_info:
        for vessel_name, node_count in info['vessel_node_counts'].items():
            vessel_node_patterns[vessel_name].append(node_count)
    
    # 检查是否有血管的节点数过于一致（可能是bug）
    for vessel_name, counts in vessel_node_patterns.items():
        if len(counts) > 3:  # 至少3个案例才分析
            unique_counts = set(counts)
            if len(unique_counts) == 1 and counts[0] > 50:
                print(f"🔴 {vessel_name}: 所有案例都是 {counts[0]} 个节点 (过于一致)")
                for info in all_cases_info:
                    if vessel_name in info['vessel_node_counts'] and info['vessel_node_counts'][vessel_name] == counts[0]:
                        suspicious_cases.append((info['filename'], 'uniform_vessel_nodes', f"{vessel_name}:{counts[0]}"))
    
    # Bug检测3: 检查特定的可疑模式
    print(f"\n🔍 Bug检测3: 特定可疑模式检测")
    print("-" * 50)
    
    for info in all_cases_info:
        vessel_counts = list(info['vessel_node_counts'].values())
        
        # 模式1: 所有血管节点数相同且很大
        if len(set(vessel_counts)) == 1 and vessel_counts[0] > 80:
            print(f"🔴 {info['filename']}: 所有血管都有 {vessel_counts[0]} 个节点 (异常统一)")
            suspicious_cases.append((info['filename'], 'all_vessels_same_size', vessel_counts[0]))
        
        # 模式2: 节点数是特定的整数倍
        suspicious_multiples = [90, 99, 100]
        for mult in suspicious_multiples:
            if any(count == mult for count in vessel_counts):
                vessels_with_mult = [v for v, c in info['vessel_node_counts'].items() if c == mult]
                if len(vessels_with_mult) > 1:
                    print(f"🔴 {info['filename']}: 多个血管都有 {mult} 个节点: {vessels_with_mult}")
                    suspicious_cases.append((info['filename'], f'multiple_vessels_{mult}', vessels_with_mult))
    
    # Bug检测4: 检查节点类别一致性
    print(f"\n🔍 Bug检测4: 节点类别一致性")
    print("-" * 50)
    
    for info in all_cases_info:
        for vessel_name, (start, end) in info['vessel_ranges'].items():
            vessel_classes = info['node_classes'][start:end+1]
            unique_classes = np.unique(vessel_classes)
            
            # 每个血管内的节点应该有相同的类别
            if len(unique_classes) > 1:
                print(f"🔴 {info['filename']} - {vessel_name}: 血管内有多个类别 {unique_classes}")
                suspicious_cases.append((info['filename'], 'mixed_classes_in_vessel', f"{vessel_name}:{unique_classes}"))
    
    # Bug检测5: 检查空间分布异常
    print(f"\n🔍 Bug检测5: 空间分布异常")
    print("-" * 50)
    
    for info in all_cases_info:
        positions = info['node_positions']
        
        # 检查是否有重复的坐标点
        unique_positions = np.unique(positions, axis=0)
        if len(unique_positions) < len(positions):
            duplicate_count = len(positions) - len(unique_positions)
            print(f"🔴 {info['filename']}: 有 {duplicate_count} 个重复坐标点")
            suspicious_cases.append((info['filename'], 'duplicate_positions', duplicate_count))
        
        # 检查坐标是否过于规律
        for vessel_name, (start, end) in info['vessel_ranges'].items():
            vessel_positions = positions[start:end+1]
            if len(vessel_positions) > 10:  # 只检查节点数较多的血管
                # 检查X坐标的间隔是否过于规律
                x_coords = vessel_positions[:, 0]
                x_diffs = np.diff(np.sort(x_coords))
                if len(np.unique(np.round(x_diffs, 1))) == 1 and len(x_diffs) > 5:
                    print(f"🔴 {info['filename']} - {vessel_name}: X坐标间隔过于规律 ({x_diffs[0]:.1f})")
                    suspicious_cases.append((info['filename'], 'regular_spacing', f"{vessel_name}:X"))
    
    # Bug检测6: 检查边连接异常
    print(f"\n🔍 Bug检测6: 边连接异常")
    print("-" * 50)
    
    for info in all_cases_info:
        total_nodes = info['total_nodes']
        total_edges = info['edges']
        
        # 对于连通图，边数应该是节点数-1
        if total_edges != total_nodes - 1:
            print(f"🔴 {info['filename']}: 边数异常 - 节点:{total_nodes}, 边:{total_edges} (期望:{total_nodes-1})")
            suspicious_cases.append((info['filename'], 'edge_count_mismatch', f"nodes:{total_nodes},edges:{total_edges}"))
    
    # 总结
    print(f"\n📊 Bug检测总结")
    print("=" * 70)
    
    if suspicious_cases:
        print(f"发现 {len(suspicious_cases)} 个可疑情况:")
        
        bug_types = defaultdict(list)
        for filename, bug_type, details in suspicious_cases:
            bug_types[bug_type].append((filename, details))
        
        for bug_type, cases in bug_types.items():
            print(f"\n🔴 {bug_type}:")
            for filename, details in cases:
                print(f"   {filename}: {details}")
        
        # 重点分析最可疑的案例
        print(f"\n🎯 最可疑的案例:")
        case_suspicion_count = defaultdict(int)
        for filename, _, _ in suspicious_cases:
            case_suspicion_count[filename] += 1
        
        most_suspicious = sorted(case_suspicion_count.items(), key=lambda x: x[1], reverse=True)
        for filename, count in most_suspicious[:5]:
            print(f"   {filename}: {count} 个异常")
    else:
        print("✅ 未发现明显的bug")
    
    return suspicious_cases, all_cases_info

def analyze_specific_large_cases():
    """专门分析三个大节点案例的详细bug"""
    print(f"\n🔬 专门分析大节点案例")
    print("=" * 70)
    
    npz_dir = "/home/lihe/classify/lungmap/data/processed"
    large_cases = ["4000010084_processed.npz", "4000010082_processed.npz", "3000020023_processed.npz"]
    
    for filename in large_cases:
        filepath = os.path.join(npz_dir, filename)
        if not os.path.exists(filepath):
            continue
            
        print(f"\n🔍 深度分析: {filename}")
        print("-" * 40)
        
        data = np.load(filepath, allow_pickle=True)
        
        case_id = data['case_id'].item()
        node_features = data['node_features']
        node_positions = data['node_positions']
        vessel_ranges = data['vessel_node_ranges'].item()
        node_classes = data['node_classes']
        
        # 检查1: 血管节点数的整数性
        vessel_counts = []
        for vessel_name, (start, end) in vessel_ranges.items():
            count = end - start + 1
            vessel_counts.append(count)
            print(f"  {vessel_name}: {count} 个节点")
        
        # 检查是否所有血管节点数都是特定值
        if len(set(vessel_counts)) == 1:
            print(f"  🔴 BUG: 所有血管都有相同的节点数 {vessel_counts[0]}")
        
        # 检查2: 空间坐标分析
        print(f"\n  坐标分析:")
        for vessel_name, (start, end) in vessel_ranges.items():
            vessel_pos = node_positions[start:end+1]
            
            # 检查坐标是否有规律
            x_range = vessel_pos[:, 0].max() - vessel_pos[:, 0].min()
            y_range = vessel_pos[:, 1].max() - vessel_pos[:, 1].min()
            z_range = vessel_pos[:, 2].max() - vessel_pos[:, 2].min()
            
            print(f"    {vessel_name}: X跨度={x_range:.1f}, Y跨度={y_range:.1f}, Z跨度={z_range:.1f}")
            
            # 检查是否有重复坐标
            unique_pos = np.unique(vessel_pos, axis=0)
            if len(unique_pos) != len(vessel_pos):
                duplicate_count = len(vessel_pos) - len(unique_pos)
                print(f"    🔴 BUG: {vessel_name} 有 {duplicate_count} 个重复坐标")
        
        # 检查3: 特征分析
        print(f"\n  特征分析:")
        for vessel_name, (start, end) in vessel_ranges.items():
            vessel_features = node_features[start:end+1]
            
            # 检查特征是否完全相同
            unique_features = np.unique(vessel_features, axis=0)
            if len(unique_features) == 1:
                print(f"    🔴 BUG: {vessel_name} 所有节点特征完全相同")
            elif len(unique_features) < len(vessel_features) * 0.8:
                similarity = len(unique_features) / len(vessel_features)
                print(f"    ⚠️  {vessel_name} 特征相似度过高 ({similarity:.2f})")

if __name__ == "__main__":
    suspicious_cases, all_cases_info = detect_preprocessing_bugs()
    analyze_specific_large_cases()
