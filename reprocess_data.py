#!/usr/bin/env python3
"""
重新处理血管数据脚本 - 使用修复后的15类血管层次结构
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
import time

# 添加src路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'preprocessing'))

def reprocess_vessel_data(args):
    """重新处理血管数据"""
    print("🔄 开始重新处理血管数据")
    print("=" * 60)
    
    # 检查路径
    ct_dir = args.ct_dir
    label_dir = args.label_dir
    output_dir = args.output_dir
    
    print(f"📂 CT数据目录: {ct_dir}")
    print(f"📂 标签数据目录: {label_dir}")
    print(f"📂 输出目录: {output_dir}")
    
    # 验证输入路径
    if not os.path.exists(ct_dir):
        print(f"❌ CT数据目录不存在: {ct_dir}")
        return False
        
    if not os.path.exists(label_dir):
        print(f"❌ 标签数据目录不存在: {label_dir}")
        return False
    
    # 备份现有数据
    if os.path.exists(output_dir) and args.backup_existing:
        backup_dir = f"{output_dir}_backup_{int(time.time())}"
        print(f"💾 备份现有数据到: {backup_dir}")
        shutil.copytree(output_dir, backup_dir)
    
    # 清理现有数据
    if args.clean_existing and os.path.exists(output_dir):
        print("🧹 清理现有处理后数据...")
        shutil.rmtree(output_dir)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 导入并创建预处理器
    try:
        from vessel_preprocessing import VesselPreprocessor
        print("✅ 成功导入血管预处理模块")
        
        preprocessor = VesselPreprocessor(
            ct_dir=ct_dir,
            label_dir=label_dir,
            output_dir=output_dir,
            cube_size=args.cube_size
        )
        print("✅ 血管预处理器已初始化")
        
        # 验证血管层次结构
        print("🔍 验证血管层次结构...")
        vessel_hierarchy = preprocessor.vessel_hierarchy
        print(f"  血管类型数: {len(vessel_hierarchy)}")
        print(f"  血管类型: {list(vessel_hierarchy.keys())}")
        
        if len(vessel_hierarchy) == 15:
            print("  ✅ 血管数量正确: 15类")
        else:
            print(f"  ⚠️ 血管数量: {len(vessel_hierarchy)}类 (期望15类)")
        
    except ImportError as e:
        print(f"❌ 导入预处理模块失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 创建预处理器失败: {e}")
        return False
    
    # 开始处理
    print(f"\n🚀 开始处理血管数据...")
    
    try:
        # 调用预处理器处理所有案例
        results = preprocessor.process_all_cases()
        
        if results:
            print(f"🎉 处理完成!")
            print(f"📊 成功处理案例数: {len(results)}")
            print(f"� 输出目录: {output_dir}")
            
            # 验证处理结果
            print(f"\n🔍 验证处理结果...")
            validate_processed_data(output_dir, args.verbose)
            
            return True
        else:
            print("❌ 未能处理任何案例")
            return False
            
    except Exception as e:
        print(f"❌ 处理过程中发生错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False

def validate_processed_data(output_dir, verbose=False):
    """验证处理后的数据"""
    import numpy as np
    
    processed_files = list(Path(output_dir).glob("*_processed.npz"))
    
    if not processed_files:
        print("⚠️ 未找到处理后的文件")
        return
    
    print(f"🔍 验证 {len(processed_files)} 个处理后的文件...")
    
    total_nodes = 0
    total_edges = 0
    vessel_types = set()
    class_ranges = set()
    valid_files = 0
    
    for file_path in processed_files:
        try:
            data = np.load(file_path, allow_pickle=True)
            
            # 基本检查
            required_keys = ['node_features', 'node_positions', 'edge_index', 
                           'node_classes', 'vessel_node_ranges']
            
            missing_keys = [key for key in required_keys if key not in data.keys()]
            if missing_keys:
                print(f"  ❌ {file_path.name}: 缺失键 {missing_keys}")
                continue
            
            # 统计信息
            node_count = len(data['node_features'])
            edge_count = data['edge_index'].shape[1] if data['edge_index'].size > 0 else 0
            vessel_ranges = data['vessel_node_ranges'].item()
            node_classes = data['node_classes']
            
            total_nodes += node_count
            total_edges += edge_count
            vessel_types.update(vessel_ranges.keys())
            class_ranges.update(np.unique(node_classes))
            valid_files += 1
            
            if verbose:
                print(f"  ✅ {file_path.name}: {node_count} 节点, {edge_count} 边, {len(vessel_ranges)} 血管")
                
        except Exception as e:
            print(f"  ❌ {file_path.name}: 验证错误 {e}")
    
    if valid_files == 0:
        print("❌ 没有有效的处理文件")
        return
    
    print(f"📊 验证统计:")
    print(f"  有效文件: {valid_files}/{len(processed_files)}")
    print(f"  总节点数: {total_nodes:,}")
    print(f"  总边数: {total_edges:,}")
    print(f"  血管类型数: {len(vessel_types)}")
    print(f"  血管类型: {sorted(vessel_types)}")
    print(f"  类别范围: {sorted(class_ranges)}")
    
    # 检查血管层次结构
    if len(vessel_types) == 15:
        print("  ✅ 血管数量正确: 15类")
    else:
        print(f"  ⚠️ 血管数量: {len(vessel_types)}类 (期望15类)")
    
    if len(class_ranges) <= 15 and min(class_ranges) >= 0:
        print(f"  ✅ 类别范围合理: {min(class_ranges)}-{max(class_ranges)}")
    else:
        print(f"  ⚠️ 类别范围异常: {min(class_ranges)}-{max(class_ranges)}")

def main():
    parser = argparse.ArgumentParser(description='重新处理血管数据')
    
    # 路径参数
    parser.add_argument('--ct_dir', type=str, 
                       default='/home/lihe/classify/lungmap/data/raw/train',
                       help='CT数据目录')
    parser.add_argument('--label_dir', type=str,
                       default='/home/lihe/classify/lungmap/data/raw/label_filtered',
                       help='标签数据目录')
    parser.add_argument('--output_dir', type=str,
                       default='/home/lihe/classify/lungmap/data/processed',
                       help='处理后数据输出目录')
    
    # 处理参数
    parser.add_argument('--cube_size', type=int, default=32,
                       help='图像块大小')
    
    # 控制参数
    parser.add_argument('--clean_existing', action='store_true',
                       help='清理现有的处理后数据')
    parser.add_argument('--backup_existing', action='store_true',
                       help='备份现有的处理后数据')
    parser.add_argument('--verbose', action='store_true',
                       help='详细输出处理信息')
    
    args = parser.parse_args()
    
    # 显示配置
    print("🔧 数据重新处理配置:")
    print(f"  CT数据目录: {args.ct_dir}")
    print(f"  标签数据目录: {args.label_dir}")
    print(f"  输出数据目录: {args.output_dir}")
    print(f"  图像块大小: {args.cube_size}")
    print(f"  清理现有数据: {'是' if args.clean_existing else '否'}")
    print(f"  备份现有数据: {'是' if args.backup_existing else '否'}")
    
    # 确认操作
    if args.clean_existing:
        response = input("⚠️ 确定要清理现有的处理后数据吗？这将删除所有现有的.npz文件 (y/N): ")
        if response.lower() != 'y':
            print("❌ 操作已取消")
            return
    
    # 开始处理
    success = reprocess_vessel_data(args)
    
    if success:
        print("\n🎉 数据重新处理完成!")
        print("📝 建议接下来:")
        print("  1. 运行训练前验证: python validate_processed_data.py")
        print("  2. 检查数据一致性: python test_all_vessel_consistency.py") 
        print("  3. 开始训练: python train.py --enable_large_cases")
    else:
        print("\n❌ 数据重新处理失败!")
        sys.exit(1)

if __name__ == "__main__":
    main()
