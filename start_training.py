#!/usr/bin/env python3
"""
血管分类训练启动脚本
简化版本，便于快速开始训练
"""

import subprocess
import sys
import os

def check_gpu():
    """检查GPU状态"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if 'MiB' in line and '/' in line:
                    # 提取显存信息
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'MiB' in part and i > 0:
                            used = parts[i-1]
                            total = part.replace('MiB', '')
                            print(f"📊 GPU显存: {used}MiB / {total}MiB")
                            total_gb = int(total) / 1024
                            if total_gb >= 20:
                                print(f"✅ 检测到 {total_gb:.1f}GB 显存，可以启用大案例训练")
                                return True
                            else:
                                print(f"⚠️  显存较小 ({total_gb:.1f}GB)，建议使用保守模式")
                                return False
    except:
        print("⚠️  无法检测GPU状态")
        return False

def main():
    print("🚀 CPR-TaG-Net 血管分类训练启动器")
    print("=" * 50)
    
    # 检查当前目录
    if not os.path.exists('train.py'):
        print("❌ 请在 lungmap 项目根目录下运行此脚本")
        sys.exit(1)
    
    # 检查数据
    if not os.path.exists('data/processed'):
        print("❌ 数据目录不存在: data/processed")
        sys.exit(1)
    
    processed_files = os.listdir('data/processed')
    npz_files = [f for f in processed_files if f.endswith('.npz')]
    print(f"📁 找到 {len(npz_files)} 个预处理数据文件")
    
    if len(npz_files) == 0:
        print("❌ 没有找到预处理数据文件 (.npz)")
        sys.exit(1)
    
    # 检查GPU
    has_large_gpu = check_gpu()
    
    print("\n🔧 训练配置选项:")
    print("1. 快速测试模式 (小数据集, 10轮)")
    print("2. 标准训练模式 (中等数据集, 50轮)")
    print("3. 完整训练模式 (大数据集, 100轮) - 需要大显存")
    print("4. 自定义参数")
    
    choice = input("\n请选择训练模式 [1-4]: ").strip()
    
    # 基础命令
    base_cmd = [
        'conda', 'run', '--live-stream', '--name', 'lungmap', 'python', 'train.py'
    ]
    
    if choice == '1':
        # 快速测试模式
        cmd = base_cmd + [
            '--epochs', '10',
            '--max_nodes', '500',
            '--node_batch_size', '200',
            '--save_freq', '2',
            '--enable_visualization',
            '--save_training_curves'
        ]
        print("🔧 使用快速测试模式")
        
    elif choice == '2':
        # 标准训练模式
        cmd = base_cmd + [
            '--epochs', '50',
            '--max_nodes', '2000',
            '--node_batch_size', '300',
            '--save_freq', '5',
            '--enable_vessel_aware',
            '--enable_visualization',
            '--save_training_curves',
            '--save_confusion_matrix'
        ]
        print("🔧 使用标准训练模式")
        
    elif choice == '3':
        # 完整训练模式
        if not has_large_gpu:
            print("⚠️  检测到显存可能不足，建议选择模式1或2")
            confirm = input("是否继续? [y/N]: ").strip().lower()
            if confirm != 'y':
                return
        
        cmd = base_cmd + [
            '--epochs', '100',
            '--enable_large_cases',
            '--max_nodes_per_case', '8000',
            '--node_batch_size', '500',
            '--save_freq', '10',
            '--enable_vessel_aware',
            '--enable_visualization',
            '--save_training_curves',
            '--save_confusion_matrix',
            '--enable_graph_completion'
        ]
        print("🔧 使用完整训练模式")
        
    elif choice == '4':
        # 自定义参数
        print("\n自定义训练参数:")
        epochs = input("训练轮数 [50]: ").strip() or '50'
        batch_size = input("节点批大小 [300]: ").strip() or '300'
        lr = input("学习率 [0.001]: ").strip() or '0.001'
        
        cmd = base_cmd + [
            '--epochs', epochs,
            '--node_batch_size', batch_size,
            '--learning_rate', lr,
            '--enable_vessel_aware',
            '--enable_visualization',
            '--save_training_curves'
        ]
        
        if has_large_gpu:
            use_large = input("启用大案例训练? [y/N]: ").strip().lower()
            if use_large == 'y':
                cmd.extend(['--enable_large_cases', '--max_nodes_per_case', '8000'])
        
        print("🔧 使用自定义配置")
        
    else:
        print("❌ 无效选择")
        return
    
    # 显示完整命令
    print(f"\n🎯 执行命令:")
    print(' '.join(cmd))
    
    # 确认开始训练
    print(f"\n🚀 准备开始训练...")
    confirm = input("按 Enter 开始训练，或 Ctrl+C 取消: ")
    
    try:
        # 执行训练命令
        subprocess.run(cmd, check=True)
        print("\n🎉 训练完成!")
        
    except KeyboardInterrupt:
        print("\n⏹️  训练被用户中断")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 训练过程中出现错误: {e}")
        print("请检查错误信息并重试")

if __name__ == "__main__":
    main()
