#!/usr/bin/env python3
"""
CPR-TaG-Net 优化训练脚本 - 24GB显存专用版本
"""

import os
import subprocess
import sys

def run_training():
    """运行优化的CPR-TaG-Net训练"""
    
    print("🚀 启动CPR-TaG-Net 24GB显存优化训练")
    print("=" * 50)
    
    # 24GB显存优化配置
    cmd = [
        "python", "train.py",
        "--enable_large_cases",
        "--max_nodes_per_case", "6000",  # 避免超大案例
        "--node_batch_size", "300",      # 适中的批大小
        "--epochs", "25",                # 充足的训练轮数
        "--learning_rate", "0.0005",     # 稍微保守的学习率
        "--weight_decay", "5e-5",        # 适度正则化
        "--step_size", "15",             # 学习率衰减
        "--gamma", "0.7"                 # 温和衰减
    ]
    
    # 设置环境变量
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "2"  # 使用GPU 2
    
    print(f"Command: {' '.join(cmd)}")
    print(f"GPU: {env.get('CUDA_VISIBLE_DEVICES', 'default')}")
    print("=" * 50)
    
    try:
        # 运行训练
        result = subprocess.run(cmd, env=env, check=True)
        print("\n🎉 训练成功完成!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 训练失败: {e}")
        return False
        
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
        return False

def show_gpu_info():
    """显示GPU信息"""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        print("💻 GPU状态:")
        print(result.stdout)
    except:
        print("⚠️  无法获取GPU信息")

if __name__ == "__main__":
    show_gpu_info()
    
    # 确认是否继续
    response = input("\n继续训练? (y/N): ").strip().lower()
    if response in ['y', 'yes']:
        success = run_training()
        if success:
            print("\n✅ 训练完成! 检查输出目录:")
            print("  - 模型: /home/lihe/classify/lungmap/outputs/checkpoints/")
            print("  - 日志: /home/lihe/classify/lungmap/outputs/logs/")
        else:
            print("\n❌ 训练未完成，请检查错误信息")
    else:
        print("训练取消")
