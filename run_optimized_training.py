#!/usr/bin/env python3
"""
CPR-TaG-Net 优化训练脚本 - 集成高级功能版本
"""

import os
import subprocess
import sys
from config_manager import get_config_manager

def run_training():
    """运行优化的CPR-TaG-Net增强训练"""
    
    print("🚀 启动CPR-TaG-Net 增强训练 (集成图形补全+可视化)")
    print("=" * 60)
    
    # 加载统一配置
    config_mgr = get_config_manager()
    config_mgr.print_config_summary()
    
    # 从配置生成命令行参数，添加增强功能标志
    cmd = ["python", "train.py"] + config_mgr.get_command_line_args() + [
        "--enable_graph_completion",      # 启用图形补全
        "--enable_visualization",         # 启用可视化
        "--save_confusion_matrix",        # 保存混淆矩阵
        "--save_training_curves"          # 保存训练曲线
    ]
    
    # 设置环境变量
    env = os.environ.copy()
    env.update(config_mgr.get_env_vars())
    
    print(f"\nCommand: {' '.join(cmd)}")
    print(f"GPU: {env.get('CUDA_VISIBLE_DEVICES', 'default')}")
    print("🔧 增强功能: 图形补全 + 混淆矩阵 + 训练可视化")
    print("=" * 60)
    
    try:
        # 运行训练
        result = subprocess.run(cmd, env=env, check=True)
        print("\n🎉 增强训练成功完成!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\\n❌ 训练失败: {e}")
        return False
        
    except KeyboardInterrupt:
        print("\\n⚠️  训练被用户中断")
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
    try:
        show_gpu_info()
        
        print("\n🧠 增强功能说明:")
        print("✅ 图形补全: 基于血管连续性优化分类结果")
        print("✅ 混淆矩阵: 18类血管分类错误分析")
        print("✅ 训练可视化: 损失和准确率曲线")
        print("✅ 预测质量分析: 困难类别识别")
        
        # 确认是否继续
        response = input("\n继续增强训练? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            success = run_training()
            if success:
                print("\n✅ 增强训练完成! 检查输出目录:")
                print("  - 模型: /home/lihe/classify/lungmap/outputs/checkpoints/")
                print("  - 日志: /home/lihe/classify/lungmap/outputs/logs/")
                print("  - 可视化: /home/lihe/classify/lungmap/outputs/visualizations/")
            else:
                print("\n❌ 训练未完成，请检查错误信息")
        else:
            print("训练取消")
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        print("请检查配置文件是否存在且格式正确")
