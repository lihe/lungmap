#!/usr/bin/env python3
"""
训练状态检查和监控脚本
"""

import os
import glob
import time
from datetime import datetime
import subprocess

def check_gpu_status():
    """检查GPU使用状态"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines):
                used, total, util = line.split(', ')
                used_gb = int(used) / 1024
                total_gb = int(total) / 1024
                print(f"🖥️  GPU {i}: {used_gb:.1f}GB / {total_gb:.1f}GB ({util}% 使用率)")
        else:
            print("⚠️  无法获取GPU状态")
    except:
        print("⚠️  nvidia-smi 不可用")

def check_training_logs():
    """检查训练日志"""
    log_dir = "outputs/logs"
    if not os.path.exists(log_dir):
        print("📝 没有找到训练日志目录")
        return
    
    # 查找最新的日志目录
    log_folders = glob.glob(os.path.join(log_dir, "cpr_tagnet_training_*"))
    if not log_folders:
        print("📝 没有找到训练日志")
        return
    
    latest_log = max(log_folders, key=os.path.getctime)
    print(f"📝 最新训练日志: {os.path.basename(latest_log)}")
    
    # 检查日志文件
    log_files = os.listdir(latest_log)
    if log_files:
        print("   日志文件:")
        for f in sorted(log_files):
            file_path = os.path.join(latest_log, f)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path) / 1024  # KB
                mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                print(f"     {f} ({size:.1f}KB, {mtime.strftime('%H:%M:%S')})")

def check_checkpoints():
    """检查模型检查点"""
    checkpoint_dir = "outputs/checkpoints"
    if not os.path.exists(checkpoint_dir):
        print("💾 没有找到检查点目录")
        return
    
    checkpoints = os.listdir(checkpoint_dir)
    if not checkpoints:
        print("💾 没有找到检查点文件")
        return
    
    print("💾 模型检查点:")
    for ckpt in sorted(checkpoints):
        ckpt_path = os.path.join(checkpoint_dir, ckpt)
        if os.path.isfile(ckpt_path):
            size = os.path.getsize(ckpt_path) / 1024 / 1024  # MB
            mtime = datetime.fromtimestamp(os.path.getmtime(ckpt_path))
            print(f"     {ckpt} ({size:.1f}MB, {mtime.strftime('%Y-%m-%d %H:%M:%S')})")

def check_visualizations():
    """检查可视化结果"""
    viz_dir = "outputs/visualizations"
    if not os.path.exists(viz_dir):
        print("📊 没有找到可视化目录")
        return
    
    viz_files = os.listdir(viz_dir)
    if not viz_files:
        print("📊 没有找到可视化文件")
        return
    
    print("📊 可视化结果:")
    for f in sorted(viz_files):
        if f.endswith(('.png', '.jpg', '.txt')):
            file_path = os.path.join(viz_dir, f)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path) / 1024  # KB
                mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                print(f"     {f} ({size:.1f}KB, {mtime.strftime('%H:%M:%S')})")

def check_running_processes():
    """检查正在运行的训练进程"""
    try:
        result = subprocess.run(['pgrep', '-f', 'train.py'], capture_output=True, text=True)
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            print(f"🏃 发现 {len(pids)} 个训练进程:")
            for pid in pids:
                if pid:
                    print(f"     PID: {pid}")
        else:
            print("🏃 没有发现正在运行的训练进程")
    except:
        print("⚠️  无法检查进程状态")

def show_recent_log_content():
    """显示最近的日志内容"""
    log_dir = "outputs/logs"
    if not os.path.exists(log_dir):
        return
    
    log_folders = glob.glob(os.path.join(log_dir, "cpr_tagnet_training_*"))
    if not log_folders:
        return
    
    latest_log = max(log_folders, key=os.path.getctime)
    
    # 查找主要日志文件
    main_log = None
    for f in os.listdir(latest_log):
        if f.endswith('.txt') and 'training' in f.lower():
            main_log = os.path.join(latest_log, f)
            break
    
    if main_log and os.path.exists(main_log):
        print(f"\n📖 最近的训练日志 (最后20行):")
        print("-" * 50)
        try:
            with open(main_log, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines[-20:]:
                    print(line.rstrip())
        except:
            print("无法读取日志文件")
        print("-" * 50)

def main():
    print("🔍 CPR-TaG-Net 训练状态检查")
    print("=" * 50)
    print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 检查各个方面
    check_running_processes()
    print()
    
    check_gpu_status()
    print()
    
    check_training_logs()
    print()
    
    check_checkpoints()
    print()
    
    check_visualizations()
    print()
    
    # 显示最近日志内容
    show_recent_log_content()

if __name__ == "__main__":
    main()
