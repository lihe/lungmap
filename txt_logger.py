#!/usr/bin/env python3
"""
TXT格式训练日志记录器 - 替代TensorBoard
"""

import os
import time
from datetime import datetime
from pathlib import Path
import json

class TxtLogger:
    """TXT格式训练日志记录器"""
    
    def __init__(self, log_dir, experiment_name="cpr_tagnet_training"):
        """
        初始化日志记录器
        Args:
            log_dir: 日志保存目录
            experiment_name: 实验名称
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建带时间戳的实验目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.log_dir / f"{experiment_name}_{timestamp}"
        self.experiment_dir.mkdir(exist_ok=True)
        
        # 日志文件路径
        self.train_log_file = self.experiment_dir / "training_log.txt"
        self.metric_log_file = self.experiment_dir / "metrics.txt"
        self.config_file = self.experiment_dir / "config.json"
        
        # 内存中保存指标数据
        self.metrics = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': [],
            'epochs': []
        }
        
        # 初始化日志文件
        self._init_log_files()
        
        print(f"📝 TXT日志初始化完成")
        print(f"   实验目录: {self.experiment_dir}")
        print(f"   训练日志: {self.train_log_file.name}")
        print(f"   指标日志: {self.metric_log_file.name}")
    
    def _init_log_files(self):
        """初始化日志文件"""
        # 训练日志文件头
        with open(self.train_log_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("CPR-TaG-Net 训练日志\n")
            f.write(f"实验开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
        
        # 指标日志文件头
        with open(self.metric_log_file, 'w', encoding='utf-8') as f:
            f.write("# CPR-TaG-Net 训练指标记录\n")
            f.write("# 格式: Epoch, Train_Loss, Train_Acc, Val_Loss, Val_Acc, Learning_Rate, Timestamp\n")
            f.write("Epoch\tTrain_Loss\tTrain_Acc\tVal_Loss\tVal_Acc\tLearning_Rate\tTimestamp\n")
    
    def log_message(self, message, level="INFO"):
        """记录文本消息"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}\n"
        
        # 写入训练日志
        with open(self.train_log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        
        # 同时输出到控制台
        print(f"📝 {message}")
    
    def log_config(self, config_dict):
        """记录配置信息"""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        self.log_message(f"配置信息已保存到 {self.config_file.name}")
    
    def add_scalar(self, tag, value, step):
        """
        记录标量指标 (兼容TensorBoard接口)
        Args:
            tag: 指标标签 (如 'Train/Loss', 'Val/Accuracy')
            value: 指标值
            step: 步数 (通常是epoch)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 解析标签
        if '/' in tag:
            category, metric = tag.split('/', 1)
        else:
            category, metric = 'General', tag
        
        # 记录到内存
        if tag == 'Train/EpochLoss':
            self.metrics['train_loss'].append((step, value))
        elif tag == 'Train/EpochAccuracy':
            self.metrics['train_accuracy'].append((step, value))
        elif tag == 'Val/EpochLoss':
            self.metrics['val_loss'].append((step, value))
        elif tag == 'Val/EpochAccuracy':
            self.metrics['val_accuracy'].append((step, value))
        elif 'LearningRate' in tag:
            self.metrics['learning_rate'].append((step, value))
        
        # 记录到训练日志
        self.log_message(f"{category}/{metric}: {value:.6f} (Step: {step})")
    
    def log_epoch_summary(self, epoch, train_loss, train_acc, val_loss, val_acc, lr, extra_info=None):
        """记录每个epoch的汇总信息"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 写入指标文件
        with open(self.metric_log_file, 'a', encoding='utf-8') as f:
            f.write(f"{epoch}\t{train_loss:.6f}\t{train_acc:.4f}\t{val_loss:.6f}\t{val_acc:.4f}\t{lr:.8f}\t{timestamp}\n")
        
        # 写入训练日志
        summary = f"""
{'='*60}
Epoch {epoch} 训练总结
{'='*60}
训练损失: {train_loss:.6f}
训练准确率: {train_acc:.4f}%
验证损失: {val_loss:.6f}  
验证准确率: {val_acc:.4f}%
学习率: {lr:.8f}
时间: {timestamp}
"""
        
        if extra_info:
            summary += f"额外信息: {extra_info}\n"
        
        summary += "=" * 60 + "\n"
        
        with open(self.train_log_file, 'a', encoding='utf-8') as f:
            f.write(summary)
    
    def log_training_start(self, model_info, data_info, config_info):
        """记录训练开始信息"""
        start_info = f"""
{'='*80}
训练开始
{'='*80}

🤖 模型信息:
{model_info}

📊 数据信息:
{data_info}

⚙️ 配置信息:
{config_info}

{'='*80}

"""
        with open(self.train_log_file, 'a', encoding='utf-8') as f:
            f.write(start_info)
    
    def log_training_end(self, best_epoch, best_acc, total_time):
        """记录训练结束信息"""
        end_info = f"""
{'='*80}
训练结束
{'='*80}
🏆 最佳验证准确率: {best_acc:.4f}% (Epoch {best_epoch})
⏱️ 总训练时间: {total_time:.2f} 分钟
📅 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
📁 日志目录: {self.experiment_dir}
{'='*80}
"""
        with open(self.train_log_file, 'a', encoding='utf-8') as f:
            f.write(end_info)
        
        # 生成训练总结报告
        self._generate_summary_report(best_epoch, best_acc, total_time)
    
    def _generate_summary_report(self, best_epoch, best_acc, total_time):
        """生成训练总结报告"""
        summary_file = self.experiment_dir / "training_summary.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("CPR-TaG-Net 训练总结报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"🎯 最佳结果:\n")
            f.write(f"   最佳验证准确率: {best_acc:.4f}%\n")
            f.write(f"   最佳Epoch: {best_epoch}\n")
            f.write(f"   总训练时间: {total_time:.2f} 分钟\n\n")
            
            f.write(f"📊 训练统计:\n")
            if self.metrics['train_loss']:
                final_train_loss = self.metrics['train_loss'][-1][1]
                f.write(f"   最终训练损失: {final_train_loss:.6f}\n")
            
            if self.metrics['train_accuracy']:
                final_train_acc = self.metrics['train_accuracy'][-1][1]
                f.write(f"   最终训练准确率: {final_train_acc:.4f}%\n")
                
            if self.metrics['val_loss']:
                final_val_loss = self.metrics['val_loss'][-1][1]
                f.write(f"   最终验证损失: {final_val_loss:.6f}\n")
                
            if self.metrics['val_accuracy']:
                final_val_acc = self.metrics['val_accuracy'][-1][1]
                f.write(f"   最终验证准确率: {final_val_acc:.4f}%\n")
            
            f.write(f"\n📁 文件列表:\n")
            f.write(f"   训练日志: {self.train_log_file.name}\n")
            f.write(f"   指标数据: {self.metric_log_file.name}\n")
            f.write(f"   配置文件: {self.config_file.name}\n")
            f.write(f"   总结报告: {summary_file.name}\n")
        
        print(f"📋 训练总结报告已生成: {summary_file}")
    
    def close(self):
        """关闭日志记录器"""
        self.log_message("日志记录器关闭", "INFO")
        print(f"📝 日志已保存到: {self.experiment_dir}")
    
    def get_experiment_dir(self):
        """获取实验目录路径"""
        return str(self.experiment_dir)

def create_txt_logger(log_dir, experiment_name="cpr_tagnet_training"):
    """创建TXT日志记录器的工厂函数"""
    return TxtLogger(log_dir, experiment_name)

# 测试代码
if __name__ == "__main__":
    # 测试日志记录器
    logger = create_txt_logger("test_logs", "test_experiment")
    
    # 测试配置记录
    test_config = {
        "epochs": 25,
        "learning_rate": 0.0005,
        "batch_size": 300
    }
    logger.log_config(test_config)
    
    # 测试训练开始
    logger.log_training_start(
        "CPR-TaG-Net (513K参数)",
        "18类血管，24个案例",
        "25 epochs, lr=0.0005"
    )
    
    # 测试指标记录
    for epoch in range(1, 4):
        train_loss = 2.5 - 0.1 * epoch
        train_acc = 40 + 5 * epoch
        val_loss = 2.8 - 0.08 * epoch
        val_acc = 35 + 6 * epoch
        lr = 0.0005
        
        logger.add_scalar('Train/EpochLoss', train_loss, epoch)
        logger.add_scalar('Train/EpochAccuracy', train_acc, epoch)
        logger.add_scalar('Val/EpochLoss', val_loss, epoch)
        logger.add_scalar('Val/EpochAccuracy', val_acc, epoch)
        
        logger.log_epoch_summary(epoch, train_loss, train_acc, val_loss, val_acc, lr)
    
    # 测试训练结束
    logger.log_training_end(best_epoch=3, best_acc=53.5, total_time=45.2)
    logger.close()
    
    print("✅ TXT日志记录器测试完成")
