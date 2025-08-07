#!/usr/bin/env python3
"""
CPR-TaG-Net 增强训练工具 - 集成图形补全、可视化等高级功能
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
import os
from pathlib import Path

class EnhancedTrainingUtils:
    """集成图形补全、可视化等高级功能的训练工具"""
    
    def __init__(self, vessel_classes=None, save_dir="outputs/visualizations"):
        """
        初始化增强训练工具
        Args:
            vessel_classes: 血管类别字典，格式 {"类别名": 类别ID}
            save_dir: 可视化结果保存目录
        """
        self.vessel_classes = vessel_classes or self._get_default_vessel_classes()
        self.class_names = list(self.vessel_classes.keys())
        self.num_classes = len(self.class_names)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🧠 增强训练工具初始化完成")
        print(f"   血管类别数: {self.num_classes}")
        print(f"   可视化保存目录: {self.save_dir}")
        
    def _get_default_vessel_classes(self):
        """获取默认的15类血管分类"""
        return {
            'MPA': 0, 'LPA': 1, 'RPA': 2,
            'Lupper': 3, 'Rupper': 4,
            'L1+2': 5, 'R1+2': 6,
            'L1+3': 7, 'R1+3': 8,
            'Linternal': 9, 'Rinternal': 10,
            'Lmedium': 11, 'Rmedium': 12,
            'Ldown': 13, 'RDown': 14
        }
        
    def complete_graph(self, centerline_pos, predicted_labels, distance_threshold=5.0):
        """
        🧠 图形补全：基于血管连续性约束优化分类结果
        
        Args:
            centerline_pos: 中心线位置 [N, 3]
            predicted_labels: 预测标签 [N]
            distance_threshold: 距离阈值
            
        Returns:
            refined_labels: 优化后的标签 [N]
            edge_index: 新建立的边连接 [2, E]
        """
        if len(centerline_pos) == 0:
            return predicted_labels, torch.empty((2, 0), dtype=torch.long)
            
        print(f"🔧 执行图形补全: {len(centerline_pos)} 个节点")
        
        # 确保输入是tensor格式
        if not isinstance(centerline_pos, torch.Tensor):
            centerline_pos = torch.tensor(centerline_pos, dtype=torch.float32)
        if not isinstance(predicted_labels, torch.Tensor):
            predicted_labels = torch.tensor(predicted_labels, dtype=torch.long)
            
        # 计算节点间距离矩阵
        distances = torch.cdist(centerline_pos, centerline_pos)  # [N, N]
        
        # 找到相近的节点对
        close_pairs = torch.where((distances < distance_threshold) & (distances > 0))
        
        # 基于标签一致性和解剖学规则建立连接
        refined_edges = []
        refined_labels = predicted_labels.clone()
        connection_count = 0
        label_update_count = 0
        
        for i, j in zip(close_pairs[0], close_pairs[1]):
            if i < j:  # 避免重复边
                label_i, label_j = predicted_labels[i], predicted_labels[j]
                
                # 如果标签相同或相邻（基于血管解剖学），建立连接
                if self._should_connect(label_i, label_j):
                    refined_edges.append([i.item(), j.item()])
                    connection_count += 1
                    
                    # 标签平滑：基于邻域一致性更新标签
                    if self._should_update_label(i, j, distances, predicted_labels):
                        refined_labels[i] = predicted_labels[j]
                        label_update_count += 1
        
        print(f"   ✅ 建立连接: {connection_count} 条边")
        print(f"   ✅ 标签优化: {label_update_count} 个节点")
        
        if len(refined_edges) == 0:
            return refined_labels, torch.empty((2, 0), dtype=torch.long)
        
        edge_index = torch.tensor(refined_edges, dtype=torch.long).T  # [2, E]
        return refined_labels, edge_index
    
    def _should_connect(self, label_i, label_j):
        """判断两个标签的血管段是否应该连接"""
        # 血管解剖连接规则 - 基于15类肺血管树结构
        anatomical_connections = {
            0: [1, 2],      # MPA -> LPA, RPA
            1: [3, 5, 7, 9, 11, 13],    # LPA -> Lupper, L1+2, L1+3, Linternal, Lmedium, Ldown
            2: [4, 6, 8, 10, 12, 14],   # RPA -> Rupper, R1+2, R1+3, Rinternal, Rmedium, RDown
        }
        
        # 相同标签可以连接
        if label_i == label_j:
            return True
            
        # 检查解剖连接关系（双向）
        i_val, j_val = label_i.item(), label_j.item()
        return (j_val in anatomical_connections.get(i_val, [])) or \
               (i_val in anatomical_connections.get(j_val, []))
    
    def _should_update_label(self, i, j, distances, labels, threshold=3.0):
        """基于邻域标签一致性判断是否更新标签"""
        # 找到节点i的所有近邻
        neighbors = torch.where(distances[i] < threshold)[0]
        
        if len(neighbors) < 3:  # 邻居太少，不进行标签更新
            return False
            
        neighbor_labels = labels[neighbors]
        
        # 统计邻域标签分布
        label_counts = torch.bincount(neighbor_labels, minlength=self.num_classes)
        most_common_label = torch.argmax(label_counts)
        
        # 如果当前标签不是邻域主要标签且主要标签有足够支持，考虑更新
        return (labels[i] != most_common_label and 
                label_counts[most_common_label] >= 3 and
                label_counts[most_common_label] > label_counts[labels[i]])
    
    def plot_confusion_matrix(self, y_true, y_pred, epoch=None, show_percentages=True):
        """
        📊 绘制增强版混淆矩阵
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            epoch: 当前epoch（用于文件名）
            show_percentages: 是否显示百分比
        """
        print(f"📊 生成混淆矩阵 (样本数: {len(y_true)})")
        
        # 转换为numpy数组
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
        
        # 准备注释数据
        if show_percentages:
            cm_percent = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8) * 100
            annot_data = np.array([[f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)' 
                                  if cm[i,j] > 0 else ''
                                  for j in range(cm.shape[1])] 
                                 for i in range(cm.shape[0])])
        else:
            annot_data = cm
        
        # 创建图形
        plt.figure(figsize=(16, 14))
        
        # 使用更好的颜色映射
        mask = cm == 0
        sns.heatmap(cm, 
                   annot=annot_data, 
                   fmt='' if show_percentages else 'd',
                   cmap="Blues", 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Sample Count'},
                   mask=mask,
                   linewidths=0.5,
                   square=True)
        
        plt.xlabel("Predicted Label", fontsize=14, fontweight='bold')
        plt.ylabel("True Label", fontsize=14, fontweight='bold')
        
        title = "CPR-TaG-Net 血管分类混淆矩阵"
        if epoch is not None:
            title += f" (Epoch {epoch})"
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # 保存图像
        save_name = f"confusion_matrix_epoch_{epoch}.png" if epoch else "confusion_matrix.png"
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   💾 已保存: {save_path}")
        
        plt.show()
        
        # 打印分类报告
        print("\n📋 详细分类报告:")
        try:
            report = classification_report(y_true, y_pred, 
                                         target_names=self.class_names,
                                         labels=range(self.num_classes),
                                         zero_division=0)
            print(report)
            
            # 保存分类报告
            report_path = self.save_dir / f"classification_report_epoch_{epoch}.txt" if epoch else self.save_dir / "classification_report.txt"
            with open(report_path, 'w') as f:
                f.write(report)
            print(f"   💾 分类报告已保存: {report_path}")
            
        except Exception as e:
            print(f"   ⚠️ 分类报告生成失败: {e}")
    
    def analyze_prediction_quality(self, y_true, y_pred, positions=None):
        """
        🔍 分析预测质量和错误模式
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            positions: 节点位置（可选）
            
        Returns:
            analysis: 分析结果字典
        """
        print(f"🔍 分析预测质量")
        
        # 转换为numpy数组
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        
        analysis = {
            'total_samples': len(y_true),
            'overall_accuracy': float((y_true == y_pred).sum() / len(y_true)),
            'class_accuracy': {},
            'class_support': {},
            'error_patterns': defaultdict(int),
            'difficult_classes': [],
            'well_classified_classes': []
        }
        
        # 按类别分析准确率
        for class_idx, class_name in enumerate(self.class_names):
            mask = (y_true == class_idx)
            support = mask.sum()
            analysis['class_support'][class_name] = int(support)
            
            if support > 0:
                class_acc = float((y_pred[mask] == class_idx).sum() / support)
                analysis['class_accuracy'][class_name] = class_acc
                
                if class_acc < 0.5:  # 准确率低于50%的困难类别
                    analysis['difficult_classes'].append((class_name, class_acc))
                elif class_acc > 0.8:  # 准确率高于80%的良好类别
                    analysis['well_classified_classes'].append((class_name, class_acc))
        
        # 分析错误模式
        errors = y_true != y_pred
        if errors.sum() > 0:
            for true_label, pred_label in zip(y_true[errors], y_pred[errors]):
                if 0 <= true_label < len(self.class_names) and 0 <= pred_label < len(self.class_names):
                    error_pattern = f"{self.class_names[true_label]} -> {self.class_names[pred_label]}"
                    analysis['error_patterns'][error_pattern] += 1
        
        # 排序困难类别和错误模式
        analysis['difficult_classes'] = sorted(analysis['difficult_classes'], key=lambda x: x[1])
        analysis['error_patterns'] = dict(sorted(analysis['error_patterns'].items(), 
                                                key=lambda x: x[1], reverse=True)[:10])  # 前10个错误模式
        
        # 打印分析结果
        print(f"   总体准确率: {analysis['overall_accuracy']:.4f}")
        print(f"   困难类别数: {len(analysis['difficult_classes'])}")
        print(f"   主要错误模式数: {len(analysis['error_patterns'])}")
        
        if analysis['difficult_classes']:
            print("   🔴 困难类别:")
            for class_name, acc in analysis['difficult_classes'][:5]:
                print(f"      {class_name}: {acc:.3f}")
        
        if analysis['error_patterns']:
            print("   ⚠️ 主要错误模式:")
            for pattern, count in list(analysis['error_patterns'].items())[:3]:
                print(f"      {pattern}: {count} 次")
        
        return analysis
    
    def visualize_training_progress(self, train_losses, train_accs, val_losses=None, val_accs=None, epoch=None):
        """
        📈 可视化训练进度
        
        Args:
            train_losses: 训练损失列表
            train_accs: 训练准确率列表
            val_losses: 验证损失列表（可选）
            val_accs: 验证准确率列表（可选）
            epoch: 当前epoch（用于文件名）
        """
        print(f"📈 生成训练进度图表")
        
        epochs = range(1, len(train_losses) + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('CPR-TaG-Net 训练进度监控', fontsize=16, fontweight='bold')
        
        # 训练损失
        axes[0, 0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, marker='o', markersize=3)
        if val_losses:
            val_epochs = range(1, len(val_losses) + 1)
            axes[0, 0].plot(val_epochs, val_losses, 'r--', label='Validation Loss', linewidth=2, marker='s', markersize=3)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('训练损失曲线')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 训练准确率
        axes[0, 1].plot(epochs, train_accs, 'g-', label='Training Accuracy', linewidth=2, marker='o', markersize=3)
        if val_accs:
            val_epochs = range(1, len(val_accs) + 1)
            axes[0, 1].plot(val_epochs, val_accs, 'r--', label='Validation Accuracy', linewidth=2, marker='s', markersize=3)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('训练准确率曲线')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 1)
        
        # 损失变化率
        if len(train_losses) > 1:
            loss_changes = np.diff(train_losses)
            axes[1, 0].plot(epochs[1:], loss_changes, 'purple', linewidth=2, marker='o', markersize=3)
            axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss Change')
            axes[1, 0].set_title('损失变化率')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 准确率变化率
        if len(train_accs) > 1:
            acc_changes = np.diff(train_accs)
            axes[1, 1].plot(epochs[1:], acc_changes, 'orange', linewidth=2, marker='o', markersize=3)
            axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy Change')
            axes[1, 1].set_title('准确率变化率')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像
        save_name = f"training_progress_epoch_{epoch}.png" if epoch else "training_progress.png"
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   💾 已保存: {save_path}")
        
        plt.show()
    
    def save_analysis_report(self, analysis, epoch=None):
        """保存分析报告到文件"""
        report_name = f"analysis_report_epoch_{epoch}.txt" if epoch else "analysis_report.txt"
        report_path = self.save_dir / report_name
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("CPR-TaG-Net 预测质量分析报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"总样本数: {analysis['total_samples']}\n")
            f.write(f"总体准确率: {analysis['overall_accuracy']:.4f}\n\n")
            
            f.write("各类别准确率:\n")
            for class_name, acc in analysis['class_accuracy'].items():
                support = analysis['class_support'][class_name]
                f.write(f"  {class_name}: {acc:.4f} (支持度: {support})\n")
            
            f.write(f"\n困难类别 (准确率 < 0.5):\n")
            for class_name, acc in analysis['difficult_classes']:
                f.write(f"  {class_name}: {acc:.4f}\n")
            
            f.write(f"\n主要错误模式:\n")
            for pattern, count in analysis['error_patterns'].items():
                f.write(f"  {pattern}: {count} 次\n")
        
        print(f"   💾 分析报告已保存: {report_path}")

def create_enhanced_trainer(save_dir="outputs/visualizations"):
    """创建增强训练工具实例"""
    return EnhancedTrainingUtils(save_dir=save_dir)

# 示例用法
if __name__ == "__main__":
    # 测试增强训练工具
    trainer = create_enhanced_trainer()
    print("✅ 增强训练工具测试完成")
