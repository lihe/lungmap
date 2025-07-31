#!/usr/bin/env python3
"""
CPR-TaG-Net 统一配置管理模块
"""

import os
import json
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

class ConfigManager:
    """统一配置管理器"""
    
    def __init__(self, config_dir: str = "src/models/CPR_TaG_Net/configs"):
        self.config_dir = Path(config_dir)
        self.train_config_path = self.config_dir / "train.yaml"
        self.label_rules_path = self.config_dir / "label_rules.json"
        
        # 加载配置
        self.train_config = self._load_train_config()
        self.label_rules = self._load_label_rules()
    
    def _load_train_config(self) -> Dict[str, Any]:
        """加载训练配置"""
        if not self.train_config_path.exists():
            raise FileNotFoundError(f"训练配置文件不存在: {self.train_config_path}")
        
        with open(self.train_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 验证必要的配置项
        self._validate_train_config(config)
        return config
    
    def _load_label_rules(self) -> Dict[str, Any]:
        """加载标签规则"""
        if not self.label_rules_path.exists():
            raise FileNotFoundError(f"标签规则文件不存在: {self.label_rules_path}")
        
        with open(self.label_rules_path, 'r', encoding='utf-8') as f:
            rules = json.load(f)
        
        return rules
    
    def _validate_train_config(self, config: Dict[str, Any]):
        """验证训练配置"""
        required_sections = ['model', 'training', 'dataset', 'log']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"配置文件缺少必要的section: {section}")
    
    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return self.train_config['model']
    
    def get_training_config(self) -> Dict[str, Any]:
        """获取训练配置"""
        return self.train_config['training']
    
    def get_dataset_config(self) -> Dict[str, Any]:
        """获取数据集配置"""
        return self.train_config['dataset']
    
    def get_log_config(self) -> Dict[str, Any]:
        """获取日志配置"""
        return self.train_config['log']
    
    def get_gpu_config(self) -> Dict[str, Any]:
        """获取GPU配置"""
        return self.train_config.get('gpu', {'device_id': 0, 'memory_limit': 24})
    
    def get_vessel_classes(self) -> list:
        """获取血管类别列表"""
        return self.train_config.get('vessel_classes', [])
    
    def get_vessel_connections(self) -> list:
        """获取血管连接关系"""
        return self.label_rules.get('pulmonary_vessels', {}).get('anatomical_connections', [])
    
    def get_class_mapping(self) -> Dict[str, int]:
        """获取类别映射"""
        return self.label_rules.get('pulmonary_vessels', {}).get('class_mapping', {})
    
    def get_vessel_hierarchy(self) -> Dict[str, int]:
        """获取血管层级"""
        return self.label_rules.get('pulmonary_vessels', {}).get('vessel_hierarchy', {})
    
    def get_command_line_args(self) -> list:
        """生成命令行参数列表（用于subprocess调用）"""
        training_cfg = self.get_training_config()
        gpu_cfg = self.get_gpu_config()
        
        args = [
            "--epochs", str(training_cfg['epochs']),
            "--learning_rate", str(training_cfg['lr']),
            "--weight_decay", str(training_cfg['weight_decay']),
            "--step_size", str(training_cfg['step_size']),
            "--gamma", str(training_cfg['gamma']),
            "--node_batch_size", str(training_cfg['node_batch_size']),
            "--max_nodes_per_case", str(training_cfg['max_nodes_per_case']),
        ]
        
        if training_cfg.get('enable_large_cases', False):
            args.append("--enable_large_cases")
        
        return args
    
    def get_env_vars(self) -> Dict[str, str]:
        """获取环境变量"""
        gpu_cfg = self.get_gpu_config()
        env_vars = {}
        
        if 'device_id' in gpu_cfg:
            env_vars['CUDA_VISIBLE_DEVICES'] = str(gpu_cfg['device_id'])
        
        return env_vars
    
    def print_config_summary(self):
        """打印配置摘要"""
        print("🔧 CPR-TaG-Net 配置摘要")
        print("=" * 50)
        
        # 模型配置
        model_cfg = self.get_model_config()
        print(f"📊 模型配置:")
        print(f"  - 类别数: {model_cfg['num_classes']}")
        print(f"  - 节点特征维度: {model_cfg['node_feature_dim']}")
        print(f"  - 图像通道数: {model_cfg['image_condition_channels']}")
        
        # 训练配置
        training_cfg = self.get_training_config()
        print(f"🚀 训练配置:")
        print(f"  - 训练轮数: {training_cfg['epochs']}")
        print(f"  - 学习率: {training_cfg['lr']}")
        print(f"  - 节点批大小: {training_cfg['node_batch_size']}")
        print(f"  - 最大节点数: {training_cfg['max_nodes_per_case']}")
        print(f"  - 启用大案例: {training_cfg.get('enable_large_cases', False)}")
        
        # GPU配置
        gpu_cfg = self.get_gpu_config()
        print(f"💻 GPU配置:")
        print(f"  - 设备ID: {gpu_cfg['device_id']}")
        print(f"  - 显存限制: {gpu_cfg['memory_limit']}GB")
        
        # 数据集配置
        dataset_cfg = self.get_dataset_config()
        print(f"📁 数据集配置:")
        print(f"  - 数据目录: {dataset_cfg['data_dir']}")
        print(f"  - 文件模式: {dataset_cfg['file_pattern']}")
        
        # 血管类别
        vessel_classes = self.get_vessel_classes()
        print(f"🩺 血管类别: 共{len(vessel_classes)}类")
        
        print("=" * 50)

# 全局配置管理器实例
_config_manager = None

def get_config_manager() -> ConfigManager:
    """获取全局配置管理器实例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def load_config() -> Dict[str, Any]:
    """兼容性函数：加载完整配置"""
    return get_config_manager().train_config

# 便捷函数
def get_model_config() -> Dict[str, Any]:
    return get_config_manager().get_model_config()

def get_training_config() -> Dict[str, Any]:
    return get_config_manager().get_training_config()

def get_dataset_config() -> Dict[str, Any]:
    return get_config_manager().get_dataset_config()

def get_vessel_classes() -> list:
    return get_config_manager().get_vessel_classes()

def get_vessel_connections() -> list:
    return get_config_manager().get_vessel_connections()

if __name__ == "__main__":
    # 测试配置管理器
    try:
        config_mgr = ConfigManager()
        config_mgr.print_config_summary()
        
        print("\\n🧪 测试命令行参数生成:")
        args = config_mgr.get_command_line_args()
        print(" ".join(args))
        
        print("\\n🧪 测试环境变量:")
        env_vars = config_mgr.get_env_vars()
        for key, value in env_vars.items():
            print(f"{key}={value}")
            
    except Exception as e:
        print(f"❌ 配置管理器测试失败: {e}")
