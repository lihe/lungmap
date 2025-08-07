import torch
import torch.nn as nn
from models.modules.fp_module import TopoFPModule
from models.modules.sa_module import TopoSAModule
from models.modules.cnn3d import CNN3D

class CPRTaGNet(nn.Module):
    def __init__(self, num_classes=18, node_feature_dim=54, image_channels=1):
        super().__init__()
        self.num_classes = num_classes
        self.node_feature_dim = node_feature_dim
        
        # 图像条件提取路径
        self.image_encoder = nn.Sequential(
            CNN3D(in_channels=image_channels, hidden_channels=64),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten()
        )
        # 简化图像特征处理，去掉BiLSTM，直接使用CNN3D输出
        self.image_proj = nn.Linear(64, 256)

        # SA 层（Encoder）
        self.sa1 = TopoSAModule(in_channels=node_feature_dim, out_channels=128, k=8, sample_ratio=0.5)
        self.sa2 = TopoSAModule(in_channels=128, out_channels=256, k=8, sample_ratio=0.5)

        # FP 层（Decoder）
        self.fp1 = TopoFPModule(in_channels=256, skip_channels=128, out_channels=128)
        self.fp2 = TopoFPModule(in_channels=128, skip_channels=node_feature_dim, out_channels=64)

        # 分类器（用于分支标签预测）
        self.classifier = nn.Sequential(
            nn.Linear(64 + 256, 128),  # 64 from fp2_feat + 256 from image_cond
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_node, pos, edge_index, image_cubes):
        """
        x_node: 节点特征 [N, node_feature_dim]（如坐标 + 半径等）
        pos: 节点位置 [N, 3]
        edge_index: 图连接 [2, E]
        image_cubes: 对应节点图像块 [N, C, D, H, W]
        """
        # 输入验证
        self._validate_inputs(x_node, pos, edge_index, image_cubes)
        
        batch_size = x_node.shape[0]
        
        # ===== 图像条件路径 =====
        try:
            # 为4D图像数据添加通道维度 [N, 32, 32, 32] -> [N, 1, 32, 32, 32]
            if image_cubes.dim() == 4:
                image_cubes = image_cubes.unsqueeze(1)
            image_feat = self.image_encoder(image_cubes)  # [N, 64]
            image_cond = self.image_proj(image_feat)  # [N, 256]
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            raise RuntimeError(f"图像特征提取失败: {e}") from e

        # ===== 图结构路径：Encoder（SA） ===== 
        try:
            sa1_feat, sa1_pos, sa1_edge = self.sa1(x_node, pos, edge_index)  # [M1, 128]
            sa2_feat, sa2_pos, sa2_edge = self.sa2(sa1_feat, sa1_pos, sa1_edge)  # [M2, 256]
        except (RuntimeError, IndexError, ValueError) as e:
            if "tensor" in str(e).lower() and "dimension" in str(e).lower():
                raise ValueError(f"SA模块tensor维度错误: {e}") from e
            elif "index" in str(e).lower():
                raise IndexError(f"SA模块索引错误: {e}") from e
            else:
                raise RuntimeError(f"SA模块处理失败: {e}") from e

        # ===== 图结构路径：Decoder（FP） =====
        try:
            fp1_feat = self.fp1(sa2_feat, sa2_pos, sa1_pos, sa1_feat)  # [M1, 128]
            fp2_feat = self.fp2(fp1_feat, sa1_pos, pos, x_node)        # [N, 64]
        except (RuntimeError, ValueError) as e:
            if "shape" in str(e).lower() or "dimension" in str(e).lower():
                raise ValueError(f"FP模块维度不匹配: {e}") from e
            else:
                raise RuntimeError(f"FP模块处理失败: {e}") from e

        # ===== 特征维度对齐 =====
        fp2_feat, image_cond = self._align_feature_dimensions(fp2_feat, image_cond)

        # ===== 特征融合与分类 =====
        try:
            out_feat = torch.cat([fp2_feat, image_cond], dim=1)  # [N, 320]
            logits = self.classifier(out_feat)  # [N, num_classes]
            return logits
        except (RuntimeError, ValueError) as e:
            if "cat" in str(e).lower():
                raise ValueError(f"特征拼接失败，维度不匹配: fp2_feat {fp2_feat.shape}, image_cond {image_cond.shape}") from e
            else:
                raise RuntimeError(f"分类器处理失败: {e}") from e
    
    def _validate_inputs(self, x_node, pos, edge_index, image_cubes):
        """输入数据验证"""
        if x_node.dim() != 2 or x_node.shape[1] != self.node_feature_dim:
            raise ValueError(f"x_node维度错误: 期望 [N, {self.node_feature_dim}], 实际 {x_node.shape}")
        
        if pos.dim() != 2 or pos.shape[1] != 3:
            raise ValueError(f"pos维度错误: 期望 [N, 3], 实际 {pos.shape}")
        
        if x_node.shape[0] != pos.shape[0]:
            raise ValueError(f"节点数量不匹配: x_node {x_node.shape[0]}, pos {pos.shape[0]}")
        
        if edge_index.dim() != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"edge_index维度错误: 期望 [2, E], 实际 {edge_index.shape}")
        
        if image_cubes.dim() not in [4, 5]:
            raise ValueError(f"image_cubes维度错误: 期望 [N, D, H, W] 或 [N, C, D, H, W], 实际 {image_cubes.shape}")
        
        if image_cubes.shape[0] != x_node.shape[0]:
            raise ValueError(f"图像块数量不匹配: image_cubes {image_cubes.shape[0]}, x_node {x_node.shape[0]}")
    
    def _align_feature_dimensions(self, fp2_feat, image_cond):
        """特征维度对齐处理"""
        # 检查节点数量维度匹配
        if fp2_feat.shape[0] != image_cond.shape[0]:
            min_nodes = min(fp2_feat.shape[0], image_cond.shape[0])
            print(f"⚠️  节点数量不匹配，对齐到 {min_nodes}: fp2_feat {fp2_feat.shape[0]} -> {min_nodes}, image_cond {image_cond.shape[0]} -> {min_nodes}")
            fp2_feat = fp2_feat[:min_nodes]
            image_cond = image_cond[:min_nodes]
        
        # 检查特征维度
        expected_graph_dim = 64
        expected_image_dim = 256
        
        if fp2_feat.shape[1] != expected_graph_dim:
            raise ValueError(f"图特征维度错误: 期望 {expected_graph_dim}, 实际 {fp2_feat.shape[1]}")
        
        if image_cond.shape[1] != expected_image_dim:
            raise ValueError(f"图像特征维度错误: 期望 {expected_image_dim}, 实际 {image_cond.shape[1]}")
        
        return fp2_feat, image_cond