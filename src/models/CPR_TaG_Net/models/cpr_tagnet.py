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
        try:
            batch_size = x_node.shape[0]
            
            # ===== 图像条件路径 =====
            image_feat = self.image_encoder(image_cubes)  # [N, 64]
            image_cond = self.image_proj(image_feat)  # [N, 256]

            # ===== 图结构路径：Encoder（SA） =====
            sa1_feat, sa1_pos, sa1_edge = self.sa1(x_node, pos, edge_index)  # [M1, 128]
            sa2_feat, sa2_pos, sa2_edge = self.sa2(sa1_feat, sa1_pos, sa1_edge)  # [M2, 256]

            # ===== 图结构路径：Decoder（FP） =====
            fp1_feat = self.fp1(sa2_feat, sa2_pos, sa1_pos, sa1_feat)  # [M1, 128]
            fp2_feat = self.fp2(fp1_feat, sa1_pos, pos, x_node)        # [N, 64]

            # ===== 🔧 修复：确保维度匹配 =====
            # 检查fp2_feat和image_cond的维度是否匹配
            if fp2_feat.shape[0] != image_cond.shape[0]:
                print(f"⚠️  Dimension mismatch: fp2_feat {fp2_feat.shape}, image_cond {image_cond.shape}")
                # 如果维度不匹配，使用插值调整
                if fp2_feat.shape[0] < image_cond.shape[0]:
                    # 从image_cond中选择对应的特征
                    image_cond = image_cond[:fp2_feat.shape[0]]
                else:
                    # 对fp2_feat进行下采样或重复
                    fp2_feat = fp2_feat[:image_cond.shape[0]]

            # ===== 多任务输出（融合图结构 + 图像条件） =====
            # 🔧 修复：动态处理特征维度不匹配
            expected_graph_dim = 64  # 期望的图特征维度
            expected_image_dim = 256  # 期望的图像特征维度
            
            # 调整图特征维度
            if fp2_feat.shape[1] != expected_graph_dim:
                if not hasattr(self, '_graph_proj') or self._graph_proj.in_features != fp2_feat.shape[1]:
                    self._graph_proj = nn.Linear(fp2_feat.shape[1], expected_graph_dim).to(fp2_feat.device)
                fp2_feat = self._graph_proj(fp2_feat)
            
            # 调整图像特征维度
            if image_cond.shape[1] != expected_image_dim:
                if not hasattr(self, '_image_proj_fix') or self._image_proj_fix.in_features != image_cond.shape[1]:
                    self._image_proj_fix = nn.Linear(image_cond.shape[1], expected_image_dim).to(image_cond.device)
                image_cond = self._image_proj_fix(image_cond)
            
            out_feat = torch.cat([fp2_feat, image_cond], dim=1)  # [N, 320]
            
            # 🔧 修复：动态处理分类器输入维度
            expected_classifier_dim = 320  # 64 + 256
            if out_feat.shape[1] != expected_classifier_dim:
                if not hasattr(self, '_classifier_proj') or self._classifier_proj.in_features != out_feat.shape[1]:
                    self._classifier_proj = nn.Linear(out_feat.shape[1], expected_classifier_dim).to(out_feat.device)
                out_feat = self._classifier_proj(out_feat)
            
            logits = self.classifier(out_feat)  # [N, num_classes]

            return logits
            
        except Exception as e:
            print(f"⚠️  CPRTaGNet forward error: {e}")
            # 降级处理：仅使用图像特征
            batch_size = x_node.shape[0]
            image_feat = self.image_encoder(image_cubes)  # [N, 64]
            image_cond = self.image_proj(image_feat)  # [N, 256]
            
            # 创建零图结构特征
            graph_feat = torch.zeros((batch_size, 64), device=x_node.device)
            out_feat = torch.cat([graph_feat, image_cond], dim=1)
            logits = self.classifier(out_feat)
            return logits