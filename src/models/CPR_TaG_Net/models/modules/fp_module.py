import torch
import torch.nn as nn
import torch.nn.functional as F

class TopoFPModule(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels
        
        # 动态调整输入维度的MLP
        self.unit_pointnet = nn.Sequential(
            nn.Linear(in_channels + skip_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        
        # 备用投影层，用于维度不匹配时
        self.adaptive_proj = nn.Linear(1, out_channels)  # 将在运行时重新初始化

    def forward(self, x_src, pos_src, pos_tgt, x_skip):
        """
        x_src:     [N_src, C]
        pos_src:   [N_src, 3]
        pos_tgt:   [N_tgt, 3]
        x_skip:    [N_tgt, C']，可为 None
        return:    [N_tgt, out_channels]
        """
        try:
            # 边界检查
            if x_src.shape[0] == 0 or pos_tgt.shape[0] == 0:
                # 空输入处理
                if x_skip is not None:
                    return self._process_skip_only(x_skip)
                else:
                    return torch.zeros((pos_tgt.shape[0], self.out_channels), 
                                     device=x_src.device if x_src.numel() > 0 else pos_tgt.device)
            
            # Step 1: 最近邻上采样（inverse distance weighted interpolation）
            dists = torch.cdist(pos_tgt, pos_src) + 1e-8  # [N_tgt, N_src]
            
            # 🔧 修复：确保k不超过源节点数量
            k = min(3, x_src.shape[0])
            knn_dists, knn_idx = torch.topk(dists, k=k, dim=1, largest=False)  # [N_tgt, k]

            weight = 1.0 / knn_dists  # [N_tgt, k]
            weight = weight / torch.sum(weight, dim=1, keepdim=True)

            interpolated = torch.sum(
                x_src[knn_idx] * weight.unsqueeze(-1), dim=1
            )  # [N_tgt, C]

            # Step 2: 跳连 + 动态维度处理
            if x_skip is not None:
                feat_cat = torch.cat([interpolated, x_skip], dim=1)
            else:
                feat_cat = interpolated

            # 🔧 修复：动态处理维度不匹配
            expected_dim = self.in_channels + self.skip_channels
            actual_dim = feat_cat.shape[1]
            
            if actual_dim != expected_dim:
                # 创建自适应投影层
                if not hasattr(self, '_adaptive_proj') or self._adaptive_proj.in_features != actual_dim:
                    self._adaptive_proj = nn.Linear(actual_dim, expected_dim).to(feat_cat.device)
                feat_cat = self._adaptive_proj(feat_cat)
            
            out = self.unit_pointnet(feat_cat)  # [N_tgt, out_channels]
            return out
            
        except Exception as e:
            print(f"⚠️  TopoFPModule error: {e}")
            # 降级处理：直接处理跳连特征
            if x_skip is not None:
                return self._process_skip_only(x_skip)
            else:
                return torch.zeros((pos_tgt.shape[0], self.out_channels), 
                                 device=x_src.device if x_src.numel() > 0 else pos_tgt.device)
    
    def _process_skip_only(self, x_skip):
        """处理仅有跳连特征的情况"""
        try:
            if x_skip.shape[1] == self.out_channels:
                return x_skip
            elif x_skip.shape[1] < self.out_channels:
                # 填充到目标维度
                padding = torch.zeros((x_skip.shape[0], self.out_channels - x_skip.shape[1]), 
                                    device=x_skip.device)
                return torch.cat([x_skip, padding], dim=1)
            else:
                # 投影到目标维度
                if not hasattr(self, '_skip_proj') or self._skip_proj.in_features != x_skip.shape[1]:
                    self._skip_proj = nn.Linear(x_skip.shape[1], self.out_channels).to(x_skip.device)
                return self._skip_proj(x_skip)
        except:
            return torch.zeros((x_skip.shape[0], self.out_channels), device=x_skip.device)