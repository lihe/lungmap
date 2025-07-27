
import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageConditionEncoder(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=128, lstm_hidden=256):
        super(ImageConditionEncoder, self).__init__()

        # 3D CNN 编码器（提取局部图像块特征）
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(64, hidden_dim, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )

        # BiLSTM 进一步建模邻域图像块上下文（可选）
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x):
        # 输入: x [B, C, D, H, W] 图像块或 [B, N, C, D, H, W] 序列图像块
        # 输出: [B, F] 或 [B, N, F] 表征特征
        if x.dim() == 5:
            feat = self.encoder(x)  # [B, hidden_dim, 1,1,1]
            feat = feat.view(feat.size(0), -1)
            return feat
        elif x.dim() == 6:
            B, N = x.shape[:2]
            x = x.view(B * N, *x.shape[2:])  # [B*N, C, D, H, W]
            feat = self.encoder(x).view(B, N, -1)  # [B, N, F]
            feat, _ = self.bilstm(feat)  # [B, N, lstm_hidden]
            return feat
        else:
            raise ValueError("Invalid input shape for ImageConditionEncoder")
