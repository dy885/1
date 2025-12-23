# sub_model/transformer.py
import torch
import torch.nn as nn


class SpatioTemporalTransformer(nn.Module):
    """
    Spatio-Temporal Transformer

    支持两种输入模式：
    1. [B, T, C] - 已池化的时序特征（来自conflict_model）
    2. [B, T, C, H, W] - 完整的时空特征
    """

    def __init__(
            self,
            in_channels,
            d_model=256,
            nhead=8,
            num_layers=4,
            dropout=0.1
    ):
        super().__init__()

        self.proj = nn.Linear(in_channels, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.d_model = d_model

    def forward(self, feat):
        """
        前向传播 - 支持两种输入格式

        模式1（来自conflict_model）:
            输入: [B, T, C]
            输出: [B, T, d_model]

        模式2（完整时空）:
            输入: [B, T, C, H, W]
            输出: [B, T, d_model]
        """

        if feat.dim() == 3:
            # 模式1：已池化的特征 [B, T, C]
            B, T, C = feat.shape

            # 投影到 d_model
            tokens = self.proj(feat)  # [B, T, d_model]

            # Transformer编码
            tokens = self.transformer(tokens)  # [B, T, d_model]

            return tokens

        elif feat.dim() == 5:
            # 模式2：完整时空特征 [B, T, C, H, W]
            B, T, C, H, W = feat.shape

            # 空间展平
            feat = feat.view(B, T, C, H * W)  # [B, T, C, HW]
            feat = feat.permute(0, 1, 3, 2)  # [B, T, HW, C]
            feat = feat.reshape(B, T * H * W, C)  # [B, T*HW, C]

            # 投影
            tokens = self.proj(feat)  # [B, T*HW, d_model]

            # Transformer编码
            tokens = self.transformer(tokens)  # [B, T*HW, d_model]

            # 时序池化
            tokens = tokens.view(B, T, H * W, self.d_model)
            tokens = tokens.mean(dim=2)  # [B, T, d_model] - 空间平均

            return tokens

        else:
            raise ValueError(f"Expected 3D or 5D input, got {feat.dim()}D tensor")


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("测试 SpatioTemporalTransformer")
    print("=" * 60)

    transformer = SpatioTemporalTransformer(
        in_channels=512,
        d_model=128,
        num_layers=4
    )

    # 测试模式1：3D输入（来自conflict_model）
    print("\n[测试1] 3D输入 [B, T, C]:")
    x_3d = torch.randn(2, 9, 512)
    print(f"  输入形状: {x_3d.shape}")
    output = transformer(x_3d)
    print(f"  输出形状: {output.shape}")
    assert output.shape == (2, 9, 128), "3D输出维度错误"
    print("  ✅ 3D输入测试通过")

    # 测试模式2：5D输入（完整时空）
    print("\n[测试2] 5D输入 [B, T, C, H, W]:")
    x_5d = torch.randn(2, 9, 512, 20, 20)
    print(f"  输入形状: {x_5d.shape}")
    output = transformer(x_5d)
    print(f"  输出形状: {output.shape}")
    assert output.shape == (2, 9, 128), "5D输出维度错误"
    print("  ✅ 5D输入测试通过")

    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)