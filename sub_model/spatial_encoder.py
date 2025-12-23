import torch
import torch.nn as nn
from torchvision import models


class UAVSpatialEncoder(nn.Module):
    """
    空间编码器：从图像提取空间特征
    支持不同的 ResNet backbone，自动处理通道转换
    """

    def __init__(self, backbone="resnet50", pretrained=True, output_channels=512):
        """
        参数:
            backbone: ResNet backbone 类型
            pretrained: 是否使用预训练权重
            output_channels: 期望的输出通道数（用于后续模块）
        """
        super().__init__()

        self.backbone_name = backbone

        # ResNet 各版本的输出通道数
        resnet_output_channels = {
            "resnet18": 512,
            "resnet34": 512,
            "resnet50": 2048,
            "resnet101": 2048,
            "resnet152": 2048,
        }

        # 加载 ResNet
        if backbone == "resnet18":
            resnet = models.resnet18(pretrained=pretrained)
        elif backbone == "resnet34":
            resnet = models.resnet34(pretrained=pretrained)
        elif backbone == "resnet50":
            resnet = models.resnet50(pretrained=pretrained)
        elif backbone == "resnet101":
            resnet = models.resnet101(pretrained=pretrained)
        elif backbone == "resnet152":
            resnet = models.resnet152(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # 提取特征层（去掉全局池化和全连接层）
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])

        # 获取 backbone 的实际输出通道数
        backbone_out_channels = resnet_output_channels[backbone]

        # ✅ 关键修复：添加通道适配器
        # 如果 backbone 输出通道数 != 期望输出通道数，添加转换层
        if backbone_out_channels != output_channels:
            self.channel_adapter = nn.Sequential(
                nn.Conv2d(backbone_out_channels, output_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )
            print(f"[INFO] UAVSpatialEncoder: Added channel adapter {backbone_out_channels} -> {output_channels}")
        else:
            self.channel_adapter = nn.Identity()
            print(f"[INFO] UAVSpatialEncoder: No channel adapter needed (already {output_channels} channels)")

        # 设置输出维度属性
        self.output_dim = output_channels

        print(f"[INFO] UAVSpatialEncoder initialized:")
        print(f"  - Backbone: {backbone}")
        print(f"  - Pretrained: {pretrained}")
        print(f"  - Output channels: {self.output_dim}")

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入张量
               - 如果是 4D: [B, C, H, W] - 单帧
               - 如果是 5D: [B, T, C, H, W] - 多帧

        返回:
            features: 提取的特征
               - 如果输入是 4D: [B, output_dim, h, w]
               - 如果输入是 5D: [B, T, output_dim, h, w]
        """
        # 处理多帧输入
        if x.dim() == 5:  # [B, T, C, H, W]
            B, T, C, H, W = x.shape

            # 展平时间维度以批量处理
            x = x.view(B * T, C, H, W)  # [B*T, C, H, W]

            # 提取特征
            features = self.feature_extractor(x)  # [B*T, backbone_channels, h, w]

            # 通道转换
            features = self.channel_adapter(features)  # [B*T, output_dim, h, w]

            # 恢复时间维度
            _, C_out, h, w = features.shape
            features = features.view(B, T, C_out, h, w)  # [B, T, output_dim, h, w]

            return features

        elif x.dim() == 4:  # [B, C, H, W] - 单帧
            # 提取特征
            features = self.feature_extractor(x)  # [B, backbone_channels, h, w]

            # 通道转换
            features = self.channel_adapter(features)  # [B, output_dim, h, w]

            return features

        else:
            raise ValueError(f"Expected 4D or 5D input, got {x.dim()}D")


# ============================================================
# 轻量级版本
# ============================================================
class UAVSpatialEncoderLite(UAVSpatialEncoder):
    """
    轻量级空间编码器，使用 ResNet-18
    """

    def __init__(self, pretrained=True, output_channels=512):
        super().__init__(
            backbone="resnet18",
            pretrained=pretrained,
            output_channels=output_channels
        )


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("测试 UAVSpatialEncoder")
    print("=" * 60)

    # 测试不同的 backbone
    for backbone in ["resnet18", "resnet50"]:
        print(f"\n{'=' * 60}")
        print(f"测试 {backbone}")
        print("=" * 60)

        encoder = UAVSpatialEncoder(
            backbone=backbone,
            pretrained=False,  # 测试时不加载预训练
            output_channels=512
        )

        print(f"\n输出维度: {encoder.output_dim}")

        # 测试单帧输入
        print("\n[测试1] 单帧输入:")
        x_single = torch.randn(2, 3, 640, 640)  # [B, C, H, W]
        print(f"  输入形状: {x_single.shape}")

        try:
            output = encoder(x_single)
            print(f"  ✅ 输出形状: {output.shape}")
            assert output.shape[1] == 512, f"Expected 512 channels, got {output.shape[1]}"
        except Exception as e:
            print(f"  ❌ 错误: {e}")

        # 测试多帧输入
        print("\n[测试2] 多帧输入:")
        x_multi = torch.randn(2, 9, 3, 640, 640)  # [B, T, C, H, W]
        print(f"  输入形状: {x_multi.shape}")

        try:
            output = encoder(x_multi)
            print(f"  ✅ 输出形状: {output.shape}")
            assert output.shape[2] == 512, f"Expected 512 channels, got {output.shape[2]}"
        except Exception as e:
            print(f"  ❌ 错误: {e}")

        # 统计参数量
        total_params = sum(p.numel() for p in encoder.parameters())
        print(f"\n  总参数量: {total_params / 1e6:.2f}M")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)