import torch
import torch.nn as nn

from sub_model.spatial_encoder import UAVSpatialEncoder
from sub_model.transformer import SpatioTemporalTransformer
from sub_model.predictor import ConvOccupancyPredictor, MultiTaskPredictor, MotionPredictor


class FlexiblePredictor(nn.Module):
    """
    支持按需初始化的预测器容器
    mode: 'motion', 'occupancy', or 'multitask'
    """

    def __init__(self, in_dim, modes, future_steps, out_size, mode='multitask'):
        super().__init__()
        self.mode = mode

        # 共享特征层
        self.shared_proj = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.ReLU()
        )

        # 按需初始化 Head
        self.occ_head = None
        self.motion_head = None

        if mode in ['occupancy', 'multitask']:
            self.occ_head = ConvOccupancyPredictor(in_dim, modes, future_steps, out_size)

        if mode in ['motion', 'multitask']:
            self.motion_head = MotionPredictor(in_dim, modes, future_steps, out_dim=2)

    def forward(self, x, return_confidence=False):
        feat = self.shared_proj(x)

        if self.mode == 'occupancy':
            return self.occ_head(feat, return_confidence)

        elif self.mode == 'motion':
            return self.motion_head(feat, return_confidence)

        elif self.mode == 'multitask':
            occ_out = self.occ_head(feat, return_confidence)
            motion_out = self.motion_head(feat, return_confidence)
            return {"occ": occ_out, "motion": motion_out}


class UAVConflictModel(nn.Module):
    """
    UAV 冲突预测模型
    修复: 确保输出尺寸为640x640
    """

    def __init__(self,
                 mode="multitask",
                 hidden_dim=128,
                 modes=5,
                 encoder_backbone="resnet50",
                 future_steps=1,  # 修改为1，只预测单帧
                 out_size=(640, 640)):  # 明确指定输出尺寸
        super().__init__()

        self.mode = mode
        self.out_size = out_size

        # 1. 编码器 (共享)
        self.encoder = UAVSpatialEncoder(
            backbone=encoder_backbone,
            pretrained=True,
            output_channels=2048
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # 2. 时序模型 (共享)
        self.transformer = SpatioTemporalTransformer(
            in_channels=self.encoder.output_dim,
            d_model=hidden_dim,
            num_layers=4
        )

        # 3. 预测器 (根据 mode 动态构建)
        self.predictor = FlexiblePredictor(
            in_dim=hidden_dim,
            modes=modes,
            future_steps=future_steps,
            out_size=out_size,
            mode=mode
        )

        print(f"[Init] Model Mode: {mode.upper()}")
        print(f"[Init] Output Size: {out_size}")
        print(f"[Init] Future Steps: {future_steps}")

        if mode == 'multitask':
            print("  - Initialized both Motion and Occupancy heads")
        elif mode == 'motion':
            print("  - Initialized Motion head ONLY")
        elif mode == 'occupancy':
            print("  - Initialized Occupancy head ONLY")

    def forward(self, history, return_confidence=True):
        """
        Args:
            history: [B, T, 3, H, W]
        Returns:
            根据mode返回不同格式:
            - occupancy: (pred, conf) where pred: [B, M, T_f, 640, 640]
            - motion: (pred, conf) where pred: [B, M, T_f, 2]
            - multitask: {"occ": (pred, conf), "motion": (pred, conf)}
        """
        B, T, C, H, W = history.shape

        # Backbone 提取
        flat_hist = history.view(B * T, C, H, W)
        feat = self.encoder(flat_hist)
        feat = self.pool(feat).view(B, T, -1)

        # Transformer
        feat = self.transformer(feat)
        context = feat[:, -1]  # 取最后一帧特征

        # Predictor
        return self.predictor(context, return_confidence=return_confidence)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Testing UAVConflictModel with 640x640 output")
    print("=" * 60)

    # 模拟数据
    B, T = 2, 8
    dummy_input = torch.randn(B, T, 3, 256, 256)

    print("\n>>> Test 1: Occupancy Only Mode")
    print("=" * 60)
    model_occ = UAVConflictModel(
        mode="occupancy",
        hidden_dim=128,
        future_steps=1,
        out_size=(640, 640)
    )
    out_o, conf_o = model_occ(dummy_input)
    print(f"Output Shape: {out_o.shape}")  # Expected: [2, 5, 1, 640, 640]
    assert out_o.shape[-2:] == (640, 640), f"Wrong output size: {out_o.shape}"
    print("✓ Occupancy mode passed")

    print("\n>>> Test 2: Motion Only Mode")
    print("=" * 60)
    model_motion = UAVConflictModel(mode="motion", hidden_dim=128, future_steps=10)
    out_m, conf_m = model_motion(dummy_input)
    print(f"Output Shape: {out_m.shape}")  # Expected: [2, 5, 10, 2]
    print("✓ Motion mode passed")

    print("\n>>> Test 3: Multitask Mode")
    print("=" * 60)
    model_multi = UAVConflictModel(
        mode="multitask",
        hidden_dim=128,
        future_steps=1,
        out_size=(640, 640)
    )
    outputs = model_multi(dummy_input)

    pred_occ, conf_occ = outputs["occ"]
    pred_mot, conf_mot = outputs["motion"]

    print(f"Occ Shape: {pred_occ.shape}")  # Expected: [2, 5, 1, 640, 640]
    print(f"Mot Shape: {pred_mot.shape}")  # Expected: [2, 5, 1, 2]

    assert pred_occ.shape[-2:] == (640, 640), f"Wrong occ size: {pred_occ.shape}"
    print("✓ Multitask mode passed")


    # 参数量对比
    def count_params(model):
        return sum(p.numel() for p in model.parameters())


    p_m = count_params(model_motion)
    p_o = count_params(model_occ)
    p_mul = count_params(model_multi)

    print("\n[参数量对比]")
    print(f"Motion Only:   {p_m / 1e6:.2f} M")
    print(f"Occupancy Only:{p_o / 1e6:.2f} M")
    print(f"Multitask:     {p_mul / 1e6:.2f} M")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)