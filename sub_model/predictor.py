import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvOccupancyPredictor(nn.Module):
    """
    Multi-modal convolutional occupancy predictor with temporal modeling
    修复: 输出尺寸从64x64改为640x640
    """

    def __init__(self, in_dim=128, modes=5, future_steps=1, out_size=(640, 640), use_temporal_conv=True):
        super().__init__()
        self.modes = modes
        self.future_steps = future_steps
        self.H, self.W = out_size
        self.use_temporal_conv = use_temporal_conv

        # 扩大初始特征图尺寸
        self.fc = nn.Linear(in_dim, 512 * 8 * 8)

        self.confidence = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, modes),
            nn.Softmax(dim=-1)
        )

        self.decoders = nn.ModuleList([self._build_decoder() for _ in range(modes)])

        if use_temporal_conv:
            self.temporal_refine = nn.ModuleList([
                nn.Conv1d(future_steps, future_steps, kernel_size=3, padding=1, groups=future_steps)
                for _ in range(modes)
            ])

    def _build_decoder(self):
        """
        构建上采样解码器: 8x8 -> 640x640
        8 -> 16 -> 32 -> 64 -> 128 -> 256 -> 512 -> 640
        """
        return nn.Sequential(
            # 8 -> 16
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 16 -> 32
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 32 -> 64
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 64 -> 128
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 128 -> 256
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            # 256 -> 512
            nn.ConvTranspose2d(16, 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            # 512 -> 640 (使用插值调整到精确尺寸)
            nn.ConvTranspose2d(8, self.future_steps, 3, 1, 1),
        )

    def forward(self, x, return_confidence=False):
        B = x.size(0)
        seed = self.fc(x).view(B, 512, 8, 8)

        futures = []
        for i, decoder in enumerate(self.decoders):
            occ = decoder(seed)  # [B, T_f, H', W']

            # 插值到目标尺寸
            if occ.shape[-2:] != (self.H, self.W):
                occ = F.interpolate(occ, size=(self.H, self.W), mode='bilinear', align_corners=False)

            if self.use_temporal_conv and self.future_steps > 1:
                B_sz, T, H, W = occ.shape
                occ = torch.sigmoid(self.temporal_refine[i](occ.view(B_sz, T, -1)).view(B_sz, T, H, W))

            futures.append(occ)

        out = torch.stack(futures, dim=1)  # [B, M, T_f, H, W]

        if return_confidence:
            return out, self.confidence(x)
        return out


class MotionPredictor(nn.Module):
    """
    Multi-modal Trajectory Predictor (Regression based)
    """

    def __init__(self, in_dim=128, modes=5, future_steps=20, out_dim=2):
        super().__init__()
        self.modes = modes
        self.future_steps = future_steps
        self.out_dim = out_dim

        self.shared_mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True)
        )

        self.confidence = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, modes),
            nn.Softmax(dim=-1)
        )

        self.reg_head = nn.Linear(256, modes * future_steps * out_dim)

    def forward(self, x, return_confidence=False):
        B = x.size(0)
        feat = self.shared_mlp(x)

        flat_traj = self.reg_head(feat)
        traj = flat_traj.view(B, self.modes, self.future_steps, self.out_dim)

        if return_confidence:
            conf = self.confidence(feat)
            return traj, conf
        return traj


class MultiTaskPredictor(nn.Module):
    """
    Multi-task predictor: Combines Occupancy (Grid) and Motion (Trajectory)
    """

    def __init__(
            self,
            in_dim,
            modes=5,
            future_steps=20,
            out_size=(640, 640),
            out_dim=2,
            shared_backbone=True
    ):
        super().__init__()
        self.shared_backbone = shared_backbone

        if shared_backbone:
            self.backbone = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.LayerNorm(in_dim),
                nn.ReLU(inplace=True)
            )
            backbone_dim = in_dim
        else:
            self.backbone = nn.Identity()
            backbone_dim = in_dim

        self.occ_head = ConvOccupancyPredictor(
            in_dim=backbone_dim,
            modes=modes,
            future_steps=future_steps,
            out_size=out_size
        )

        self.motion_head = MotionPredictor(
            in_dim=backbone_dim,
            modes=modes,
            future_steps=future_steps,
            out_dim=out_dim
        )

        self.task_weights = nn.Parameter(torch.ones(2))

    def forward(self, x, task="both", return_confidence=False):
        feat = self.backbone(x)

        if task == "occ":
            return self.occ_head(feat, return_confidence)
        elif task == "motion":
            return self.motion_head(feat, return_confidence)
        elif task == "both":
            occ_out = self.occ_head(feat, return_confidence)
            motion_out = self.motion_head(feat, return_confidence)
            return {"occ": occ_out, "motion": motion_out}
        else:
            raise ValueError(f"Unknown task: {task}")

    def get_task_weights(self):
        return F.softmax(self.task_weights, dim=0)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Predictor with 640x640 output")
    print("=" * 60)

    B, C = 4, 128
    x = torch.randn(B, C)

    # 测试Occupancy Predictor
    print("\n[Test 1] Occupancy Predictor")
    occ_pred = ConvOccupancyPredictor(C, modes=3, future_steps=1, out_size=(640, 640))
    occ_out, conf = occ_pred(x, return_confidence=True)
    print(f"  Input: {x.shape}")
    print(f"  Output: {occ_out.shape}")  # Expected: [4, 3, 1, 640, 640]
    print(f"  Confidence: {conf.shape}")  # Expected: [4, 3]
    assert occ_out.shape == (4, 3, 1, 640, 640), f"Wrong shape: {occ_out.shape}"
    print("  ✓ Occupancy test passed")

    # 测试Motion Predictor
    print("\n[Test 2] Motion Predictor")
    mot_pred = MotionPredictor(C, modes=3, future_steps=10)
    mot_out, mot_conf = mot_pred(x, return_confidence=True)
    print(f"  Output: {mot_out.shape}")  # Expected: [4, 3, 10, 2]
    print(f"  Confidence: {mot_conf.shape}")
    assert mot_out.shape == (4, 3, 10, 2)
    print("  ✓ Motion test passed")

    # 测试MultiTask
    print("\n[Test 3] MultiTask Predictor")
    multi_pred = MultiTaskPredictor(C, modes=3, future_steps=1, out_size=(640, 640))
    outputs = multi_pred(x, task="both", return_confidence=True)

    occ_res, occ_conf = outputs["occ"]
    mot_res, mot_conf = outputs["motion"]

    print(f"  Occupancy: {occ_res.shape}")
    print(f"  Motion: {mot_res.shape}")
    assert occ_res.shape == (4, 3, 1, 640, 640)
    print("  ✓ MultiTask test passed")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)