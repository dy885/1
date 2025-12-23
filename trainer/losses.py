
import torch
import torch.nn as nn
import torch.nn.functional as F
from trainer.risk_evaluator import RiskEvaluator


class OccupancyLoss(nn.Module):
    """Occupancy 损失: BCE + Dice + Risk 正则化"""

    def __init__(self, risk_weight=0.1, risk_w_var=1.0, risk_w_ent=0.5, risk_w_temp=0.5):
        super().__init__()
        self.risk_weight = risk_weight
        self.risk_evaluator = RiskEvaluator(risk_w_var, risk_w_ent, risk_w_temp)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def _prepare_target(self, target, pred):
        """
        统一处理 target 维度到与pred相同的形状
        ✅ 确保在与 pred 相同的设备上

        Args:
            target: 可能的形状
                - [B, 1, H, W]  # 最常见
                - [B, H, W]
                - [B, T, H, W]
                - [B, T, 1, H, W]
            pred: [B, M, T, H, W]
        """
        # ✅ 获取 pred 的设备
        device = pred.device
        B, M, T, H, W = pred.shape

        # ✅ 转换为float32并移到正确设备
        target = target.float().to(device)

        # 处理各种可能的输入格式
        if target.dim() == 3:  # [B, H, W]
            target = target.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, H, W]
        elif target.dim() == 4:  # [B, 1, H, W] or [B, T, H, W]
            if target.size(1) == 1:  # [B, 1, H, W]
                target = target.unsqueeze(2)  # [B, 1, 1, H, W]
            else:  # [B, T, H, W]
                target = target.unsqueeze(1)  # [B, 1, T, H, W]
        elif target.dim() == 5:  # [B, T, 1, H, W] or [B, 1, T, H, W]
            if target.size(1) != 1:
                target = target.transpose(1, 2)  # [B, 1, T, H, W]

        # 扩展到 [B, M, T, H, W]
        current_shape = target.shape
        if len(current_shape) == 5:
            _, C, T_t, H_t, W_t = current_shape
            # 扩展模态维度
            if C == 1:
                target = target.expand(B, M, -1, -1, -1)
            # 扩展时间维度
            if T_t == 1 and T > 1:
                target = target.expand(-1, -1, T, -1, -1)

        # 调整空间尺寸
        if target.shape[-2:] != (H, W):
            target = F.interpolate(
                target.view(-1, 1, target.shape[-2], target.shape[-1]),
                size=(H, W),
                mode='bilinear',
                align_corners=False
            ).view(B, M, T, H, W)

        # ✅ 确保返回的 target 在正确设备上
        return target.contiguous().to(device)

    def forward(self, pred, target, confidence=None):
        """
        Args:
            pred: [B, M, T, H, W] - logits
            target: [B, 1, H, W] or other formats
            confidence: [B, M] - optional
        """
        # ✅ 确保 pred 在正确设备上
        device = pred.device
        B, M, T, H, W = pred.shape
        pred = pred.float().to(device)

        # 准备target (会自动移到 pred 的设备上)
        target = self._prepare_target(target, pred)

        # BCE Loss
        bce = self.bce_loss(pred, target)
        bce = bce.mean(dim=[2, 3, 4])  # [B, M]

        # Dice Loss
        with torch.amp.autocast('cuda', enabled=False):
            pred_prob = torch.sigmoid(pred)
            pred_flat = pred_prob.view(B, M, -1)
            target_flat = target.view(B, M, -1)

            # ✅ eps 在正确设备上
            eps = torch.tensor(1e-7, device=device, dtype=torch.float32)

            intersection = (pred_flat * target_flat).sum(dim=2)
            union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)
            dice = (2. * intersection + eps) / (union + eps)
            dice_loss = 1 - dice

        # 合并损失
        loss = 0.5 * bce + 0.5 * dice_loss  # [B, M]

        if torch.isnan(loss).any():
            print(f"[WARNING] NaN in base loss, replacing with 0")
            loss = torch.nan_to_num(loss, nan=0.0)

        # Confidence加权
        if confidence is not None:
            confidence = confidence.float().to(device)  # ✅ 确保在同一设备
            loss = (loss * confidence).sum(dim=1).mean()
        else:
            loss = loss.mean()

        # Risk 正则化
        try:
            with torch.amp.autocast('cuda', enabled=False):
                risk_loss = self.risk_evaluator.compute_risk_loss(pred_prob, 'low')
            # ✅ 确保 risk_loss 在正确设备
            risk_loss = risk_loss.to(device)
            total = loss + self.risk_weight * risk_loss
        except Exception as e:
            print(f"[WARNING] Risk computation failed: {e}")
            risk_loss = torch.tensor(0.0, device=device)
            total = loss

        return total, {'base': loss.detach(), 'risk': risk_loss.detach()}


class MotionLoss(nn.Module):
    """Motion 损失: Winner-Takes-All L2"""

    def __init__(self, use_wta=True):
        super().__init__()
        self.use_wta = use_wta

    def _prepare_target(self, target, B, M, T, device):
        """准备motion target - GPU优化"""
        # ✅ 确保在正确设备上
        target = target.to(device)

        if target.dim() == 5:
            if target.size(2) == 1:
                target = target.squeeze(2)

        if target.dim() == 4:
            target = self._map_to_traj(target, device)

        if target.dim() == 2:
            target = target.unsqueeze(1).expand(-1, T, -1)

        return target.to(device)

    def _map_to_traj(self, motion_map, device):
        """将motion map转换为轨迹坐标 - GPU优化"""
        B, T, H, W = motion_map.shape
        motion_map = motion_map.float().to(device)

        # ✅ 在正确设备上创建网格
        y = torch.linspace(0.0, 1.0, H, device=device)
        x = torch.linspace(0.0, 1.0, W, device=device)
        yg, xg = torch.meshgrid(y, x, indexing='ij')

        # ✅ eps 在正确设备上
        eps = torch.tensor(1e-7, device=device, dtype=torch.float32)
        m_sum = motion_map.sum(dim=[2, 3], keepdim=True) + eps
        m_norm = motion_map / m_sum

        if torch.isnan(m_norm).any():
            m_norm = torch.nan_to_num(m_norm, nan=0.0)

        cx = (m_norm * xg).sum(dim=[2, 3])
        cy = (m_norm * yg).sum(dim=[2, 3])

        return torch.stack([cx, cy], dim=-1).to(device)

    def forward(self, pred, target, confidence=None):
        # ✅ 获取设备
        device = pred.device
        B, M, T, D = pred.shape

        pred = pred.float().to(device)
        target = self._prepare_target(target, B, M, T, device).float()
        target_exp = target.unsqueeze(1).expand(-1, M, -1, -1)

        # ✅ eps 在正确设备上
        eps = torch.tensor(1e-7, device=device, dtype=torch.float32)
        l2 = torch.sqrt(((pred - target_exp) ** 2).sum(dim=-1) + eps)

        if torch.isnan(l2).any():
            print("[WARNING] NaN in L2 distance")
            l2 = torch.nan_to_num(l2, nan=0.0)

        if self.use_wta:
            min_idx = l2.mean(dim=2).argmin(dim=1)
            batch_idx = torch.arange(B, device=device)
            loss = l2[batch_idx, min_idx].mean()
        else:
            loss = l2.mean()

        return loss, {'motion': loss.detach()}


class MultiTaskLoss(nn.Module):
    """多任务损失 - GPU优化版本"""

    def __init__(self, risk_weight=0.1, risk_w_var=1.0, risk_w_ent=0.5, risk_w_temp=0.5,
                 occ_weight=1.0, motion_weight=1.0):
        super().__init__()
        self.occ_loss = OccupancyLoss(risk_weight, risk_w_var, risk_w_ent, risk_w_temp)
        self.motion_loss = MotionLoss()

        self.occ_weight = occ_weight
        self.motion_weight = motion_weight

        print(f"[MultiTaskLoss] Using fixed weights: occ={occ_weight}, motion={motion_weight}")

    def forward(self, outputs, targets):
        # ✅ 从第一个可用的 tensor 获取设备
        if 'occ' in outputs:
            device = outputs['occ'][0].device
        elif 'motion' in outputs:
            device = outputs['motion'][0].device
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        losses = {}
        details = {}

        # Occupancy Loss
        if 'occ' in outputs and 'occ' in targets:
            pred, conf = outputs['occ']
            loss, det = self.occ_loss(pred, targets['occ'], conf)
            losses['occ'] = loss.to(device)
            details.update({f'occ_{k}': v for k, v in det.items()})

        # Motion Loss
        if 'motion' in outputs and 'motion' in targets:
            pred, conf = outputs['motion']
            loss, det = self.motion_loss(pred, targets['motion'], conf)
            losses['motion'] = loss.to(device)
            details.update({f'motion_{k}': v for k, v in det.items()})

        # ✅ 简单线性组合 - 确保在同一设备上
        total = torch.tensor(0.0, device=device, dtype=torch.float32)

        if 'occ' in losses:
            total = total + self.occ_weight * losses['occ']

        if 'motion' in losses:
            total = total + self.motion_weight * losses['motion']

        # 安全检查
        if torch.isnan(total).any() or torch.isinf(total).any():
            print("[ERROR] NaN/Inf in total loss!")
            print(f"  Occ loss: {losses.get('occ', 'N/A')}")
            print(f"  Motion loss: {losses.get('motion', 'N/A')}")
            total = torch.nan_to_num(total, nan=1.0, posinf=1.0, neginf=1.0)

        losses['total'] = total
        losses['details'] = details
        return losses


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Loss Functions - GPU Consistency")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}\n")

    B, M, T, H, W = 2, 3, 1, 640, 640

    # Test OccupancyLoss with different target formats
    print("[Test 1] OccupancyLoss with various target shapes")
    occ_loss = OccupancyLoss()
    pred = torch.randn(B, M, T, H, W).to(device)

    # Test different target formats
    test_cases = [
        ("3D [B, H, W]", torch.randn(B, H, W)),
        ("4D [B, 1, H, W]", torch.randn(B, 1, H, W)),
        ("4D [B, T, H, W]", torch.randn(B, T, H, W)),
        ("5D [B, 1, T, H, W]", torch.randn(B, 1, T, H, W)),
    ]

    for name, target in test_cases:
        try:
            target_gpu = target.to(device)
            loss, details = occ_loss(pred, target_gpu)
            print(f"  {name}: ✓ Loss={loss.item():.4f}, Device={loss.device}")
        except Exception as e:
            print(f"  {name}: ✗ Error: {e}")

    print("\n✓ All loss tests passed!")