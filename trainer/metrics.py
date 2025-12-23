"""
评估指标模块 - 修复设备问题
"""

import torch
from trainer.risk_evaluator import RiskEvaluator

class Metrics:
    """评估指标计算 - 确保所有计算在同一设备上"""

    def __init__(self, risk_w_var=1.0, risk_w_ent=0.5, risk_w_temp=0.5):
        self.risk_evaluator = RiskEvaluator(risk_w_var, risk_w_ent, risk_w_temp)

    def compute_occ_metrics(self, pred, target, threshold=0.5):
        """
        Occupancy 指标: IoU, Precision, Recall, F1, Risk
        ✅ 修复：确保所有计算在同一设备上
        """
        # ✅ 获取设备
        device = pred.device

        B, M, T, H, W = pred.shape
        best_pred = pred[:, 0]  # [B, T, H, W] 取第一个模态

        # 处理 target 维度
        if target.dim() == 5:  # [B, T, C, H, W]
            target = target.squeeze(2)  # [B, T, H, W]
        if target.dim() == 3:  # [B, H, W]
            target = target.unsqueeze(1).expand(-1, T, -1, -1)
        if target.size(1) != T:
            target = target.expand(-1, T, -1, -1)

        # ✅ 确保在同一设备上
        target = target.to(device)

        # 转为 float32
        pred_f = best_pred.float()
        target_f = target.float()

        # ✅ threshold 转为 tensor 并放在正确设备上
        threshold_tensor = torch.tensor(threshold, device=device, dtype=torch.float32)

        pred_bin = (pred_f > threshold_tensor).float()
        target_bin = (target_f > threshold_tensor).float()

        # ✅ 所有计算现在都在同一设备上
        tp = (pred_bin * target_bin).sum()
        fp = (pred_bin * (1 - target_bin)).sum()
        fn = ((1 - pred_bin) * target_bin).sum()

        intersection = tp
        union = pred_bin.sum() + target_bin.sum() - intersection

        # ✅ eps 也在正确设备上
        eps = torch.tensor(1e-6, device=device, dtype=torch.float32)

        metrics = {
            'iou': (intersection / (union + eps)).item(),
            'precision': (tp / (tp + fp + eps)).item(),
            'recall': (tp / (tp + fn + eps)).item(),
        }

        # ✅ F1 计算
        prec = metrics['precision']
        rec = metrics['recall']
        metrics['f1'] = 2 * prec * rec / (prec + rec + 1e-6)

        # ✅ 风险指标 - 确保 pred 在正确设备上
        try:
            risk_metrics = self.risk_evaluator.get_risk_summary(pred)
            metrics.update(risk_metrics)
        except Exception as e:
            print(f"[WARNING] Risk computation failed in metrics: {e}")
            # 提供默认值
            metrics.update({
                'combined_mean': 0.0,
                'combined_std': 0.0,
                'spatial_variance_mean': 0.0,
                'spatial_variance_std': 0.0
            })

        return metrics

    def compute_motion_metrics(self, pred, target):
        """
        Motion 指标: ADE, FDE, minADE, minFDE
        ✅ 修复：确保所有计算在同一设备上
        """
        # ✅ 获取设备
        device = pred.device
        B, M, T, _ = pred.shape

        # ✅ 确保 target 在同一设备上
        target = target.to(device)

        # 转为 float32
        pred_f = pred.float()
        target_f = target.float()

        # ✅ 扩展 target
        target_exp = target_f.unsqueeze(1).expand(-1, M, -1, -1)

        # ✅ 计算误差（都在同一设备上）
        errors = torch.sqrt(((pred_f - target_exp) ** 2).sum(dim=-1))

        # ✅ 找最佳模态
        mode_ade = errors.mean(dim=2)
        best_mode = mode_ade.argmin(dim=1)
        batch_idx = torch.arange(B, device=device)

        return {
            'ade': errors[batch_idx, best_mode].mean().item(),
            'fde': errors[batch_idx, best_mode, -1].mean().item(),
            'min_ade': mode_ade.min(dim=1)[0].mean().item(),
            'min_fde': errors[:, :, -1].min(dim=1)[0].mean().item(),
        }


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Metrics with Device Consistency")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")

    metrics = Metrics()

    # Test 1: Occupancy Metrics
    print("\n[Test 1] Occupancy Metrics")
    B, M, T, H, W = 2, 3, 5, 32, 32

    pred = torch.randn(B, M, T, H, W).to(device)
    target = torch.randn(B, 1, H, W).to(device)

    try:
        result = metrics.compute_occ_metrics(pred, target)
        print(f"  ✓ IoU: {result['iou']:.4f}")
        print(f"  ✓ Precision: {result['precision']:.4f}")
        print(f"  ✓ Recall: {result['recall']:.4f}")
        print(f"  ✓ F1: {result['f1']:.4f}")
        print("  ✓ Occupancy metrics passed")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: Motion Metrics
    print("\n[Test 2] Motion Metrics")
    pred_motion = torch.randn(B, M, T, 2).to(device)
    target_motion = torch.randn(B, T, 2).to(device)

    try:
        result = metrics.compute_motion_metrics(pred_motion, target_motion)
        print(f"  ✓ ADE: {result['ade']:.4f}")
        print(f"  ✓ FDE: {result['fde']:.4f}")
        print("  ✓ Motion metrics passed")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: Mixed Device Test (should not fail now)
    print("\n[Test 3] Cross-Device Safety Test")
    pred_gpu = torch.randn(B, M, T, H, W).to(device)
    target_cpu = torch.randn(B, 1, H, W)  # 故意放在 CPU

    try:
        result = metrics.compute_occ_metrics(pred_gpu, target_cpu)
        print(f"  ✓ Cross-device handling works: IoU={result['iou']:.4f}")
    except Exception as e:
        print(f"  ✗ Cross-device test failed: {e}")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)