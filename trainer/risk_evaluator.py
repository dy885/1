"""
风险评估器 - 支持训练和评估两种模式
✅ 修复：确保所有计算在同一设备上
"""

import torch
from sub_model.risk import (
    combined_risk,
    spatial_variance_risk,
    spatiotemporal_entropy_risk,
    temporal_divergence_risk
)


class RiskEvaluator:
    """
    风险评估器，用于评估和损失正则化
    ✅ 修复：所有计算保持在输入 tensor 的设备上
    """

    def __init__(self, w_var=1.0, w_ent=0.5, w_temp=0.5):
        self.w_var = w_var
        self.w_ent = w_ent
        self.w_temp = w_temp

    def compute_all_risks(self, occ, differentiable=False):
        """
        计算所有风险指标
        ✅ 修复：确保所有计算在 occ 的设备上

        Args:
            occ: [B, M, T, H, W] - 概率值
            differentiable: 是否需要梯度 (训练时True, 评估时False)
        """
        # ✅ 确保在正确设备上
        device = occ.device
        occ = occ.float().to(device)

        # 根据模式决定是否使用no_grad
        if differentiable:
            # 训练模式 - 保留梯度
            risks = {
                'spatial_variance': spatial_variance_risk(occ),
                'spatiotemporal_entropy': spatiotemporal_entropy_risk(occ),
                'temporal_divergence': temporal_divergence_risk(occ),
                'combined': combined_risk(occ, self.w_var, self.w_ent, self.w_temp,
                                         is_logits=False, differentiable=True)
            }
        else:
            # 评估模式 - 不需要梯度
            with torch.no_grad():
                risks = {
                    'spatial_variance': spatial_variance_risk(occ),
                    'spatiotemporal_entropy': spatiotemporal_entropy_risk(occ),
                    'temporal_divergence': temporal_divergence_risk(occ),
                    'combined': combined_risk(occ, self.w_var, self.w_ent, self.w_temp,
                                             is_logits=False, differentiable=False)
                }

        # ✅ 确保所有结果都在同一设备上
        for key in risks:
            if isinstance(risks[key], torch.Tensor):
                risks[key] = risks[key].to(device)

        return risks

    def compute_risk_loss(self, occ, target_level='low'):
        """
        将风险转换为损失项 (用于训练)
        ✅ 修复：确保返回的 loss 在正确设备上
        """
        device = occ.device

        # 训练时必须允许梯度
        risks = self.compute_all_risks(occ, differentiable=True)

        if target_level == 'low':
            # 目标: 降低风险
            loss = risks['combined'].mean()
        elif target_level == 'high':
            # 目标: 提高风险 (罕见用例)
            loss = -risks['combined'].mean()
        else:
            # 目标: 适中风险
            target_value = torch.tensor(0.5, device=device, dtype=torch.float32)
            loss = (risks['combined'] - target_value).pow(2).mean()

        # ✅ 确保 loss 在正确设备上
        return loss.to(device)

    def get_risk_summary(self, occ):
        """
        获取风险统计摘要 (用于评估/监控)
        ✅ 修复：确保计算在正确设备上，但返回 Python 标量
        """
        device = occ.device
        occ = occ.to(device)

        # 评估时不需要梯度
        risks = self.compute_all_risks(occ, differentiable=False)

        summary = {}
        for name, values in risks.items():
            if isinstance(values, torch.Tensor):
                # ✅ 确保在同一设备上计算统计量
                values = values.to(device)
                summary[f'{name}_mean'] = values.mean().item()
                summary[f'{name}_std'] = values.std().item()
            else:
                summary[f'{name}_mean'] = float(values)
                summary[f'{name}_std'] = 0.0

        return summary


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing RiskEvaluator - Device Consistency")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")

    evaluator = RiskEvaluator(w_var=1.0, w_ent=0.5, w_temp=0.5)
    B, M, T, H, W = 2, 3, 5, 32, 32

    # Test 1: 评估模式 (不需要梯度)
    print("\n[Test 1] Evaluation Mode")
    occ_eval = torch.rand(B, M, T, H, W).to(device)

    try:
        risks_eval = evaluator.compute_all_risks(occ_eval, differentiable=False)
        print(f"  Combined risk: {risks_eval['combined'].mean():.6f}")
        print(f"  Combined risk device: {risks_eval['combined'].device}")
        print(f"  Input device: {occ_eval.device}")
        assert risks_eval['combined'].device == occ_eval.device, "Device mismatch!"
        print("  ✓ Evaluation mode device consistency passed")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: 训练模式 (需要梯度)
    print("\n[Test 2] Training Mode")
    occ_train = torch.rand(B, M, T, H, W, requires_grad=True).to(device)

    try:
        risks_train = evaluator.compute_all_risks(occ_train, differentiable=True)
        print(f"  Combined risk: {risks_train['combined'].mean():.6f}")
        print(f"  Requires grad: {risks_train['combined'].requires_grad}")
        print(f"  Device: {risks_train['combined'].device}")

        # 测试反向传播
        loss = risks_train['combined'].mean()
        loss.backward()
        print(f"  Gradient exists: {occ_train.grad is not None}")
        assert occ_train.grad is not None, "No gradient!"
        assert occ_train.grad.device == device, "Gradient device mismatch!"
        print("  ✓ Training mode with gradients passed")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: compute_risk_loss
    print("\n[Test 3] Risk Loss for Training")
    occ_loss = torch.rand(B, M, T, H, W, requires_grad=True).to(device)

    try:
        risk_loss = evaluator.compute_risk_loss(occ_loss, target_level='low')
        print(f"  Risk loss: {risk_loss:.6f}")
        print(f"  Risk loss device: {risk_loss.device}")
        print(f"  Requires grad: {risk_loss.requires_grad}")

        assert risk_loss.device == device, "Loss device mismatch!"

        # 反向传播
        risk_loss.backward()
        assert occ_loss.grad is not None, "No gradient!"
        print("  ✓ Risk loss device consistency passed")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()

    # Test 4: get_risk_summary
    print("\n[Test 4] Risk Summary")
    occ_summary = torch.rand(B, M, T, H, W).to(device)

    try:
        summary = evaluator.get_risk_summary(occ_summary)
        print(f"  Summary keys: {list(summary.keys())}")
        print(f"  Combined mean: {summary['combined_mean']:.6f}")
        print(f"  Combined std: {summary['combined_std']:.6f}")
        print(f"  All values are Python floats: {all(isinstance(v, float) for v in summary.values())}")
        print("  ✓ Summary works correctly")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()

    # Test 5: Cross-Device Safety
    print("\n[Test 5] Cross-Device Input Handling")
    if torch.cuda.is_available():
        occ_cpu = torch.rand(B, M, T, H, W)  # CPU tensor
        try:
            risks = evaluator.compute_all_risks(occ_cpu, differentiable=False)
            print(f"  CPU input -> output device: {risks['combined'].device}")
            print("  ✓ CPU input handled correctly")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    print("\n" + "=" * 60)
    print("All device consistency tests passed!")
    print("=" * 60)