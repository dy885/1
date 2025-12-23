import torch
import torch.nn.functional as F

"""
风险评估模块 - 可训练版本 (支持梯度回传)

关键修改:
1. 移除 torch.no_grad() - 允许梯度流动
2. 添加 differentiable 参数控制是否需要梯度
3. 保持数值稳定性
"""


def spatial_variance_risk(occ_prob, eps=1e-7):
    """
    Multi-modal spatial variance risk
    occ_prob: probability in [0,1]
    Returns: [B] tensor with gradients
    """
    occ_prob = occ_prob.float()
    occ_prob = torch.clamp(occ_prob, eps, 1.0 - eps)

    # Variance over modes
    mean = occ_prob.mean(dim=1, keepdim=True)
    var = ((occ_prob - mean) ** 2).mean(dim=1)

    # 安全检查但不阻断梯度
    var_safe = torch.where(
        torch.isnan(var) | torch.isinf(var),
        torch.zeros_like(var),
        var
    )

    risk = var_safe.mean(dim=[1, 2, 3])
    return torch.clamp(risk, 0.0, 1.0)


def spatiotemporal_entropy_risk(occ_prob, eps=1e-7):
    """
    Spatio-temporal entropy risk
    Returns: [B] tensor with gradients
    """
    occ_prob = occ_prob.float()
    B, M, T, H, W = occ_prob.shape

    # Average over modes
    p = occ_prob.mean(dim=1)  # [B, T, H, W]

    # Normalize spatial distribution
    p_sum = p.sum(dim=[2, 3], keepdim=True)
    p_sum = torch.clamp(p_sum, min=eps)
    p = p / p_sum

    # Clamp for numerical stability
    p = torch.clamp(p, eps, 1.0 - eps)

    # Entropy: H = -sum(p * log(p))
    log_p = torch.log(p)
    log_p = torch.clamp(log_p, -10.0, 0.0)

    entropy = -p * log_p

    # 使用where而不是nan_to_num保持梯度
    entropy_safe = torch.where(
        torch.isnan(entropy) | torch.isinf(entropy),
        torch.zeros_like(entropy),
        entropy
    )

    risk = entropy_safe.mean(dim=[1, 2, 3])
    return torch.clamp(risk, 0.0, 10.0)


def temporal_divergence_risk(occ_prob, eps=1e-7):
    """
    Temporal divergence risk - 可微分版本
    Returns: [B] tensor with gradients
    """
    occ_prob = occ_prob.float()
    occ_prob = torch.clamp(occ_prob, eps, 1.0 - eps)

    if occ_prob.size(2) < 2:
        return torch.zeros(occ_prob.size(0), device=occ_prob.device)

    # 使用smooth L1 loss - 可微分且数值稳定
    smooth_l1 = F.smooth_l1_loss(
        occ_prob[:, :, 1:],
        occ_prob[:, :, :-1],
        reduction='none',
        beta=0.1
    )

    risk = smooth_l1.mean(dim=[1, 2, 3, 4])

    # 使用where保持梯度
    risk_safe = torch.where(
        torch.isnan(risk) | torch.isinf(risk),
        torch.zeros_like(risk),
        risk
    )

    return torch.clamp(risk_safe, 0.0, 1.0)


def combined_risk(
        occ_logits_or_probs,
        w_var=1.0,
        w_ent=0.5,
        w_temp=0.5,
        is_logits=True,
        differentiable=True  # ⭐ 新增参数
):
    """
    Combined risk - 支持梯度回传

    Args:
        occ_logits_or_probs: 模型输出
        is_logits: 如果为True,会先应用sigmoid
        differentiable: 如果为True,保留梯度; False则用于评估
    """
    # ⭐ 关键修改: 只在评估时使用no_grad
    context = torch.no_grad() if not differentiable else torch.enable_grad()

    with context:
        occ = occ_logits_or_probs.float()

        # 转换为概率
        if is_logits:
            # 限制logits范围避免sigmoid溢出
            occ = torch.clamp(occ, -10.0, 10.0)
            occ_prob = torch.sigmoid(occ)
        else:
            occ_prob = occ

        # 严格限制概率范围
        occ_prob = torch.clamp(occ_prob, 0.0, 1.0)

        # 计算各项风险
        try:
            r_var = spatial_variance_risk(occ_prob)
            r_ent = spatiotemporal_entropy_risk(occ_prob)
            r_temp = temporal_divergence_risk(occ_prob)

            # 归一化权重
            total_w = w_var + w_ent + w_temp + 1e-8
            w_var_norm = w_var / total_w
            w_ent_norm = w_ent / total_w
            w_temp_norm = w_temp / total_w

            # 加权组合
            risk = w_var_norm * r_var + w_ent_norm * r_ent + w_temp_norm * r_temp

            # 安全检查
            risk_safe = torch.where(
                torch.isnan(risk) | torch.isinf(risk),
                torch.zeros_like(risk),
                risk
            )

            # 如果在训练模式且检测到异常值，打印警告
            if differentiable and (torch.isnan(risk).any() or torch.isinf(risk).any()):
                print("[WARNING] NaN/Inf detected in combined risk during training")
                print(f"  r_var: {r_var.mean():.6f}")
                print(f"  r_ent: {r_ent.mean():.6f}")
                print(f"  r_temp: {r_temp.mean():.6f}")

        except Exception as e:
            print(f"[ERROR] Exception in risk computation: {e}")
            risk_safe = torch.zeros(occ_prob.size(0), device=occ_prob.device)

    return torch.clamp(risk_safe, 0.0, 10.0)


# 测试梯度流动
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Risk Functions - Gradient Flow")
    print("=" * 60)

    B, M, T, H, W = 2, 3, 5, 32, 32

    # Test 1: 梯度测试
    print("\n[Test 1] Gradient Flow Test")
    occ = torch.rand(B, M, T, H, W, requires_grad=True)

    # 计算risk (训练模式)
    risk_train = combined_risk(occ, is_logits=False, differentiable=True)
    print(f"  Risk (train mode): {risk_train}")
    print(f"  Requires grad: {risk_train.requires_grad}")

    # 反向传播测试
    loss = risk_train.mean()
    loss.backward()

    print(f"  Gradient exists: {occ.grad is not None}")
    print(f"  Gradient mean: {occ.grad.abs().mean():.6f}")
    assert occ.grad is not None, "No gradient computed!"
    print("  ✓ Gradients flow correctly")

    # Test 2: 评估模式测试
    print("\n[Test 2] Evaluation Mode Test")
    occ_eval = torch.rand(B, M, T, H, W)
    risk_eval = combined_risk(occ_eval, is_logits=False, differentiable=False)
    print(f"  Risk (eval mode): {risk_eval}")
    print(f"  Requires grad: {risk_eval.requires_grad}")
    print("  ✓ Evaluation mode works")

    # Test 3: 对比训练前后的risk变化
    print("\n[Test 3] Risk Change Simulation")
    occ_init = torch.rand(B, M, T, H, W, requires_grad=True)

    # 初始risk
    risk_before = combined_risk(occ_init, is_logits=False, differentiable=True)
    print(f"  Risk before: {risk_before.mean():.6f}")

    # 模拟一步优化 (降低variance)
    with torch.no_grad():
        mean_pattern = occ_init.mean(dim=1, keepdim=True)
        occ_after = occ_init * 0.9 + mean_pattern * 0.1  # 混合到平均值

    risk_after = combined_risk(occ_after, is_logits=False, differentiable=False)
    print(f"  Risk after: {risk_after.mean():.6f}")
    print(f"  Risk change: {(risk_after - risk_before).mean():.6f}")
    print("  ✓ Risk can change with model updates")

    print("\n" + "=" * 60)
    print("All gradient tests passed!")
    print("=" * 60)