"""
快速设备检查脚本
在正式训练前验证所有组件的设备配置
"""

import torch
import argparse
import os
import sys
from torch.utils.data import DataLoader

# 导入项目模块
from sub_model.conflict_model import UAVConflictModel
from trainer.losses import OccupancyLoss, MotionLoss, MultiTaskLoss
from trainer.metrics import Metrics
from trainer.risk_evaluator import RiskEvaluator
from image_dataset import UAVImageDataset, UAVSimpleImageDataset
from validation_visualizer import ValidationVisualizer


class DeviceChecker:
    """设备一致性检查器"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.device == 'cuda' else 'cpu')
        self.errors = []
        self.warnings = []

        print("=" * 70)
        print("🔍 UAV 模型设备快速检查")
        print("=" * 70)
        print(f"目标设备: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print("=" * 70 + "\n")

    def log_error(self, module, message):
        """记录错误"""
        error_msg = f"❌ [{module}] {message}"
        self.errors.append(error_msg)
        print(error_msg)

    def log_warning(self, module, message):
        """记录警告"""
        warning_msg = f"⚠️  [{module}] {message}"
        self.warnings.append(warning_msg)
        print(warning_msg)

    def log_success(self, module, message):
        """记录成功"""
        print(f"✅ [{module}] {message}")

    def check_dataset(self):
        """检查数据集加载器"""
        print("\n" + "─" * 70)
        print("1️⃣  检查数据集...")
        print("─" * 70)

        try:
            use_occ = self.config.mode in ['occupancy', 'multitask']
            use_motion = self.config.mode in ['motion', 'multitask']

            if self.config.dataset_type == 'sequence':
                dataset = UAVImageDataset(
                    root_dir=self.config.data_dir,
                    history_frames=self.config.history_frames,
                    use_occ=use_occ,
                    use_motion=use_motion,
                    img_size=tuple(self.config.img_size)
                )
            else:
                dataset = UAVSimpleImageDataset(
                    root_dir=self.config.data_dir,
                    img_size=tuple(self.config.img_size),
                    use_occ=use_occ,
                    use_motion=use_motion
                )

            if len(dataset) == 0:
                self.log_error("Dataset", "数据集为空")
                return False

            self.log_success("Dataset", f"找到 {len(dataset)} 个样本")

            # 创建小型 DataLoader (只取 2 个样本)
            mini_loader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                pin_memory=torch.cuda.is_available()
            )

            # 检查第一个 batch
            history, targets = next(iter(mini_loader))

            self.log_success("Dataset", f"History shape: {history.shape}")
            for key, value in targets.items():
                if isinstance(value, torch.Tensor):
                    self.log_success("Dataset", f"Target '{key}' shape: {value.shape}")

            # 测试设备转移
            history_gpu = history.to(self.device, non_blocking=True)
            if history_gpu.device.type != self.device.type:
                self.log_error("Dataset", f"数据无法移动到 {self.device}")
                return False

            self.log_success("Dataset", f"数据成功移动到 {self.device}")

            # 保存样本用于后续测试
            self.test_history = history_gpu
            self.test_targets = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                                for k, v in targets.items()}

            return True

        except Exception as e:
            self.log_error("Dataset", f"异常: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def check_model(self):
        """检查模型"""
        print("\n" + "─" * 70)
        print("2️⃣  检查模型...")
        print("─" * 70)

        try:
            self.model = UAVConflictModel(
                mode=self.config.mode,
                hidden_dim=self.config.hidden_dim,
                modes=self.config.num_modes,
                encoder_backbone=self.config.backbone,
                future_steps=self.config.future_steps
            ).to(self.device)

            total_params = sum(p.numel() for p in self.model.parameters())
            self.log_success("Model", f"模型创建成功，参数量: {total_params / 1e6:.2f}M")

            # 检查所有参数是否在正确设备上
            devices = {p.device for p in self.model.parameters()}
            if len(devices) > 1:
                self.log_error("Model", f"模型参数在多个设备上: {devices}")
                return False

            model_device = next(self.model.parameters()).device
            if model_device.type != self.device.type:
                self.log_error("Model", f"模型在 {model_device}，期望 {self.device}")
                return False

            self.log_success("Model", f"所有参数都在 {self.device} 上")

            # ✅ 修复：使用训练模式进行前向传播（保留梯度）
            self.model.train()
            outputs = self.model(self.test_history, return_confidence=True)

            # 检查输出设备
            if self.config.mode == 'multitask':
                occ_pred, occ_conf = outputs['occ']
                if occ_pred.device.type != self.device.type:
                    self.log_error("Model", f"Occupancy输出在 {occ_pred.device}")
                    return False
                self.log_success("Model", f"Occupancy输出: {occ_pred.shape}, device: {occ_pred.device}")

                motion_pred, motion_conf = outputs['motion']
                if motion_pred.device.type != self.device.type:
                    self.log_error("Model", f"Motion输出在 {motion_pred.device}")
                    return False
                self.log_success("Model", f"Motion输出: {motion_pred.shape}, device: {motion_pred.device}")

            elif self.config.mode == 'occupancy':
                pred, conf = outputs
                if pred.device.type != self.device.type:
                    self.log_error("Model", f"输出在 {pred.device}")
                    return False
                self.log_success("Model", f"Occupancy输出: {pred.shape}, device: {pred.device}")

            else:  # motion
                pred, conf = outputs
                if pred.device.type != self.device.type:
                    self.log_error("Model", f"输出在 {pred.device}")
                    return False
                self.log_success("Model", f"Motion输出: {pred.shape}, device: {pred.device}")

            self.test_outputs = outputs
            return True

        except Exception as e:
            self.log_error("Model", f"异常: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def check_loss(self):
        """检查损失函数"""
        print("\n" + "─" * 70)
        print("3️⃣  检查损失函数...")
        print("─" * 70)

        try:
            if self.config.mode == 'multitask':
                criterion = MultiTaskLoss(
                    risk_weight=self.config.risk_weight,
                    risk_w_var=self.config.risk_w_var,
                    risk_w_ent=self.config.risk_w_ent,
                    risk_w_temp=self.config.risk_w_temp,
                    occ_weight=self.config.occ_weight,
                    motion_weight=self.config.motion_weight
                ).to(self.device)
            elif self.config.mode == 'occupancy':
                criterion = OccupancyLoss(
                    self.config.risk_weight,
                    self.config.risk_w_var,
                    self.config.risk_w_ent,
                    self.config.risk_w_temp
                ).to(self.device)
            else:
                criterion = MotionLoss().to(self.device)

            self.log_success("Loss", f"损失函数创建成功 ({self.config.mode} mode)")

            # 测试损失计算
            if self.config.mode == 'multitask':
                losses = criterion(self.test_outputs, self.test_targets)
                loss = losses['total']
            elif self.config.mode == 'occupancy':
                pred, conf = self.test_outputs
                loss, details = criterion(pred, self.test_targets['occ'], conf)
            else:
                pred, conf = self.test_outputs
                loss, details = criterion(pred, self.test_targets['motion'], conf)

            if loss.device.type != self.device.type:
                self.log_error("Loss", f"损失在 {loss.device}，期望 {self.device}")
                return False

            self.log_success("Loss", f"损失计算成功: {loss.item():.4f}, device: {loss.device}")

            # 测试反向传播
            self.model.train()
            self.model.zero_grad()
            loss.backward()

            # 检查梯度设备
            grad_devices = {p.grad.device for p in self.model.parameters() if p.grad is not None}
            if len(grad_devices) > 1:
                self.log_error("Loss", f"梯度在多个设备上: {grad_devices}")
                return False

            self.log_success("Loss", "反向传播成功，梯度计算正确")

            return True

        except Exception as e:
            self.log_error("Loss", f"异常: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def check_metrics(self):
        """检查评估指标"""
        print("\n" + "─" * 70)
        print("4️⃣  检查评估指标...")
        print("─" * 70)

        try:
            metrics = Metrics(
                self.config.risk_w_var,
                self.config.risk_w_ent,
                self.config.risk_w_temp
            )

            self.log_success("Metrics", "指标计算器创建成功")

            # 测试指标计算
            if self.config.mode in ['occupancy', 'multitask']:
                if self.config.mode == 'multitask':
                    pred, conf = self.test_outputs['occ']
                else:
                    pred, conf = self.test_outputs

                result = metrics.compute_occ_metrics(pred, self.test_targets['occ'])

                self.log_success("Metrics", f"IoU: {result['iou']:.4f}")
                self.log_success("Metrics", f"Precision: {result['precision']:.4f}")
                self.log_success("Metrics", f"Recall: {result['recall']:.4f}")

                # 检查返回值是否为 Python 标量
                if not isinstance(result['iou'], float):
                    self.log_warning("Metrics", "指标返回值不是 Python float")

            return True

        except Exception as e:
            self.log_error("Metrics", f"异常: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def check_risk_evaluator(self):
        """检查风险评估器"""
        print("\n" + "─" * 70)
        print("5️⃣  检查风险评估器...")
        print("─" * 70)

        try:
            risk_evaluator = RiskEvaluator(
                self.config.risk_w_var,
                self.config.risk_w_ent,
                self.config.risk_w_temp
            )

            self.log_success("RiskEvaluator", "风险评估器创建成功")

            # 测试风险计算
            if self.config.mode in ['occupancy', 'multitask']:
                if self.config.mode == 'multitask':
                    pred, conf = self.test_outputs['occ']
                else:
                    pred, conf = self.test_outputs

                # ✅ 修复：确保 pred 需要梯度
                # 使用 sigmoid 后的结果，并确保需要梯度
                pred_detached = pred.detach()
                occ_prob = torch.sigmoid(pred_detached).requires_grad_(True)

                # 测试训练模式 (需要梯度)
                risks_train = risk_evaluator.compute_all_risks(occ_prob, differentiable=True)

                if risks_train['combined'].device.type != self.device.type:
                    self.log_error("RiskEvaluator", f"训练模式风险在 {risks_train['combined'].device}")
                    return False

                self.log_success("RiskEvaluator", f"训练模式风险: {risks_train['combined'].mean().item():.6f}")

                # 测试梯度
                risk_loss = risks_train['combined'].mean()
                risk_loss.backward()

                if occ_prob.grad is None:
                    self.log_error("RiskEvaluator", "训练模式未计算梯度")
                    return False

                self.log_success("RiskEvaluator", "训练模式梯度计算正确")

                # 测试评估模式 (不需要梯度)
                with torch.no_grad():
                    risks_eval = risk_evaluator.compute_all_risks(torch.sigmoid(pred), differentiable=False)

                if risks_eval['combined'].requires_grad:
                    self.log_warning("RiskEvaluator", "评估模式不应该有梯度")

                self.log_success("RiskEvaluator", "评估模式正常")

                # 测试摘要
                summary = risk_evaluator.get_risk_summary(torch.sigmoid(pred))

                if not all(isinstance(v, float) for v in summary.values()):
                    self.log_warning("RiskEvaluator", "摘要返回值不是 Python float")
                else:
                    self.log_success("RiskEvaluator", "摘要计算正确")

            return True

        except Exception as e:
            self.log_error("RiskEvaluator", f"异常: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def check_visualizer(self):
        """检查可视化器"""
        print("\n" + "─" * 70)
        print("6️⃣  检查可视化器...")
        print("─" * 70)

        try:
            import tempfile
            temp_dir = tempfile.mkdtemp()

            visualizer = ValidationVisualizer(temp_dir, mode=self.config.mode)

            self.log_success("Visualizer", "可视化器创建成功")

            # 测试可视化 (数据需要在 CPU 上)
            history_cpu = self.test_history.cpu()
            targets_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                          for k, v in self.test_targets.items()}

            if self.config.mode == 'multitask':
                outputs_cpu = {
                    'occ': (self.test_outputs['occ'][0].cpu(), self.test_outputs['occ'][1].cpu()),
                    'motion': (self.test_outputs['motion'][0].cpu(), self.test_outputs['motion'][1].cpu())
                }
            elif self.config.mode == 'occupancy':
                outputs_cpu = (self.test_outputs[0].cpu(), self.test_outputs[1].cpu())
            else:
                outputs_cpu = (self.test_outputs[0].cpu(), self.test_outputs[1].cpu())

            paths = visualizer.visualize_batch(
                history_cpu, outputs_cpu, targets_cpu,
                epoch=0, batch_idx=0, max_samples=1
            )

            if paths:
                self.log_success("Visualizer", f"可视化成功: {len(paths)} 个文件")
            else:
                self.log_warning("Visualizer", "未生成可视化文件")

            # 清理临时文件
            import shutil
            shutil.rmtree(temp_dir)

            return True

        except Exception as e:
            self.log_error("Visualizer", f"异常: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def check_memory(self):
        """检查显存使用"""
        print("\n" + "─" * 70)
        print("7️⃣  检查显存使用...")
        print("─" * 70)

        if not torch.cuda.is_available():
            self.log_warning("Memory", "CPU 模式，跳过显存检查")
            return True

        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # 模拟一个完整的训练步骤
            self.model.train()
            self.model.zero_grad()

            # 前向传播
            outputs = self.model(self.test_history, return_confidence=True)

            if self.config.mode == 'multitask':
                criterion = MultiTaskLoss().to(self.device)
                losses = criterion(outputs, self.test_targets)
                loss = losses['total']
            elif self.config.mode == 'occupancy':
                criterion = OccupancyLoss().to(self.device)
                pred, conf = outputs
                loss, _ = criterion(pred, self.test_targets['occ'], conf)
            else:
                criterion = MotionLoss().to(self.device)
                pred, conf = outputs
                loss, _ = criterion(pred, self.test_targets['motion'], conf)

            # 反向传播
            loss.backward()

            # 获取显存使用
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            peak = torch.cuda.max_memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9

            self.log_success("Memory", f"当前使用: {allocated:.2f} GB")
            self.log_success("Memory", f"峰值使用: {peak:.2f} GB")
            self.log_success("Memory", f"总显存: {total:.2f} GB")
            self.log_success("Memory", f"使用率: {peak / total * 100:.1f}%")

            if peak / total > 0.9:
                self.log_warning("Memory", "显存使用率超过 90%，建议减小 batch_size")

            torch.cuda.empty_cache()
            return True

        except Exception as e:
            self.log_error("Memory", f"异常: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def run_all_checks(self):
        """运行所有检查"""
        checks = [
            ("数据集", self.check_dataset),
            ("模型", self.check_model),
            ("损失函数", self.check_loss),
            ("评估指标", self.check_metrics),
            ("风险评估器", self.check_risk_evaluator),
            ("可视化器", self.check_visualizer),
            ("显存", self.check_memory),
        ]

        results = {}
        for name, check_func in checks:
            try:
                results[name] = check_func()
            except KeyboardInterrupt:
                print("\n\n检查被用户中断")
                return False
            except Exception as e:
                print(f"\n❌ 检查 '{name}' 时发生未捕获异常: {e}")
                import traceback
                traceback.print_exc()
                results[name] = False

        # 打印总结
        print("\n" + "=" * 70)
        print("📊 检查结果汇总")
        print("=" * 70)

        for name, passed in results.items():
            status = "✅ 通过" if passed else "❌ 失败"
            print(f"{status} - {name}")

        print("=" * 70)

        if self.errors:
            print(f"\n❌ 发现 {len(self.errors)} 个错误:")
            for error in self.errors:
                print(f"  {error}")

        if self.warnings:
            print(f"\n⚠️  发现 {len(self.warnings)} 个警告:")
            for warning in self.warnings:
                print(f"  {warning}")

        all_passed = all(results.values())

        print("\n" + "=" * 70)
        if all_passed:
            print("🎉 所有检查通过！可以开始训练。")
        else:
            print("⚠️  部分检查失败！请修复后再训练。")
        print("=" * 70 + "\n")

        return all_passed


def parse_args():
    """解析命令行参数"""
    p = argparse.ArgumentParser(description='快速设备检查')

    # 数据相关
    p.add_argument('--data_dir', default=r'D:\model_12.22_fixed\images',
                   help='数据集根目录')
    p.add_argument('--dataset_type', default='sequence', choices=['sequence', 'simple'],
                   help='数据集类型')
    p.add_argument('--history_frames', type=int, default=9,
                   help='历史帧数')
    p.add_argument('--img_size', type=int, nargs=2, default=[640, 640],
                   help='图像尺寸')

    # 模型相关
    p.add_argument('--mode', default='occupancy', choices=['multitask', 'occupancy', 'motion'],
                   help='训练模式')
    p.add_argument('--backbone', default='resnet50',
                   help='编码器backbone')
    p.add_argument('--hidden_dim', type=int, default=128,
                   help='隐藏层维度')
    p.add_argument('--num_modes', type=int, default=5,
                   help='模态数量')
    p.add_argument('--future_steps', type=int, default=1,
                   help='预测未来步数')

    # Risk相关
    p.add_argument('--risk_weight', type=float, default=0.01)
    p.add_argument('--risk_w_var', type=float, default=1.0)
    p.add_argument('--risk_w_ent', type=float, default=0.5)
    p.add_argument('--risk_w_temp', type=float, default=0.3)

    # MultiTask权重
    p.add_argument('--occ_weight', type=float, default=1.0)
    p.add_argument('--motion_weight', type=float, default=1.0)

    # 其他
    p.add_argument('--device', default='cuda',
                   help='训练设备')

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("\n" + "=" * 70)
    print("🚀 开始快速设备检查")
    print("=" * 70)
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
    print("=" * 70)

    checker = DeviceChecker(args)
    success = checker.run_all_checks()

    sys.exit(0 if success else 1)