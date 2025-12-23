import os
import argparse
import json
import time
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

from sub_model.conflict_model import UAVConflictModel
from trainer.losses import OccupancyLoss, MotionLoss, MultiTaskLoss
from trainer.metrics import Metrics
from trainer.risk_evaluator import RiskEvaluator
from image_dataset import UAVImageDataset, UAVSimpleImageDataset
from validation_visualizer import ValidationVisualizer


class Trainer:
    def __init__(self, config):
        self.config = config

        # ✅ 设备初始化 - 统一管理
        if torch.cuda.is_available() and config.device == 'cuda':
            self.device = torch.device('cuda')
            torch.cuda.empty_cache()
            print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print("[WARNING] CUDA not available, using CPU")

        # 创建输出目录
        self.output_dir = os.path.join(
            config.output_dir, f"{config.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(os.path.join(self.output_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'visualizations'), exist_ok=True)

        # 初始化组件
        print("=" * 60)
        print("步骤 1/3: 正在初始化数据集...")
        print("=" * 60)
        self._init_data()

        print("\n" + "=" * 60)
        print("步骤 2/3: 正在初始化模型...")
        print("=" * 60)
        self._init_model()

        print("\n" + "=" * 60)
        print("步骤 3/3: 正在初始化优化器...")
        print("=" * 60)
        self._init_optimizer()

        self.use_amp = config.use_amp and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None

        # ✅ 确保 metrics 和 risk_evaluator 知道设备
        self.metrics = Metrics(config.risk_w_var, config.risk_w_ent, config.risk_w_temp)
        self.risk_evaluator = RiskEvaluator(config.risk_w_var, config.risk_w_ent, config.risk_w_temp)

        self.visualizer = ValidationVisualizer(
            os.path.join(self.output_dir, 'visualizations'),
            mode=config.mode
        )

        self.best_val_loss = float('inf')

        # 保存配置
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
            json.dump(vars(config), f, indent=2)

        print(f"\n{'=' * 60}")
        print("初始化完成！配置信息：")
        print(f"{'=' * 60}")
        print(f"Training mode: {config.mode}")
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"AMP: {'Enabled' if self.use_amp else 'Disabled'}")
        print(f"Batch size: {config.batch_size}")
        print(f"Accumulation steps: {config.accum_steps}")
        print(f"Effective batch size: {config.batch_size * config.accum_steps}")
        print(f"{'=' * 60}\n")

    def _init_data(self):
        """初始化数据集"""
        use_occ = self.config.mode in ['occupancy', 'multitask']
        use_motion = self.config.mode in ['motion', 'multitask']
        img_size = tuple(self.config.img_size) if self.config.img_size else (640, 640)

        print(f"配置信息:")
        print(f"  - 数据目录: {self.config.data_dir}")
        print(f"  - 数据集类型: {self.config.dataset_type}")
        print(f"  - 图像尺寸: {img_size}")
        print(f"  - 历史帧数: {self.config.history_frames}")
        print(f"  - 使用occupancy: {use_occ}")
        print(f"  - 使用motion: {use_motion}")

        try:
            print("\n正在加载数据集...")
            if self.config.dataset_type == 'sequence':
                full_dataset = UAVImageDataset(
                    root_dir=self.config.data_dir,
                    history_frames=self.config.history_frames,
                    use_occ=use_occ,
                    use_motion=use_motion,
                    img_size=img_size
                )
            else:
                full_dataset = UAVSimpleImageDataset(
                    root_dir=self.config.data_dir,
                    img_size=img_size,
                    use_occ=use_occ,
                    use_motion=use_motion
                )

            print(f"✓ 数据集创建成功，总样本数: {len(full_dataset)}")

            if len(full_dataset) == 0:
                raise ValueError("数据集为空！请检查数据目录和文件格式。")

        except Exception as e:
            print(f"✗ 数据集初始化失败: {e}")
            raise

        val_size = int(len(full_dataset) * self.config.val_ratio)
        train_size = len(full_dataset) - val_size

        print(f"\n正在划分数据集...")
        generator = torch.Generator().manual_seed(self.config.seed)
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size], generator=generator
        )

        # DataLoader设置
        num_workers = 0 if os.name == 'nt' else min(self.config.num_workers, 2)

        print(f"\n正在创建DataLoader...")
        # ✅ 确保 pin_memory 正确设置
        pin_memory = torch.cuda.is_available()

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            persistent_workers=False
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=False
        )

        print(f"✓ DataLoader创建成功")
        print(f"  - 训练集: {train_size} 样本, {len(self.train_loader)} batches")
        print(f"  - 验证集: {val_size} 样本, {len(self.val_loader)} batches")
        print(f"  - Num workers: {num_workers}")
        print(f"  - Pin memory: {pin_memory}")

    def _init_model(self):
        """初始化模型和损失函数"""
        try:
            print("正在创建模型...")
            self.model = UAVConflictModel(
                mode=self.config.mode,
                hidden_dim=self.config.hidden_dim,
                modes=self.config.num_modes,
                encoder_backbone=self.config.backbone,
                future_steps=self.config.future_steps
            ).to(self.device)  # ✅ 确保模型在 GPU 上

            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"✓ 模型创建成功")
            print(f"  - Backbone: {self.config.backbone}")
            print(f"  - Hidden dim: {self.config.hidden_dim}")
            print(f"  - 总参数量: {total_params:,}")
            print(f"  - 可训练参数: {trainable_params:,}")

        except Exception as e:
            print(f"✗ 模型初始化失败: {e}")
            raise

        try:
            print("\n正在创建损失函数...")
            if self.config.mode == 'multitask':
                self.criterion = MultiTaskLoss(
                    risk_weight=self.config.risk_weight,
                    risk_w_var=self.config.risk_w_var,
                    risk_w_ent=self.config.risk_w_ent,
                    risk_w_temp=self.config.risk_w_temp,
                    occ_weight=self.config.occ_weight,
                    motion_weight=self.config.motion_weight
                ).to(self.device)  # ✅ 确保 criterion 在 GPU 上
            elif self.config.mode == 'occupancy':
                self.criterion = OccupancyLoss(
                    self.config.risk_weight,
                    self.config.risk_w_var,
                    self.config.risk_w_ent,
                    self.config.risk_w_temp
                ).to(self.device)  # ✅ 确保 criterion 在 GPU 上
            else:
                self.criterion = MotionLoss().to(self.device)  # ✅ 确保 criterion 在 GPU 上

            print(f"✓ 损失函数创建成功 ({self.config.mode} mode)")

        except Exception as e:
            print(f"✗ 损失函数初始化失败: {e}")
            raise

    def _init_optimizer(self):
        """初始化优化器和学习率调度器"""
        params = list(self.model.parameters())
        if hasattr(self.criterion, 'parameters'):
            params += list(self.criterion.parameters())

        print("正在创建优化器...")
        self.optimizer = AdamW(
            params,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            eps=1e-8
        )

        if self.config.scheduler == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.epochs)
            print(f"✓ 优化器创建成功 (AdamW + CosineAnnealing)")
        else:
            self.scheduler = None
            print(f"✓ 优化器创建成功 (AdamW)")

        print(f"  - Learning rate: {self.config.lr}")
        print(f"  - Weight decay: {self.config.weight_decay}")

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_risk = 0
        batch_count = 0

        total_batches = len(self.train_loader)
        self.optimizer.zero_grad()

        epoch_start = time.time()

        print(f"训练进度: ", end='', flush=True)

        for batch_idx, batch_data in enumerate(self.train_loader):
            try:
                # ✅ 解包数据并移到 GPU
                history, targets = batch_data

                # ✅ 移动到设备 (non_blocking 加速)
                history = history.to(self.device, non_blocking=True)

                # ✅ 确保所有 targets 都在 GPU 上
                targets_gpu = {}
                for k, v in targets.items():
                    if isinstance(v, torch.Tensor):
                        targets_gpu[k] = v.to(self.device, non_blocking=True)
                    else:
                        targets_gpu[k] = v

                # 前向传播
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    outputs = self.model(history, return_confidence=True)

                    if self.config.mode == 'multitask':
                        losses = self.criterion(outputs, targets_gpu)
                        loss = losses['total']

                        occ_pred, _ = outputs['occ']
                        risks = self.risk_evaluator.compute_all_risks(
                            torch.sigmoid(occ_pred),
                            differentiable=True
                        )
                        total_risk += risks['combined'].mean().item()

                    elif self.config.mode == 'occupancy':
                        pred, conf = outputs
                        loss, details = self.criterion(pred, targets_gpu['occ'], conf)

                        risks = self.risk_evaluator.compute_all_risks(
                            torch.sigmoid(pred),
                            differentiable=True
                        )
                        total_risk += risks['combined'].mean().item()

                    else:  # motion
                        pred, conf = outputs
                        loss, details = self.criterion(pred, targets_gpu['motion'], conf)

                # 反向传播
                loss = loss / self.config.accum_steps

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # 优化器步进
                if (batch_idx + 1) % self.config.accum_steps == 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.optimizer.zero_grad()

                total_loss += loss.item() * self.config.accum_steps
                batch_count += 1

                # 更新进度条
                progress = (batch_idx + 1) / total_batches
                bar_length = 40
                filled = int(bar_length * progress)
                bar = '█' * filled + '░' * (bar_length - filled)

                avg_loss = total_loss / batch_count
                elapsed = time.time() - epoch_start
                eta = elapsed / progress - elapsed if progress > 0 else 0

                print(f"\r训练进度: [{bar}] {progress * 100:.1f}% | Loss: {avg_loss:.4f} | ETA: {eta / 60:.1f}min",
                      end='', flush=True)

            except Exception as e:
                print(f"\n✗ [ERROR] Batch {batch_idx} failed: {e}")
                import traceback
                traceback.print_exc()
                continue

        print()  # 换行
        total_time = time.time() - epoch_start
        avg_loss = total_loss / max(batch_count, 1)
        avg_risk = total_risk / max(batch_count, 1)

        print(f"✓ 训练完成 | 耗时: {total_time / 60:.1f}min | 平均Loss: {avg_loss:.4f} | Risk: {avg_risk:.6f}")

        return avg_loss, avg_risk

    def validate(self, epoch):
        """验证 - GPU 优化版本"""
        self.model.eval()
        total_loss = 0
        total_risk = 0
        all_metrics = {}

        vis_count = 0
        total_batches = len(self.val_loader)

        val_start = time.time()

        print(f"验证进度: ", end='', flush=True)

        with torch.no_grad():
            for batch_idx, (history, targets) in enumerate(self.val_loader):
                try:
                    # ✅ 移动到 GPU
                    history = history.to(self.device, non_blocking=True)

                    # ✅ 确保所有 targets 都在 GPU 上
                    targets_gpu = {}
                    for k, v in targets.items():
                        if isinstance(v, torch.Tensor):
                            targets_gpu[k] = v.to(self.device, non_blocking=True)
                        else:
                            targets_gpu[k] = v

                    # 前向传播
                    outputs = self.model(history, return_confidence=True)

                    if self.config.mode == 'multitask':
                        losses = self.criterion(outputs, targets_gpu)
                        loss = losses['total']

                        occ_pred, _ = outputs['occ']
                        risks = self.risk_evaluator.compute_all_risks(
                            torch.sigmoid(occ_pred),
                            differentiable=False
                        )
                        total_risk += risks['combined'].mean().item()

                        if 'occ' in targets_gpu:
                            metrics = self.metrics.compute_occ_metrics(occ_pred, targets_gpu['occ'])
                            for k, v in metrics.items():
                                all_metrics[k] = all_metrics.get(k, 0) + v

                    elif self.config.mode == 'occupancy':
                        pred, conf = outputs
                        loss, _ = self.criterion(pred, targets_gpu['occ'], conf)

                        risks = self.risk_evaluator.compute_all_risks(
                            torch.sigmoid(pred),
                            differentiable=False
                        )
                        total_risk += risks['combined'].mean().item()

                        metrics = self.metrics.compute_occ_metrics(pred, targets_gpu['occ'])
                        for k, v in metrics.items():
                            all_metrics[k] = all_metrics.get(k, 0) + v

                    else:  # motion
                        pred, conf = outputs
                        loss, _ = self.criterion(pred, targets_gpu['motion'], conf)

                        metrics = self.metrics.compute_motion_metrics(pred, targets_gpu['motion'])
                        for k, v in metrics.items():
                            all_metrics[k] = all_metrics.get(k, 0) + v

                    total_loss += loss.item()

                    # ✅ 可视化时将数据移回 CPU
                    if vis_count < self.config.max_vis_batches:
                        history_cpu = history.cpu()
                        targets_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                                       for k, v in targets_gpu.items()}

                        if self.config.mode == 'multitask':
                            outputs_cpu = {
                                'occ': (outputs['occ'][0].cpu(), outputs['occ'][1].cpu()),
                                'motion': (outputs['motion'][0].cpu(), outputs['motion'][1].cpu())
                            }
                        elif self.config.mode == 'occupancy':
                            outputs_cpu = (outputs[0].cpu(), outputs[1].cpu())
                        else:
                            outputs_cpu = (outputs[0].cpu(), outputs[1].cpu())

                        self.visualizer.visualize_batch(
                            history_cpu, outputs_cpu, targets_cpu,
                            epoch=epoch, batch_idx=batch_idx,
                            max_samples=self.config.max_vis_samples
                        )
                        vis_count += 1

                    # 进度条更新
                    progress = (batch_idx + 1) / total_batches
                    bar_length = 40
                    filled = int(bar_length * progress)
                    bar = '█' * filled + '░' * (bar_length - filled)

                    print(f"\r验证进度: [{bar}] {progress * 100:.1f}%", end='', flush=True)

                except Exception as e:
                    print(f"\n⚠ [WARNING] Validation batch {batch_idx} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        print()  # 换行
        val_time = time.time() - val_start
        avg_loss = total_loss / len(self.val_loader)
        avg_metrics = {k: v / len(self.val_loader) for k, v in all_metrics.items()}
        avg_risk = total_risk / len(self.val_loader)

        print(f"✓ 验证完成 | 耗时: {val_time:.1f}s | Loss: {avg_loss:.4f} | Risk: {avg_risk:.6f}")

        return avg_loss, avg_metrics, avg_risk

    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        path = os.path.join(self.output_dir, 'checkpoints', f'epoch_{epoch}.pt')
        torch.save(ckpt, path)
        if is_best:
            best_path = os.path.join(self.output_dir, 'checkpoints', 'best.pt')
            torch.save(ckpt, best_path)
            print(f"  ✓ 最佳模型已保存: {best_path}")

    def train(self):
        """主训练循环"""
        print(f"\n{'=' * 60}")
        print(f"开始训练")
        print(f"{'=' * 60}")
        print(f"总Epochs: {self.config.epochs}")
        print(f"输出目录: {self.output_dir}")
        print(f"{'=' * 60}\n")

        for epoch in range(1, self.config.epochs + 1):
            print(f"\n{'━' * 60}")
            print(f"EPOCH {epoch}/{self.config.epochs}")
            print(f"{'━' * 60}")

            # 训练
            train_loss, train_risk = self.train_epoch(epoch)

            # 验证
            val_loss, val_metrics, val_risk = self.validate(epoch)

            # 学习率调度
            if self.scheduler:
                self.scheduler.step()
            lr = self.optimizer.param_groups[0]['lr']

            # 结果汇总
            print(f"{'─' * 60}")
            print(f"Epoch {epoch} 汇总 | LR: {lr:.6f}")
            print(f"  训练: Loss={train_loss:.4f}, Risk={train_risk:.6f}")
            print(f"  验证: Loss={val_loss:.4f}, Risk={val_risk:.6f}, IoU={val_metrics.get('iou', 0):.4f}")

            # 保存最佳模型
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                print(f"  🎉 新的最佳模型！")

            # 定期保存
            if epoch % self.config.save_interval == 0 or is_best:
                self.save_checkpoint(epoch, is_best)

        print(f"\n{'┏' * 60}")
        print(f"训练完成！")
        print(f"{'┗' * 60}")
        print(f"最佳验证损失: {self.best_val_loss:.4f}")
        print(f"模型保存位置: {self.output_dir}")
        print(f"{'─' * 60}\n")


def parse_args():
    p = argparse.ArgumentParser(description='UAV Conflict Model Trainer')

    # 数据相关
    p.add_argument('--data_dir', default=r'D:\model_12.22_fixed\images',
                   help='数据集根目录')
    p.add_argument('--dataset_type', default='sequence', choices=['sequence', 'simple'],
                   help='数据集类型')
    p.add_argument('--history_frames', type=int, default=9,
                   help='历史帧数')
    p.add_argument('--output_dir', default='outputs',
                   help='输出目录')
    p.add_argument('--val_ratio', type=float, default=0.2,
                   help='验证集比例')
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

    # 训练相关
    p.add_argument('--epochs', type=int, default=100,
                   help='训练轮数')
    p.add_argument('--batch_size', type=int, default=2,
                   help='批次大小')
    p.add_argument('--accum_steps', type=int, default=4,
                   help='梯度累积步数')
    p.add_argument('--lr', type=float, default=1e-4,
                   help='学习率')
    p.add_argument('--weight_decay', type=float, default=1e-4,
                   help='权重衰减')
    p.add_argument('--grad_clip', type=float, default=1.0,
                   help='梯度裁剪')
    p.add_argument('--scheduler', default='cosine', choices=['cosine', 'none'],
                   help='学习率调度器')
    p.add_argument('--use_amp', action='store_true', default=False,
                   help='使用混合精度训练')

    # Risk相关
    p.add_argument('--risk_weight', type=float, default=0.01)
    p.add_argument('--risk_w_var', type=float, default=1.0)
    p.add_argument('--risk_w_ent', type=float, default=0.5)
    p.add_argument('--risk_w_temp', type=float, default=0.3)

    # MultiTask权重
    p.add_argument('--occ_weight', type=float, default=1.0)
    p.add_argument('--motion_weight', type=float, default=1.0)

    # 可视化相关
    p.add_argument('--max_vis_batches', type=int, default=3)
    p.add_argument('--max_vis_samples', type=int, default=3)

    # 其他
    p.add_argument('--num_workers', type=int, default=1,
                   help='DataLoader工作进程数')
    p.add_argument('--device', default='cuda',
                   help='训练设备')
    p.add_argument('--seed', type=int, default=42,
                   help='随机种子')
    p.add_argument('--save_interval', type=int, default=10,
                   help='保存间隔')

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("=" * 60)
    print("UAV Conflict Model Trainer - GPU Optimized")
    print("=" * 60)
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("=" * 60)

    try:
        trainer = Trainer(args)
        trainer.train()
    except KeyboardInterrupt:
        print("\n\n训练被用户中断！")
    except Exception as e:
        print(f"\n{'=' * 60}")
        print("✗ 训练失败！")
        print(f"{'=' * 60}")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {e}")
        print(f"\n详细堆栈:")
        import traceback

        traceback.print_exc()
        print(f"\n{'=' * 60}")
        print("调试建议:")
        print("1. 检查上面的详细错误信息")
        print("2. 确认数据路径和格式正确")
        print("3. 检查GPU显存是否充足")
        print("4. 尝试减小batch_size")
        print(f"{'=' * 60}\n")