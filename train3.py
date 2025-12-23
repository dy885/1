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

        # 1. 设备初始化
        if torch.cuda.is_available() and config.device == 'cuda':
            self.device = torch.device('cuda')
            torch.cuda.empty_cache()
            print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print("[WARNING] CUDA not available, using CPU")

        # 2. 输出目录
        self.output_dir = os.path.join(
            config.output_dir, f"{config.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(os.path.join(self.output_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'visualizations'), exist_ok=True)

        # 3. 初始化各组件
        print("-" * 40)
        self._init_data()
        self._init_model()
        self._init_optimizer()

        self.use_amp = config.use_amp and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None

        # 工具类
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

        print("-" * 40)
        print(
            f"初始化完成 | Mode: {config.mode} | Batch: {config.batch_size} | Eff.Batch: {config.batch_size * config.accum_steps}")
        print("-" * 40)

    def _init_data(self):
        use_occ = self.config.mode in ['occupancy', 'multitask']
        use_motion = self.config.mode in ['motion', 'multitask']
        img_size = tuple(self.config.img_size) if self.config.img_size else (640, 640)

        print(f"加载数据集: {self.config.dataset_type} | 历史帧: {self.config.history_frames}")

        try:
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

            val_size = int(len(full_dataset) * self.config.val_ratio)
            train_size = len(full_dataset) - val_size

            generator = torch.Generator().manual_seed(self.config.seed)
            self.train_dataset, self.val_dataset = random_split(
                full_dataset, [train_size, val_size], generator=generator
            )

            pin_memory = torch.cuda.is_available()
            num_workers = 0 if os.name == 'nt' else min(self.config.num_workers, 2)

            self.train_loader = DataLoader(
                self.train_dataset, batch_size=self.config.batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=pin_memory, drop_last=True
            )
            self.val_loader = DataLoader(
                self.val_dataset, batch_size=self.config.batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory
            )
            print(f"✓ 数据集加载完毕 (Train: {train_size}, Val: {val_size})")

        except Exception as e:
            print(f"✗ 数据集初始化失败: {e}")
            raise

    def _init_model(self):
        try:
            self.model = UAVConflictModel(
                mode=self.config.mode,
                hidden_dim=self.config.hidden_dim,
                modes=self.config.num_modes,
                encoder_backbone=self.config.backbone,
                future_steps=self.config.future_steps
            ).to(self.device)

            if self.config.mode == 'multitask':
                self.criterion = MultiTaskLoss(
                    risk_weight=self.config.risk_weight,
                    risk_w_var=self.config.risk_w_var,
                    risk_w_ent=self.config.risk_w_ent,
                    risk_w_temp=self.config.risk_w_temp,
                    occ_weight=self.config.occ_weight,
                    motion_weight=self.config.motion_weight
                ).to(self.device)
            elif self.config.mode == 'occupancy':
                self.criterion = OccupancyLoss(
                    self.config.risk_weight, self.config.risk_w_var,
                    self.config.risk_w_ent, self.config.risk_w_temp
                ).to(self.device)
            else:
                self.criterion = MotionLoss().to(self.device)

            print(f"✓ 模型与损失函数构建完毕 ({self.config.backbone})")

        except Exception as e:
            print(f"✗ 模型初始化失败: {e}")
            raise

    def _init_optimizer(self):
        params = list(self.model.parameters())
        if hasattr(self.criterion, 'parameters'):
            params += list(self.criterion.parameters())

        self.optimizer = AdamW(
            params, lr=self.config.lr, weight_decay=self.config.weight_decay, eps=1e-8
        )

        if self.config.scheduler == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.epochs)
        else:
            self.scheduler = None

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_risk = 0
        batch_count = 0
        total_batches = len(self.train_loader)

        self.optimizer.zero_grad()
        epoch_start = time.time()

        for batch_idx, batch_data in enumerate(self.train_loader):
            history, targets = batch_data
            history = history.to(self.device, non_blocking=True)

            targets_gpu = {}
            for k, v in targets.items():
                if isinstance(v, torch.Tensor):
                    targets_gpu[k] = v.to(self.device, non_blocking=True)
                else:
                    targets_gpu[k] = v

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(history, return_confidence=True)

                if self.config.mode == 'multitask':
                    losses = self.criterion(outputs, targets_gpu)
                    loss = losses['total']
                    occ_pred, _ = outputs['occ']
                    risks = self.risk_evaluator.compute_all_risks(torch.sigmoid(occ_pred), differentiable=True)
                    total_risk += risks['combined'].mean().item()

                elif self.config.mode == 'occupancy':
                    pred, conf = outputs
                    loss, _ = self.criterion(pred, targets_gpu['occ'], conf)
                    risks = self.risk_evaluator.compute_all_risks(torch.sigmoid(pred), differentiable=True)
                    total_risk += risks['combined'].mean().item()

                else:  # motion
                    pred, conf = outputs
                    loss, _ = self.criterion(pred, targets_gpu['motion'], conf)

            loss = loss / self.config.accum_steps

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

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

            if batch_idx % 5 == 0 or batch_idx == total_batches - 1:
                print(
                    f"\rTrain [{epoch}/{self.config.epochs}]: {batch_idx + 1}/{total_batches} | Loss: {total_loss / batch_count:.4f}",
                    end='')

        print()
        avg_loss = total_loss / max(batch_count, 1)
        avg_risk = total_risk / max(batch_count, 1)
        return avg_loss, avg_risk

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        total_risk = 0
        all_metrics = {}
        vis_count = 0

        with torch.no_grad():
            for batch_idx, (history, targets) in enumerate(self.val_loader):
                history = history.to(self.device, non_blocking=True)
                targets_gpu = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                               for k, v in targets.items()}

                outputs = self.model(history, return_confidence=True)

                # --- Loss & Risk Calculation ---
                if self.config.mode == 'multitask':
                    losses = self.criterion(outputs, targets_gpu)
                    loss = losses['total']

                    # Occ Metrics
                    occ_pred, _ = outputs['occ']
                    risks = self.risk_evaluator.compute_all_risks(torch.sigmoid(occ_pred), differentiable=False)
                    total_risk += risks['combined'].mean().item()

                    if 'occ' in targets_gpu:
                        m = self.metrics.compute_occ_metrics(occ_pred, targets_gpu['occ'])
                        for k, v in m.items(): all_metrics[k] = all_metrics.get(k, 0) + v

                    # Motion Metrics (新增：确保多任务模式下也计算运动指标)
                    if 'motion' in targets_gpu:
                        mot_pred, _ = outputs['motion']
                        m = self.metrics.compute_motion_metrics(mot_pred, targets_gpu['motion'])
                        for k, v in m.items(): all_metrics[k] = all_metrics.get(k, 0) + v

                elif self.config.mode == 'occupancy':
                    pred, conf = outputs
                    loss, _ = self.criterion(pred, targets_gpu['occ'], conf)
                    risks = self.risk_evaluator.compute_all_risks(torch.sigmoid(pred), differentiable=False)
                    total_risk += risks['combined'].mean().item()

                    m = self.metrics.compute_occ_metrics(pred, targets_gpu['occ'])
                    for k, v in m.items(): all_metrics[k] = all_metrics.get(k, 0) + v

                else:  # motion
                    pred, conf = outputs
                    loss, _ = self.criterion(pred, targets_gpu['motion'], conf)

                    m = self.metrics.compute_motion_metrics(pred, targets_gpu['motion'])
                    for k, v in m.items(): all_metrics[k] = all_metrics.get(k, 0) + v

                total_loss += loss.item()

                # --- Visualization ---
                if vis_count < self.config.max_vis_batches:
                    history_cpu = history.cpu()
                    targets_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in targets_gpu.items()}

                    if self.config.mode == 'multitask':
                        outputs_cpu = {
                            'occ': (outputs['occ'][0].cpu(), outputs['occ'][1].cpu()),
                            'motion': (outputs['motion'][0].cpu(), outputs['motion'][1].cpu())
                        }
                    else:
                        outputs_cpu = (outputs[0].cpu(), outputs[1].cpu())

                    self.visualizer.visualize_batch(
                        history_cpu, outputs_cpu, targets_cpu,
                        epoch=epoch, batch_idx=batch_idx,
                        max_samples=self.config.max_vis_samples
                    )
                    vis_count += 1

                print(f"\rValidating: {batch_idx + 1}/{len(self.val_loader)}", end='')

        print()
        avg_loss = total_loss / len(self.val_loader)
        avg_risk = total_risk / len(self.val_loader)

        # 计算所有指标的平均值
        avg_metrics = {k: v / len(self.val_loader) for k, v in all_metrics.items()}

        return avg_loss, avg_metrics, avg_risk

    def save_checkpoint(self, epoch, is_best=False):
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

    def train(self):
        print(f"\n开始训练 (Epochs: {self.config.epochs})")
        print("=" * 60)

        for epoch in range(1, self.config.epochs + 1):
            # 训练
            train_loss, train_risk = self.train_epoch(epoch)

            # 验证
            val_loss, val_metrics, val_risk = self.validate(epoch)

            # 学习率
            if self.scheduler: self.scheduler.step()
            lr = self.optimizer.param_groups[0]['lr']

            # 保存最佳模型
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            if epoch % self.config.save_interval == 0 or is_best:
                self.save_checkpoint(epoch, is_best)

            # 动态生成指标字符串，自动包含所有返回的指标
            metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])

            print(f"Summary Ep {epoch} | LR: {lr:.2e} | "
                  f"TrLoss: {train_loss:.4f} | ValLoss: {val_loss:.4f} {('(*)' if is_best else '')}")
            print(f"   > Metrics: {metrics_str}")
            if train_risk > 0 or val_risk > 0:
                print(f"   > Risk   : Train={train_risk:.4f} | Val={val_risk:.4f}")
            print("-" * 60)

        print(f"\n训练结束. Best Loss: {self.best_val_loss:.4f}")
        print(f"Saved to: {self.output_dir}")


def parse_args():
    p = argparse.ArgumentParser(description='UAV Conflict Model Trainer')
    p.add_argument('--data_dir', default=r'D:\model_12.22_fixed\images', help='数据集根目录')
    p.add_argument('--dataset_type', default='sequence', choices=['sequence', 'simple'])
    p.add_argument('--history_frames', type=int, default=9)
    p.add_argument('--output_dir', default='outputs')
    p.add_argument('--val_ratio', type=float, default=0.2)
    p.add_argument('--img_size', type=int, nargs=2, default=[640, 640])
    p.add_argument('--mode', default='occupancy', choices=['multitask', 'occupancy', 'motion'])
    p.add_argument('--backbone', default='resnet50')
    p.add_argument('--hidden_dim', type=int, default=128)
    p.add_argument('--num_modes', type=int, default=5)
    p.add_argument('--future_steps', type=int, default=1)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--accum_steps', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--scheduler', default='cosine', choices=['cosine', 'none'])
    p.add_argument('--use_amp', action='store_true', default=False)
    p.add_argument('--risk_weight', type=float, default=0.01)
    p.add_argument('--risk_w_var', type=float, default=1.0)
    p.add_argument('--risk_w_ent', type=float, default=0.5)
    p.add_argument('--risk_w_temp', type=float, default=0.3)
    p.add_argument('--occ_weight', type=float, default=1.0)
    p.add_argument('--motion_weight', type=float, default=1.0)
    p.add_argument('--max_vis_batches', type=int, default=3)
    p.add_argument('--max_vis_samples', type=int, default=3)
    p.add_argument('--num_workers', type=int, default=1)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save_interval', type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    try:
        trainer = Trainer(args)
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nUser Interrupted.")
    except Exception as e:
        print(f"\n[Error] {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()