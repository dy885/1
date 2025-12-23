"""
验证可视化模块 - 生成验证结果图片
✅ 修复：正确处理需要梯度的 tensor
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torchvision.transforms.functional as TF
from PIL import Image


class ValidationVisualizer:
    """验证结果可视化器"""

    def __init__(self, output_dir, mode='multitask'):
        self.output_dir = output_dir
        self.mode = mode
        os.makedirs(output_dir, exist_ok=True)

    def denormalize(self, tensor):
        """反归一化图片"""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return tensor * std + mean

    def tensor_to_image(self, tensor):
        """
        将tensor转换为numpy图片
        ✅ 修复：正确处理需要梯度的 tensor
        """
        # ✅ 先 detach 再转换
        if tensor.requires_grad:
            tensor = tensor.detach()

        if tensor.dim() == 4:  # [B, C, H, W]
            tensor = tensor[0]  # 取第一个batch
        if tensor.dim() == 3:  # [C, H, W]
            if tensor.shape[0] == 3:
                tensor = self.denormalize(tensor)
            tensor = tensor.permute(1, 2, 0)  # [H, W, C]

        img = tensor.cpu().numpy()
        img = np.clip(img, 0, 1)
        return img

    def to_numpy(self, tensor):
        """
        安全地将 tensor 转换为 numpy
        ✅ 修复：处理梯度和设备
        """
        if isinstance(tensor, torch.Tensor):
            # 先 detach，再移到 CPU，再转 numpy
            return tensor.detach().cpu().numpy()
        return np.array(tensor)

    def visualize_occupancy(self, history, pred, target, confidence,
                           epoch, batch_idx, sample_idx=0):
        """
        可视化occupancy预测结果

        Args:
            history: [B, T, 3, H, W] 历史帧
            pred: [B, M, T_f, H, W] 预测occupancy
            target: [B, 1, H, W] 真实occupancy
            confidence: [B, M] 模态置信度
            epoch: 当前epoch
            batch_idx: batch索引
            sample_idx: 样本索引
        """
        fig = plt.figure(figsize=(20, 8))
        gs = GridSpec(2, 5, figure=fig, hspace=0.3, wspace=0.3)

        # 取第一个样本
        history_sample = history[sample_idx]  # [T, 3, H, W]
        pred_sample = pred[sample_idx]  # [M, T_f, H, W]
        target_sample = target[sample_idx]  # [1, H, W]
        conf_sample = confidence[sample_idx]  # [M]

        # ✅ 转换为概率（处理梯度）
        if pred_sample.requires_grad:
            pred_prob = torch.sigmoid(pred_sample.detach())
        else:
            pred_prob = torch.sigmoid(pred_sample)

        # 第一行: 显示历史帧
        num_history = min(5, history_sample.shape[0])
        for i in range(num_history):
            ax = fig.add_subplot(gs[0, i])
            img = self.tensor_to_image(history_sample[i])
            ax.imshow(img)
            ax.set_title(f'History t-{num_history-i}')
            ax.axis('off')

        # 第二行: 显示预测和目标
        # 1. 真实目标
        ax = fig.add_subplot(gs[1, 0])
        target_img = self.to_numpy(target_sample[0])
        ax.imshow(target_img, cmap='hot', vmin=0, vmax=1)
        ax.set_title('Ground Truth')
        ax.axis('off')

        # 2-4. 显示前3个模态的预测
        num_modes = min(3, pred_prob.shape[0])
        for m in range(num_modes):
            ax = fig.add_subplot(gs[1, m + 1])
            pred_img = self.to_numpy(pred_prob[m, 0])  # ✅ 使用 to_numpy
            ax.imshow(pred_img, cmap='hot', vmin=0, vmax=1)
            conf_val = self.to_numpy(conf_sample[m]).item()  # ✅ 使用 to_numpy
            ax.set_title(f'Mode {m+1} (conf={conf_val:.3f})')
            ax.axis('off')

        # 5. 加权平均预测
        ax = fig.add_subplot(gs[1, 4])
        # ✅ 使用置信度加权，处理梯度
        conf_sample_np = self.to_numpy(conf_sample)
        pred_prob_np = self.to_numpy(pred_prob[:, 0])  # [M, H, W]

        # 加权平均
        weighted_pred = np.sum(pred_prob_np * conf_sample_np.reshape(-1, 1, 1), axis=0)
        ax.imshow(weighted_pred, cmap='hot', vmin=0, vmax=1)
        ax.set_title('Weighted Average')
        ax.axis('off')

        plt.suptitle(f'Epoch {epoch} - Batch {batch_idx} - Sample {sample_idx}',
                    fontsize=16, y=0.98)

        # 保存
        save_path = os.path.join(self.output_dir,
                                f'epoch_{epoch:03d}_batch_{batch_idx:03d}_sample_{sample_idx}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        return save_path

    def visualize_motion(self, history, pred, target, confidence,
                        epoch, batch_idx, sample_idx=0):
        """
        可视化motion预测结果

        Args:
            history: [B, T, 3, H, W]
            pred: [B, M, T_f, 2] 预测轨迹
            target: [B, T_f, 2] 真实轨迹
            confidence: [B, M]
            epoch: epoch
            batch_idx: batch索引
            sample_idx: 样本索引
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # 取第一个样本 - ✅ 使用 to_numpy
        history_sample = history[sample_idx]
        pred_sample = self.to_numpy(pred[sample_idx])  # [M, T_f, 2]
        target_sample = self.to_numpy(target[sample_idx])  # [T_f, 2]
        conf_sample = self.to_numpy(confidence[sample_idx])  # [M]

        # 左图: 显示最后一帧历史 + 轨迹
        ax = axes[0]
        last_frame = self.tensor_to_image(history_sample[-1])
        ax.imshow(last_frame)

        # 绘制真实轨迹
        target_x = target_sample[:, 0] * last_frame.shape[1]
        target_y = target_sample[:, 1] * last_frame.shape[0]
        ax.plot(target_x, target_y, 'g-o', linewidth=3, markersize=8,
               label='Ground Truth', alpha=0.8)

        # 绘制预测轨迹
        colors = plt.cm.rainbow(np.linspace(0, 1, pred_sample.shape[0]))
        for m in range(pred_sample.shape[0]):
            pred_x = pred_sample[m, :, 0] * last_frame.shape[1]
            pred_y = pred_sample[m, :, 1] * last_frame.shape[0]
            ax.plot(pred_x, pred_y, '-o', color=colors[m], linewidth=2,
                   markersize=6, label=f'Mode {m+1} (conf={conf_sample[m]:.2f})',
                   alpha=0.7)

        ax.set_title('Trajectory Prediction on Last Frame')
        ax.legend(loc='upper right')
        ax.axis('off')

        # 右图: 归一化坐标系中的轨迹
        ax = axes[1]
        ax.plot(target_sample[:, 0], target_sample[:, 1], 'g-o',
               linewidth=3, markersize=8, label='Ground Truth')

        for m in range(pred_sample.shape[0]):
            ax.plot(pred_sample[m, :, 0], pred_sample[m, :, 1], '-o',
                   color=colors[m], linewidth=2, markersize=6,
                   label=f'Mode {m+1} (conf={conf_sample[m]:.2f})')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('X (normalized)')
        ax.set_ylabel('Y (normalized)')
        ax.set_title('Trajectory in Normalized Coordinates')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        plt.suptitle(f'Epoch {epoch} - Batch {batch_idx} - Sample {sample_idx}',
                    fontsize=16)

        save_path = os.path.join(self.output_dir,
                                f'motion_epoch_{epoch:03d}_batch_{batch_idx:03d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        return save_path

    def visualize_multitask(self, history, outputs, targets, epoch, batch_idx):
        """可视化多任务结果"""
        saved_paths = []

        # Occupancy
        if 'occ' in outputs and 'occ' in targets:
            pred_occ, conf_occ = outputs['occ']
            target_occ = targets['occ']
            path = self.visualize_occupancy(
                history, pred_occ, target_occ, conf_occ,
                epoch, batch_idx
            )
            saved_paths.append(path)

        # Motion
        if 'motion' in outputs and 'motion' in targets:
            pred_motion, conf_motion = outputs['motion']
            target_motion = targets['motion']
            path = self.visualize_motion(
                history, pred_motion, target_motion, conf_motion,
                epoch, batch_idx
            )
            saved_paths.append(path)

        return saved_paths

    def visualize_batch(self, history, outputs, targets, epoch, batch_idx,
                       max_samples=3):
        """
        可视化一个batch的结果

        Args:
            max_samples: 每个batch最多可视化的样本数
        """
        try:
            if self.mode == 'multitask':
                return self.visualize_multitask(history, outputs, targets, epoch, batch_idx)
            elif self.mode == 'occupancy':
                pred, conf = outputs
                target = targets['occ']

                saved_paths = []
                num_samples = min(max_samples, history.shape[0])
                for i in range(num_samples):
                    path = self.visualize_occupancy(
                        history, pred, target, conf, epoch, batch_idx, sample_idx=i
                    )
                    saved_paths.append(path)
                return saved_paths
            elif self.mode == 'motion':
                pred, conf = outputs
                target = targets['motion']
                path = self.visualize_motion(
                    history, pred, target, conf, epoch, batch_idx
                )
                return [path]

            return []
        except Exception as e:
            print(f"[WARNING] Visualization failed: {e}")
            import traceback
            traceback.print_exc()
            return []

    def create_summary_grid(self, image_paths, epoch, grid_size=(3, 3)):
        """创建总结网格图"""
        if len(image_paths) == 0:
            return None

        rows, cols = grid_size
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6))
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for idx, (ax, img_path) in enumerate(zip(axes, image_paths)):
            if os.path.exists(img_path):
                img = Image.open(img_path)
                ax.imshow(img)
                ax.set_title(f'Sample {idx + 1}')
            ax.axis('off')

        # 隐藏多余的子图
        for idx in range(len(image_paths), len(axes)):
            axes[idx].axis('off')

        plt.suptitle(f'Validation Results - Epoch {epoch}', fontsize=20)

        summary_path = os.path.join(self.output_dir, f'summary_epoch_{epoch:03d}.png')
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()

        return summary_path


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("Testing ValidationVisualizer")
    print("=" * 60)

    # 创建虚拟数据
    B, T, H, W = 2, 9, 640, 640
    M = 3

    history = torch.randn(B, T, 3, H, W)
    pred_occ = torch.randn(B, M, 1, H, W, requires_grad=True)  # ✅ 测试需要梯度的情况
    target_occ = torch.randn(B, 1, H, W)
    confidence = torch.softmax(torch.randn(B, M), dim=1)

    # 测试可视化
    visualizer = ValidationVisualizer('test_vis', mode='occupancy')

    try:
        path = visualizer.visualize_occupancy(
            history, pred_occ, target_occ, confidence,
            epoch=1, batch_idx=0
        )
        print(f"✅ Saved visualization to: {path}")
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()