"""
UAV 训练数据集
"""

import os
import torch
from torch.utils.data import Dataset


class UAVTrainDataset(Dataset):
    """UAV 训练数据集，支持 occupancy 和 motion 目标"""

    def __init__(self, root_dir, use_occ=True, use_motion=True, transform=None):
        self.root_dir = root_dir
        self.use_occ = use_occ
        self.use_motion = use_motion
        self.transform = transform

        self.files = sorted([
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir) if f.endswith(".pt")
        ])

        if len(self.files) == 0:
            raise ValueError(f"No .pt files found in {root_dir}")
        print(f"[Dataset] Loaded {len(self.files)} samples from {root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        history = data["history"]

        if history.dim() == 3:
            history = history.unsqueeze(1)

        targets = {}
        if self.use_occ and "future" in data:
            occ = data["future"]
            if occ.dim() == 2:
                occ = occ.unsqueeze(0).unsqueeze(0)
            elif occ.dim() == 3:
                occ = occ.unsqueeze(1)
            targets["occ"] = occ

        if self.use_motion and "motion" in data:
            motion = data["motion"]
            if motion.dim() == 2:
                motion = motion.unsqueeze(0).unsqueeze(0)
            elif motion.dim() == 3:
                motion = motion.unsqueeze(1)
            targets["motion"] = motion

        if self.transform:
            history = self.transform(history)

        return history, targets