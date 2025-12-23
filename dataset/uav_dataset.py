import os
import torch
from torch.utils.data import Dataset

class UAVDataset(Dataset):
    def __init__(self, root_dir, history_sec=3, use_occ=True, use_motion=False):
        """
        root_dir: .pt 文件目录
        history_sec: 历史秒数（3 或 5）
        use_occ: 是否使用 occupancy 目标
        use_motion: 是否使用 motion 目标
        """
        self.history_sec = history_sec
        self.use_occ = use_occ
        self.use_motion = use_motion

        self.files = sorted([
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith(".pt")
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        history = data["history"]  # [T_h, 3, H, W]

        targets = {}
        if self.use_occ and "future" in data:
            targets["occ"] = data["future"]  # [H, W] 或 [T_f, H, W]

        if self.use_motion and "motion" in data:
            targets["motion"] = data["motion"]

        return history, targets

if __name__ == "__main__":
    dataset = UAVDataset(
        root_dir=r"D:\project\data\train_3s",
        use_occ=True,
        use_motion=True
    )
    print(f"Dataset size: {len(dataset)}")
    sample_history, sample_targets = dataset[0]
    print(f"History shape: {sample_history.shape}")
    for key, value in sample_targets.items():
        print(f"Target {key} shape: {value.shape}")