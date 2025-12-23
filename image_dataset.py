"""

数据结构:
sequences/
  seq_0000/
    history/
      frame_00.jpg
      ...
    future/
      frame_00.jpg
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


# ==========================================
# 你的 Occupancy Proxy 生成函数
# ==========================================
def generate_occupancy_map(img, img_size=(640, 640)):
    """
    将单张图像转换为 occupancy map
    输入:
        img: PIL.Image 或 Tensor [C,H,W]
        img_size: 输出 occupancy map 大小
    输出:
        Tensor [1,H,W]
    """
    # 如果是 Tensor，先转 PIL
    if isinstance(img, torch.Tensor):
        img = transforms.ToPILImage()(img)

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.Grayscale(),  # 单通道
        transforms.ToTensor()
    ])
    occ = transform(img)
    return occ


# ==========================================
# 你的 Motion Proxy 生成函数
# ==========================================
def generate_motion_map(prev_frame, next_frame, img_size=(640, 640)):
    """
    生成单张 motion map
    输入:
        prev_frame, next_frame: PIL.Image 或 Tensor [C,H,W]
        img_size: 输出 motion map 大小
    输出:
        Tensor [1,H,W]
    """
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.Grayscale(),  # 单通道
        transforms.ToTensor()
    ])

    if isinstance(prev_frame, torch.Tensor):
        prev_frame = transforms.ToPILImage()(prev_frame)
    if isinstance(next_frame, torch.Tensor):
        next_frame = transforms.ToPILImage()(next_frame)

    prev = transform(prev_frame)
    next_ = transform(next_frame)
    motion = torch.abs(next_ - prev)
    return motion  # [1,H,W]


# ==========================================
# Dataset 类
# ==========================================
class UAVImageDataset(Dataset):
    """
    基于图片的UAV数据集 - 实时计算 occupancy 和 motion proxy
    使用你提供的生成方法
    """

    def __init__(self, root_dir, history_frames=9, use_occ=True, use_motion=False,
                 img_size=(640, 640)):
        """
        Args:
            root_dir: 数据根目录 (包含sequences文件夹)
            history_frames: 历史帧数
            use_occ: 是否使用occupancy
            use_motion: 是否使用motion
            img_size: 图片尺寸
        """
        self.root_dir = root_dir
        self.history_frames = history_frames
        self.use_occ = use_occ
        self.use_motion = use_motion
        self.img_size = img_size

        # 查找所有序列
        sequences_dir = os.path.join(root_dir, 'sequences')
        self.sequences = sorted([
            os.path.join(sequences_dir, d)
            for d in os.listdir(sequences_dir)
            if os.path.isdir(os.path.join(sequences_dir, d))
        ])

        if len(self.sequences) == 0:
            raise ValueError(f"No sequences found in {sequences_dir}")

        # History transform (标准ImageNet归一化)
        self.history_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        print(f"[Dataset] Loaded {len(self.sequences)} sequences from {root_dir}")
        print(f"[Dataset] Use Occupancy: {use_occ}, Use Motion: {use_motion}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_dir = self.sequences[idx]
        history_dir = os.path.join(seq_dir, 'history')
        future_dir = os.path.join(seq_dir, 'future')

        # ==========================================
        # 1. 加载历史帧
        # ==========================================
        history_frames = []
        history_files = sorted([
            f for f in os.listdir(history_dir)
            if f.endswith(('.jpg', '.png'))
        ])[:self.history_frames]

        for frame_file in history_files:
            frame_path = os.path.join(history_dir, frame_file)
            img = Image.open(frame_path).convert('RGB')
            history_frames.append(self.history_transform(img))

        # Stack历史帧: [T, 3, H, W]
        history = torch.stack(history_frames, dim=0)

        # ==========================================
        # 2. 生成 Proxy
        # ==========================================
        targets = {}

        future_files = sorted([
            f for f in os.listdir(future_dir)
            if f.endswith(('.jpg', '.png'))
        ])

        if len(future_files) > 0:
            future_path = os.path.join(future_dir, future_files[0])
            future_img = Image.open(future_path).convert('RGB')

            # Occupancy Proxy (使用你的方法)
            if self.use_occ:
                occ = generate_occupancy_map(future_img, self.img_size)
                targets["occ"] = occ  # [1, H, W]

            # Motion Proxy (使用你的方法)
            if self.use_motion:
                # 获取最后一帧历史帧
                last_history_path = os.path.join(history_dir, history_files[-1])
                last_history_img = Image.open(last_history_path).convert('RGB')

                # 计算motion (帧差)
                motion = generate_motion_map(
                    last_history_img,
                    future_img,
                    self.img_size
                )
                targets["motion"] = motion  # [1, H, W]

        return history, targets


# ==========================================
# 简化版 Dataset
# ==========================================
class UAVSimpleImageDataset(Dataset):
    """
    简化版图片数据集

    数据结构:
    data_root/
        sample_0000_history.jpg  # 拼接的历史帧
        sample_0000_future.jpg   # 目标帧
    """

    def __init__(self, root_dir, img_size=(640, 640),
                 use_occ=True, use_motion=False):
        self.root_dir = root_dir
        self.img_size = img_size
        self.use_occ = use_occ
        self.use_motion = use_motion

        # 查找所有样本
        history_files = sorted([
            f for f in os.listdir(root_dir)
            if f.endswith('_history.jpg') or f.endswith('_history.png')
        ])

        self.samples = []
        for hist_file in history_files:
            base_name = hist_file.replace('_history.jpg', '').replace('_history.png', '')
            future_file = None
            for ext in ['.jpg', '.png']:
                candidate = f"{base_name}_future{ext}"
                if os.path.exists(os.path.join(root_dir, candidate)):
                    future_file = candidate
                    break

            if future_file:
                self.samples.append((hist_file, future_file))

        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in {root_dir}")

        # History transform
        self.history_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        print(f"[Dataset] Loaded {len(self.samples)} samples from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hist_file, future_file = self.samples[idx]

        # 加载历史帧
        hist_path = os.path.join(self.root_dir, hist_file)
        hist_img = Image.open(hist_path).convert('RGB')
        history = self.history_transform(hist_img)
        history = history.unsqueeze(0)  # [1, 3, H, W]

        # 加载目标帧
        future_path = os.path.join(self.root_dir, future_file)
        future_img = Image.open(future_path).convert('RGB')

        targets = {}

        # Occupancy
        if self.use_occ:
            occ = generate_occupancy_map(future_img, self.img_size)
            targets["occ"] = occ

        # Motion
        if self.use_motion:
            motion = generate_motion_map(hist_img, future_img, self.img_size)
            targets["motion"] = motion

        return history, targets


# ==========================================
# 测试代码
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing UAVImageDataset with your proxy methods")
    print("=" * 60)

    # 测试路径 (请根据实际情况修改)
    test_root = r"D:\model_12.22_fixed\images"

    # 测试1: 只用 Occupancy
    print("\n[Test 1] Occupancy Only")
    print("=" * 60)
    try:
        dataset = UAVImageDataset(
            root_dir=test_root,
            history_frames=9,
            use_occ=True,
            use_motion=False,
            img_size=(640, 640)
        )

        if len(dataset) > 0:
            history, targets = dataset[0]
            print(f"✓ History shape: {history.shape}")

            if 'occ' in targets:
                print(f"✓ Occupancy shape: {targets['occ'].shape}")
                print(f"  - Value range: [{targets['occ'].min():.3f}, {targets['occ'].max():.3f}]")
        else:
            print("✗ Dataset is empty")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

    # 测试2: 只用 Motion
    print("\n[Test 2] Motion Only")
    print("=" * 60)
    try:
        dataset = UAVImageDataset(
            root_dir=test_root,
            history_frames=9,
            use_occ=False,
            use_motion=True,
            img_size=(640, 640)
        )

        if len(dataset) > 0:
            history, targets = dataset[0]
            print(f"✓ History shape: {history.shape}")

            if 'motion' in targets:
                print(f"✓ Motion shape: {targets['motion'].shape}")
                print(f"  - Value range: [{targets['motion'].min():.3f}, {targets['motion'].max():.3f}]")
        else:
            print("✗ Dataset is empty")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

    # 测试3: 同时使用两者
    print("\n[Test 3] Both Occupancy and Motion")
    print("=" * 60)
    try:
        dataset = UAVImageDataset(
            root_dir=test_root,
            history_frames=9,
            use_occ=True,
            use_motion=True,
            img_size=(640, 640)
        )

        if len(dataset) > 0:
            history, targets = dataset[0]
            print(f"✓ History shape: {history.shape}")

            for key, value in targets.items():
                print(f"✓ Target {key} shape: {value.shape}")
                print(f"  - Value range: [{value.min():.3f}, {value.max():.3f}]")

            # 测试多个样本
            print("\n[Testing multiple samples...]")
            for i in range(min(3, len(dataset))):
                history, targets = dataset[i]
                print(f"  Sample {i}: history={history.shape}, "
                      f"occ={targets.get('occ', torch.empty(0)).shape}, "
                      f"motion={targets.get('motion', torch.empty(0)).shape}")
        else:
            print("✗ Dataset is empty")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)

    print("\n使用方法:")
    print("="*60)
    print("""
from image_dataset_realtime import UAVImageDataset
from torch.utils.data import DataLoader

# 创建dataset
dataset = UAVImageDataset(
    root_dir="D:/model_12.22_fixed/images",
    history_frames=9,
    use_occ=True,
    use_motion=True,
    img_size=(640, 640)
)

# 创建DataLoader
train_loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,  # 多进程加载
    pin_memory=True
)

# 训练
for history, targets in train_loader:
    # history: [B, T, 3, H, W]
    # targets['occ']: [B, 1, H, W]
    # targets['motion']: [B, 1, H, W]
    outputs = model(history)
    loss = criterion(outputs, targets)
    ...
    """)