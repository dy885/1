import os
import torch
from torchvision import transforms
from PIL import Image
from torchvision.transforms import ToTensor

def generate_motion_map(prev_frame, next_frame, img_size=(640,640)):
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
        ToTensor()
    ])

    if isinstance(prev_frame, torch.Tensor):
        prev_frame = transforms.ToPILImage()(prev_frame)
    if isinstance(next_frame, torch.Tensor):
        next_frame = transforms.ToPILImage()(next_frame)

    prev = transform(prev_frame)
    next_ = transform(next_frame)
    motion = torch.abs(next_ - prev)
    return motion  # [1,H,W]

def generate_motion_dataset(dataset_dir="data/dataset/train",
                            motion_dir="data/dataset/train_motion",
                            img_size=(640,640)):
    """
    批量处理已有 .pt 文件，为 future 生成 motion map
    """
    os.makedirs(motion_dir, exist_ok=True)

    files = [f for f in os.listdir(dataset_dir) if f.endswith(".pt")]
    print(f"[INFO] Generating motion maps for {len(files)} samples...")

    for idx, file in enumerate(files):
        path = os.path.join(dataset_dir, file)
        data = torch.load(path)
        future = data["future"]  # [T_f, 3, H, W]

        motion_maps = []
        for t in range(1, future.shape[0]):
            motion = generate_motion_map(future[t - 1], future[t], img_size)
            motion_maps.append(motion)

        # pad 第一个时间步为 0
        first = torch.zeros_like(motion_maps[0])
        motion_maps = [first] + motion_maps  # [T_f,1,H,W]
        motion_tensor = torch.stack(motion_maps, dim=0)

        data["motion"] = motion_tensor
        torch.save(data, os.path.join(motion_dir, file))

        # 进度打印
        if (idx + 1) % 50 == 0 or (idx + 1) == len(files):
            print(f"[INFO] Processed {idx+1}/{len(files)} samples...")

    print(f"[INFO] Motion proxy generated at {motion_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Motion Map from dataset")
    parser.add_argument("--dataset_dir", type=str, default="data/dataset/train", help="原始 .pt 文件目录")
    parser.add_argument("--motion_dir", type=str, default="data/dataset/train_motion", help="保存 motion 文件目录")
    parser.add_argument("--img_size", type=int, nargs=2, default=(640,640), help="Motion map 分辨率")
    args = parser.parse_args()

    generate_motion_dataset(args.dataset_dir, args.motion_dir, tuple(args.img_size))
