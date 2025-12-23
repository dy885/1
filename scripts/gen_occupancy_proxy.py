import torch
from torchvision import transforms
from PIL import Image
import os

def generate_occupancy_map(img, img_size=(640,640)):
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


def generate_occupancy_dataset(dataset_dir="data/dataset/train",
                               occ_dir="data/dataset/train_occ",
                               img_size=(640,640)):
    """
    批量处理已有 .pt 文件，将 future 图像转为 occupancy map
    """
    os.makedirs(occ_dir, exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    files = [f for f in os.listdir(dataset_dir) if f.endswith(".pt")]
    print(f"[INFO] Generating occupancy maps for {len(files)} samples...")

    for idx, file in enumerate(files):
        path = os.path.join(dataset_dir, file)
        data = torch.load(path)
        future = data["future"]  # [T_f, 3, H, W]

        occ_maps = []
        for t in range(future.shape[0]):
            img = transforms.ToPILImage()(future[t])
            occ = transform(img)  # [1,H,W]
            occ_maps.append(occ)

        occ_tensor = torch.stack(occ_maps, dim=0)  # [T_f,1,H,W]
        data["future"] = occ_tensor

        torch.save(data, os.path.join(occ_dir, file))

        # 进度打印
        if (idx + 1) % 50 == 0 or (idx + 1) == len(files):
            print(f"[INFO] Processed {idx+1}/{len(files)} samples...")

    print(f"[INFO] Occupancy proxy generated at {occ_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Occupancy Map from dataset")
    parser.add_argument("--dataset_dir", type=str, default="data/dataset/train")
    parser.add_argument("--occ_dir", type=str, default="data/dataset/train_occ")
    parser.add_argument("--img_size", type=int, nargs=2, default=(640,640))
    args = parser.parse_args()

    generate_occupancy_dataset(args.dataset_dir, args.occ_dir, tuple(args.img_size))
