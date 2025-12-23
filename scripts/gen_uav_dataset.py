import os
import torch
from PIL import Image
from torchvision import transforms
import cv2
from gen_occupancy_proxy import generate_occupancy_map
from gen_motion_proxy import generate_motion_map


def extract_frames_from_video(video_path, frames_dir, take_every_n=10):
    """
    拆帧: 每 take_every_n 帧取一张，并保存到 frames_dir
    """
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"[INFO] Video has {total_frames} frames at {fps} FPS, taking every {take_every_n} frames.")

    idx = 0
    saved_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % take_every_n == 0:
            frame_path = os.path.join(frames_dir, f"{saved_idx:05d}.png")
            success = cv2.imwrite(frame_path, frame)
            if not success:
                print(f"[WARNING] Failed to write frame {saved_idx} to {frame_path}")
            saved_idx += 1
            if saved_idx % 50 == 0:
                print(f"[INFO] Extracted {saved_idx} frames...")

        idx += 1

    cap.release()
    print(f"[INFO] Extracted total {saved_idx} frames to {frames_dir}")
    return frames_dir, fps


def gen_uav_dataset(raw_dir, save_dir, history_sec=3, pred_sec=2,
                    original_fps=30, take_every_n=10, img_size=(64, 64),
                    use_occ=True, use_motion=True, is_video=False,
                    sample_idx_offset=0):
    """
    生成 UAV 数据集
    历史 history_sec 秒 → 预测 pred_sec 秒后的冲突情况

    参数说明：
    - history_sec: 历史窗口长度（秒）
    - pred_sec: 预测未来多少秒后的冲突
    - original_fps: 原始视频的FPS
    - take_every_n: 从原始视频中每隔多少帧取一帧
    - sample_idx_offset: 样本索引偏移量，用于避免文件名重复
    """
    if is_video:
        frames_dir = os.path.join(save_dir, "frames_temp")
        raw_dir, original_fps = extract_frames_from_video(raw_dir, frames_dir, take_every_n=take_every_n)

    os.makedirs(save_dir, exist_ok=True)
    transform = transforms.Compose([transforms.Resize((640, 640)), transforms.ToTensor()])

    frames = sorted([os.path.join(raw_dir, f) for f in os.listdir(raw_dir)
                     if f.endswith(".png") or f.endswith(".jpg")])
    total_frames = len(frames)

    # 计算参数
    # 采样后的有效FPS
    effective_fps = original_fps / take_every_n

    # 历史窗口需要的帧数
    history_len = int(history_sec * effective_fps)

    # 预测目标距离历史窗口结束的帧数
    pred_offset = int(pred_sec * effective_fps)

    # 滑动窗口步长（1秒）
    slide_step = int(1 * effective_fps)

    print(f"[INFO] Generating samples at {save_dir}")
    print(f"[INFO] Original FPS: {original_fps}, Take every {take_every_n} frames")
    print(f"[INFO] Effective FPS: {effective_fps}")
    print(f"[INFO] History window: {history_sec}s = {history_len} frames")
    print(f"[INFO] Prediction offset: {pred_sec}s = {pred_offset} frames after history")
    print(f"[INFO] Slide step: 1s = {slide_step} frames")
    print(f"[INFO] Total frames available: {total_frames}")
    print(f"[INFO] Sample index offset: {sample_idx_offset}")

    start_idx = 0
    sample_idx = sample_idx_offset

    while start_idx + history_len + pred_offset < total_frames:
        # 提取历史帧
        hist_frames = [transform(Image.open(frames[i]).convert("RGB"))
                       for i in range(start_idx, start_idx + history_len)]
        history = torch.stack(hist_frames, dim=0)  # [T_h, 3, H, W]

        # 预测目标帧的索引
        target_idx = start_idx + history_len + pred_offset

        print(f"[DEBUG] Sample {sample_idx}: history frames [{start_idx}:{start_idx + history_len}), "
              f"target frame {target_idx}")

        target_img = transform(Image.open(frames[target_idx]).convert("RGB"))

        sample = {"history": history}

        if use_occ:
            occ = generate_occupancy_map(target_img, img_size)
            sample["future"] = occ

        if use_motion:
            motion = generate_motion_map(history[-1], target_img, img_size)
            sample["motion"] = motion

        torch.save(sample, os.path.join(save_dir, f"{sample_idx:05d}.pt"))
        sample_idx += 1

        # 滑动窗口
        start_idx += slide_step

        # 打印进度
        if (sample_idx - sample_idx_offset) % 10 == 0:
            print(f"[INFO] Saved {sample_idx - sample_idx_offset} samples from current video...")

    total_samples = sample_idx - sample_idx_offset
    print(f"[INFO] Total {total_samples} samples generated from current video")
    return sample_idx  # 返回下一个可用的索引


def process_all_videos(raw_videos_dir, output_base_dir, history_sec_list=[3, 5],
                       pred_sec=2, take_every_n=10):
    """
    处理 raw_videos_dir 下的所有视频文件

    参数：
    - raw_videos_dir: 原始视频文件夹路径
    - output_base_dir: 输出基础目录
    - history_sec_list: 历史窗口长度列表（秒）
    - pred_sec: 预测未来秒数
    - take_every_n: 采样间隔
    """
    # 支持的视频格式
    video_extensions = ['.mp4', '.MP4', '.avi', '.AVI', '.mov', '.MOV', '.mkv', '.MKV']

    # 获取所有视频文件
    video_files = []
    for file in os.listdir(raw_videos_dir):
        if any(file.endswith(ext) for ext in video_extensions):
            video_files.append(file)

    video_files.sort()  # 按名称排序，确保处理顺序一致

    if not video_files:
        print(f"[ERROR] No video files found in {raw_videos_dir}")
        return

    print(f"[INFO] Found {len(video_files)} video files:")
    for i, vf in enumerate(video_files):
        print(f"  [{i + 1}] {vf}")

    # 为每个历史窗口长度创建数据集
    for history_sec in history_sec_list:
        print("\n" + "=" * 80)
        print(f"Processing with history_sec={history_sec}s, pred_sec={pred_sec}s")
        print("=" * 80)

        # 创建输出目录
        save_dir = os.path.join(output_base_dir, f"train_{history_sec}s")
        os.makedirs(save_dir, exist_ok=True)

        # 全局样本索引，确保不同视频的样本不重名
        global_sample_idx = 0

        # 处理每个视频
        for video_idx, video_file in enumerate(video_files):
            video_path = os.path.join(raw_videos_dir, video_file)
            video_name = os.path.splitext(video_file)[0]  # 去掉扩展名

            print("\n" + "-" * 80)
            print(f"[{video_idx + 1}/{len(video_files)}] Processing video: {video_file}")
            print("-" * 80)

            # 为每个视频创建临时帧目录
            frames_dir = os.path.join(save_dir, f"frames_{video_name}")
            os.makedirs(frames_dir, exist_ok=True)

            try:
                # 提取帧
                frames_dir, fps = extract_frames_from_video(video_path, frames_dir, take_every_n=take_every_n)

                # 生成数据集，传入当前的全局索引作为偏移
                next_sample_idx = gen_uav_dataset(
                    raw_dir=frames_dir,
                    save_dir=save_dir,
                    history_sec=history_sec,
                    pred_sec=pred_sec,
                    original_fps=fps,
                    take_every_n=take_every_n,
                    use_occ=True,
                    use_motion=True,
                    is_video=False,
                    sample_idx_offset=global_sample_idx
                )

                # 更新全局索引
                samples_generated = next_sample_idx - global_sample_idx
                print(f"[INFO] Generated {samples_generated} samples from {video_file}")
                global_sample_idx = next_sample_idx

            except Exception as e:
                print(f"[ERROR] Failed to process {video_file}: {str(e)}")
                continue

        print("\n" + "=" * 80)
        print(f"[SUMMARY] Total {global_sample_idx} samples saved to {save_dir}")
        print("=" * 80)


if __name__ == "__main__":
    # 配置路径
    RAW_VIDEOS_DIR = r"D:\project\data\raw_videos"
    OUTPUT_BASE_DIR = r"D:\project\data"

    # 配置参数
    HISTORY_SEC_LIST = [3, 5]  # 生成 3秒 和 5秒 历史窗口的数据集
    PRED_SEC = 2  # 预测 2秒 后的状态
    TAKE_EVERY_N = 10  # 每10帧取一帧

    print("=" * 80)
    print("UAV Dataset Batch Processing")
    print("=" * 80)
    print(f"Raw videos directory: {RAW_VIDEOS_DIR}")
    print(f"Output base directory: {OUTPUT_BASE_DIR}")
    print(f"History window lengths: {HISTORY_SEC_LIST} seconds")
    print(f"Prediction offset: {PRED_SEC} seconds")
    print(f"Frame sampling: every {TAKE_EVERY_N} frames")
    print("=" * 80)

    # 处理所有视频
    process_all_videos(
        raw_videos_dir=RAW_VIDEOS_DIR,
        output_base_dir=OUTPUT_BASE_DIR,
        history_sec_list=HISTORY_SEC_LIST,
        pred_sec=PRED_SEC,
        take_every_n=TAKE_EVERY_N
    )

    print("\n" + "=" * 80)
    print("All videos processed successfully!")
    print("=" * 80)