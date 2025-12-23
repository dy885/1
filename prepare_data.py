
import os
import shutil
from PIL import Image
import argparse
from pathlib import Path
import cv2


def create_sequence_dataset(video_frames_dir, output_dir, fps=3,
                            history_seconds=3, pred_offset_seconds=1,
                            start_idx_offset=0):

    sequences_dir = os.path.join(output_dir, 'sequences')
    os.makedirs(sequences_dir, exist_ok=True)

    # 获取所有帧
    frames = sorted([f for f in os.listdir(video_frames_dir)
                     if f.endswith(('.jpg', '.png'))])
    total_frames = len(frames)

    # 计算参数
    history_frames = fps * history_seconds  # 9帧（3秒 × 3fps）
    pred_offset = fps * pred_offset_seconds  # 3帧（1秒 × 3fps）
    slide_step = fps  # 滑动1秒 = 3帧

    print(f"[INFO] Found {total_frames} frames")
    print(f"[INFO] FPS: {fps}, History: {history_seconds}s ({history_frames} frames)")
    print(f"[INFO] Prediction offset: {pred_offset_seconds}s ({pred_offset} frames)")
    print(f"[INFO] Slide step: 1s ({slide_step} frames)")
    print(f"[INFO] Starting sequence index: {start_idx_offset}")

    sample_idx = start_idx_offset
    start_idx = 0

    # 滑动窗口：每次移动1秒（3帧）
    while start_idx + history_frames + pred_offset <= total_frames:
        # 创建样本目录
        seq_dir = os.path.join(sequences_dir, f'seq_{sample_idx:04d}')
        history_dir = os.path.join(seq_dir, 'history')
        future_dir = os.path.join(seq_dir, 'future')

        os.makedirs(history_dir, exist_ok=True)
        os.makedirs(future_dir, exist_ok=True)

        # 复制历史帧（9张，覆盖3秒）
        for i in range(history_frames):
            src = os.path.join(video_frames_dir, frames[start_idx + i])
            dst = os.path.join(history_dir, f'frame_{i:02d}.jpg')
            shutil.copy2(src, dst)

        # 复制目标帧（预测未来1秒后的帧）
        target_idx = start_idx + history_frames + pred_offset - 1
        src = os.path.join(video_frames_dir, frames[target_idx])
        dst = os.path.join(future_dir, 'frame_00.jpg')
        shutil.copy2(src, dst)

        sample_idx += 1
        start_idx += slide_step  # 滑动1秒

        if sample_idx % 50 == 0:
            print(f"[INFO] Created {sample_idx - start_idx_offset} samples...")

    print(f"[INFO] Total {sample_idx - start_idx_offset} samples created from this video")
    return sample_idx  # 返回最后的索引，供下一个视频使用


def extract_frames_from_video(video_path, output_dir, target_fps=3):

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Video: {video_path}")
    print(f"[INFO] Original FPS: {original_fps:.2f}, Total frames: {total_frames}")
    print(f"[INFO] Target FPS: {target_fps}")

    # 计算采样间隔（30fps -> 3fps，每10帧取1帧）
    frame_interval = round(original_fps / target_fps)
    print(f"[INFO] Extracting every {frame_interval} frames (30fps -> 3fps)")

    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 每隔frame_interval帧保存一次
        if frame_idx % frame_interval == 0:
            frame_path = os.path.join(output_dir, f'{saved_idx:05d}.jpg')
            cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_idx += 1

        frame_idx += 1

        if saved_idx % 100 == 0 and saved_idx > 0:
            print(f"[INFO] Extracted {saved_idx} frames...")

    cap.release()
    print(f"[INFO] Extracted {saved_idx} frames from video")

    return saved_idx


def process_all_videos(video_dir, output_dir, target_fps=3,
                       history_seconds=3, pred_offset_seconds=1):
    video_extensions = {'.mp4', '.MP4', '.avi', '.AVI', '.mov', '.MOV',
                       '.mkv', '.MKV', '.flv', '.FLV'}
    # 获取所有视频文件
    video_files = []
    for file in os.listdir(video_dir):
        if Path(file).suffix in video_extensions:
            video_files.append(os.path.join(video_dir, file))

    if not video_files:
        print(f"[ERROR] No video files found in {video_dir}")
        return

    print(f"\n{'='*60}")
    print(f"Found {len(video_files)} video(s) to process")
    print(f"{'='*60}\n")

    total_samples = 0
    next_seq_idx = 0  # 跨视频的全局序列索引

    for idx, video_path in enumerate(video_files, 1):
        print(f"\n{'='*60}")
        print(f"Processing video {idx}/{len(video_files)}")
        print(f"{'='*60}")

        video_name = Path(video_path).stem

        # 创建临时帧目录
        temp_frames_dir = os.path.join(output_dir, f'temp_frames_{video_name}')

        try:
            # 提取帧
            num_frames = extract_frames_from_video(video_path, temp_frames_dir, target_fps)

            if num_frames == 0:
                print(f"[WARNING] No frames extracted from {video_path}")
                continue

            # 创建序列数据集
            print(f"\n[INFO] Creating sequences from {video_name}...")
            next_seq_idx = create_sequence_dataset(
                temp_frames_dir,
                output_dir,
                fps=target_fps,
                history_seconds=history_seconds,
                pred_offset_seconds=pred_offset_seconds,
                start_idx_offset=next_seq_idx  # 传入全局索引
            )

            num_samples = next_seq_idx - total_samples  # 计算本视频生成的样本数
            total_samples = next_seq_idx
            print(f"[INFO] Created {num_samples} samples from {video_name}")
            print(f"[INFO] Total sequences so far: {total_samples}")

        except Exception as e:
            print(f"[ERROR] Failed to process {video_path}: {str(e)}")

        finally:
            # 清理临时文件
            if os.path.exists(temp_frames_dir):
                print(f"[INFO] Cleaning up temporary files...")
                shutil.rmtree(temp_frames_dir)

    print(f"\n{'='*60}")
    print(f"All videos processed!")
    print(f"{'='*60}")
    print(f"Total samples created: {total_samples}")
    print(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare UAV dataset from videos - 每秒3帧，滑动窗口1秒'
    )

    # 输入输出
    parser.add_argument('--video_dir',
                       default=r"D:\project\data\raw_videos",
                       help='视频文件夹路径')
    parser.add_argument('--output',
                       default=r'D:\model_12.22_fixed\images',
                       help='输出目录')

    # 数据集参数
    parser.add_argument('--fps', type=int, default=3,
                       help='目标fps（每秒帧数，默认3）')
    parser.add_argument('--history_seconds', type=int, default=3,
                       help='历史时长（秒，默认3秒=9帧）')
    parser.add_argument('--pred_offset_seconds', type=int, default=1,
                       help='预测偏移（秒，默认1秒=3帧）')

    args = parser.parse_args()

    # 检查输入目录
    if not os.path.exists(args.video_dir):
        print(f"[ERROR] Video directory not found: {args.video_dir}")
        return

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    print("\n" + "="*60)
    print("UAV Dataset Preparation")
    print("="*60)
    print(f"Video directory: {args.video_dir}")
    print(f"Output directory: {args.output}")
    print(f"Settings:")
    print(f"  - Original video: 30fps")
    print(f"  - Target FPS: {args.fps} (每秒{args.fps}帧)")
    print(f"  - Sampling: 每10帧取1帧 (30fps -> 3fps)")
    print(f"  - History: {args.history_seconds}秒 ({args.fps * args.history_seconds}帧)")
    print(f"  - Prediction offset: {args.pred_offset_seconds}秒")
    print(f"  - Slide step: 1秒 ({args.fps}帧)")
    print("="*60 + "\n")

    # 处理所有视频
    process_all_videos(
        args.video_dir,
        args.output,
        target_fps=args.fps,
        history_seconds=args.history_seconds,
        pred_offset_seconds=args.pred_offset_seconds
    )

    print("\n" + "="*60)
    print("Dataset preparation complete!")
    print("="*60)
    print(f"\n数据集结构:")
    print(f"  {args.output}/")
    print(f"    sequences/")
    print(f"      seq_0000/")
    print(f"        history/")
    print(f"          frame_00.jpg  # 第1秒第1帧")
    print(f"          frame_01.jpg  # 第1秒第2帧")
    print(f"          frame_02.jpg  # 第1秒第3帧")
    print(f"          frame_03.jpg  # 第2秒第1帧")
    print(f"          ...")
    print(f"          frame_08.jpg  # 第3秒第3帧")
    print(f"        future/")
    print(f"          frame_00.jpg  # 第4秒后某帧")
    print(f"      seq_0001/  # 滑动1秒后的窗口（第2-4秒）")
    print(f"        ...")


if __name__ == "__main__":
    main()