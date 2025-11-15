"""
视频数据标注脚本。

从说话人脸视频中提取 MotionVector 序列，作为 Audio2Motion 训练的标签。

使用 MediaPipe Face Mesh 提取关键点，然后转换为 MotionVector 格式。
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple
import json

import cv2
import numpy as np
import torchaudio
from tqdm import tqdm


def extract_facial_landmarks(video_path: str, output_dir: Path) -> Optional[np.ndarray]:
    """
    从视频提取面部关键点。

    使用 MediaPipe Face Mesh 提取 468 个 3D 关键点。

    参数:
        video_path: 视频文件路径
        output_dir: 输出目录

    返回:
        landmarks: 关键点序列，shape (num_frames, 468, 3)
    """
    try:
        import mediapipe as mp
    except ImportError:
        print("需要安装 MediaPipe: pip install mediapipe")
        return None

    mp_face_mesh = mp.solutions.face_mesh

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    landmarks_list = []

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        pbar = tqdm(total=total_frames, desc="提取关键点")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 转换为 RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 检测关键点
            results = face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                # 获取第一个人脸的关键点
                face_landmarks = results.multi_face_landmarks[0]

                # 转换为 numpy 数组 (468, 3)
                landmarks = np.array([
                    [lm.x, lm.y, lm.z]
                    for lm in face_landmarks.landmark
                ])
                landmarks_list.append(landmarks)
            else:
                # 没检测到人脸，使用上一帧或零
                if landmarks_list:
                    landmarks_list.append(landmarks_list[-1])
                else:
                    landmarks_list.append(np.zeros((468, 3)))

            pbar.update(1)

        pbar.close()

    cap.release()

    landmarks_array = np.array(landmarks_list)  # (num_frames, 468, 3)
    print(f"提取到 {len(landmarks_array)} 帧关键点")

    return landmarks_array, fps


def landmarks_to_motion_vector(landmarks: np.ndarray) -> np.ndarray:
    """
    将 MediaPipe 关键点转换为 MotionVector。

    参数:
        landmarks: 关键点，shape (num_frames, 468, 3)

    返回:
        motion_vectors: MotionVector 序列，shape (num_frames, 7)
    """
    num_frames = landmarks.shape[0]
    motion_vectors = np.zeros((num_frames, 7), dtype=np.float32)

    # MediaPipe 关键点索引（参考官方文档）
    # 嘴唇关键点
    upper_lip_top = 13
    lower_lip_bottom = 14
    mouth_left = 61
    mouth_right = 291

    # 眼睛关键点
    left_eye_top = 159
    left_eye_bottom = 145
    right_eye_top = 386
    right_eye_bottom = 374

    # 鼻尖（用于头部姿态参考）
    nose_tip = 1

    for i in range(num_frames):
        lm = landmarks[i]  # (468, 3)

        # === 计算 jaw_open（张嘴程度）===
        # 上下嘴唇距离 / 参考距离
        lip_height = abs(lm[upper_lip_top, 1] - lm[lower_lip_bottom, 1])
        # 归一化到 [-1, 1]，需要定义基准
        # 这里简化处理：0.05 为完全闭合，0.15 为最大张开
        jaw_open = np.clip((lip_height - 0.05) / 0.1 * 2 - 1, -1, 1)
        motion_vectors[i, 0] = jaw_open

        # === 计算 mouth_wide（嘴横向宽度）===
        mouth_width = abs(lm[mouth_left, 0] - lm[mouth_right, 0])
        # 归一化
        mouth_wide = np.clip((mouth_width - 0.15) / 0.1 * 2 - 1, -1, 1)
        motion_vectors[i, 1] = mouth_wide

        # === 计算 head_yaw（左右转头）===
        # 使用鼻尖 x 坐标变化（中心为 0.5）
        head_yaw = (lm[nose_tip, 0] - 0.5) * 4  # 缩放到 [-2, 2] 范围
        head_yaw = np.clip(head_yaw, -1, 1)
        motion_vectors[i, 3] = head_yaw

        # === 计算 head_pitch（上下抬头）===
        # 使用鼻尖 y 坐标变化
        head_pitch = (0.5 - lm[nose_tip, 1]) * 4
        head_pitch = np.clip(head_pitch, -1, 1)
        motion_vectors[i, 4] = head_pitch

        # === 计算 head_roll（侧倾）===
        # 使用左右眼睛的相对位置
        left_eye_y = (lm[left_eye_top, 1] + lm[left_eye_bottom, 1]) / 2
        right_eye_y = (lm[right_eye_top, 1] + lm[right_eye_bottom, 1]) / 2
        head_roll = (left_eye_y - right_eye_y) * 10
        head_roll = np.clip(head_roll, -1, 1)
        motion_vectors[i, 5] = head_roll

        # === 计算 eye_blink（眨眼）===
        # 左眼闭合程度
        left_eye_height = abs(lm[left_eye_top, 1] - lm[left_eye_bottom, 1])
        # 右眼闭合程度
        right_eye_height = abs(lm[right_eye_top, 1] - lm[right_eye_bottom, 1])
        # 平均眨眼程度（0=睁开，1=闭合）
        eye_open = (left_eye_height + right_eye_height) / 2
        eye_blink = np.clip(1 - eye_open / 0.02, 0, 1)  # 0.02 为完全睁开的阈值
        motion_vectors[i, 6] = eye_blink

    return motion_vectors


def process_video(
    video_path: str,
    output_dir: str,
    extract_audio: bool = True,
) -> bool:
    """
    处理单个视频文件。

    提取音频和 MotionVector 标签。

    参数:
        video_path: 视频文件路径
        output_dir: 输出目录
        extract_audio: 是否提取音频

    返回:
        success: 是否成功
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_name = video_path.stem

    print(f"\n处理视频: {video_path}")

    # 1. 提取关键点
    print("步骤 1/3: 提取面部关键点...")
    result = extract_facial_landmarks(str(video_path), output_dir)
    if result is None:
        return False

    landmarks, fps = result

    # 2. 转换为 MotionVector
    print("步骤 2/3: 转换为 MotionVector...")
    motion_vectors = landmarks_to_motion_vector(landmarks)

    # 保存 MotionVector
    motion_path = output_dir / f"{video_name}_motion.npy"
    np.save(motion_path, motion_vectors)
    print(f"  保存 MotionVector: {motion_path}")

    # 保存元数据
    metadata = {
        "video_path": str(video_path),
        "num_frames": len(motion_vectors),
        "fps": float(fps),
        "duration_seconds": len(motion_vectors) / fps,
    }
    metadata_path = output_dir / f"{video_name}_meta.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # 3. 提取音频（使用 ffmpeg）
    if extract_audio:
        print("步骤 3/3: 提取音频...")
        audio_path = output_dir / f"{video_name}_audio.wav"

        import subprocess
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn",  # 不要视频
            "-acodec", "pcm_s16le",  # PCM 16-bit
            "-ar", "16000",  # 16kHz
            "-ac", "1",  # 单声道
            str(audio_path)
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"  保存音频: {audio_path}")
        except subprocess.CalledProcessError as e:
            print(f"  音频提取失败: {e}")
            return False
        except FileNotFoundError:
            print("  ffmpeg 未安装，跳过音频提取")

    print(f"✅ 处理完成: {video_name}")
    return True


def main():
    parser = argparse.ArgumentParser(description="从视频提取 MotionVector 标签")
    parser.add_argument("video_path", type=str, help="视频文件路径或目录")
    parser.add_argument("--output_dir", type=str, default="./data/processed",
                        help="输出目录")
    parser.add_argument("--no_audio", action="store_true",
                        help="不提取音频")

    args = parser.parse_args()

    video_path = Path(args.video_path)

    if video_path.is_dir():
        # 批量处理目录下的所有视频
        video_files = list(video_path.glob("*.mp4")) + list(video_path.glob("*.avi"))
        print(f"找到 {len(video_files)} 个视频文件")

        success_count = 0
        for vf in video_files:
            if process_video(str(vf), args.output_dir, not args.no_audio):
                success_count += 1

        print(f"\n处理完成: {success_count}/{len(video_files)} 成功")

    else:
        # 处理单个视频
        process_video(str(video_path), args.output_dir, not args.no_audio)


if __name__ == "__main__":
    main()
