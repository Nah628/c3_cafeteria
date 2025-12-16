import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# =========================
# 設定
# =========================
video_dir = r"C:\Users\yoshi\c3\c3_2\videos"   # mp4動画を置くフォルダ
save_root = r"C:\Users\yoshi\c3\c3_2\data\auttocorr_images"

sigma = 0.2                 # 位相乱れ強度
process_every_sec = False   # Trueにすると1秒ごと処理

# =========================
# 関数
# =========================
def autocorr_2d(x):
    F = np.fft.fft2(x)
    return np.fft.fftshift(np.abs(np.fft.ifft2(np.abs(F) ** 2)))

def normalize01(x):
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x

# =========================
# 動画リスト取得
# =========================
video_paths = glob.glob(os.path.join(video_dir, "*.mp4"))

if len(video_paths) == 0:
    raise RuntimeError("mp4動画が見つかりません")

# =========================
# 動画リスト取得
# =========================
video_paths = glob.glob(os.path.join(video_dir, "*.mp4"))

if len(video_paths) == 0:
    raise RuntimeError("mp4動画が見つかりません")

# =========================
# 動画ごとの処理
# =========================
for video_path in video_paths:

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_dir = os.path.join(save_root, video_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n処理開始: {video_name}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  開けません: {video_path}")
        continue

    # ---- ここで取得する（重要）----
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = int(fps) if process_every_sec else 1

    print(f"[{video_name}] FPS={fps:.2f}, Total frames={total_frames}")

    frame_count = 0
    save_index = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ---- 進捗表示（100フレームごと）----
        if frame_count % 100 == 0:
            print(f"[{video_name}] {frame_count}/{total_frames}")

        if frame_count % step != 0:
            frame_count += 1
            continue

        # ---- グレースケール ----
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if img.max() > 0:
            img /= img.max()

        # ---- フーリエ変換 ----
        F = np.fft.fft2(img)
        amp = np.abs(F)
        phase_orig = np.angle(F)

        # ---- 弱ランダム位相 ----
        delta_phase = np.random.normal(0, sigma, img.shape)
        phase_new = phase_orig + delta_phase
        F_rand = amp * np.exp(1j * phase_new)

        # ---- 散乱画像 ----
        img_rand = np.real(np.fft.ifft2(F_rand))

        # ---- 自己相関 ----
        ac = autocorr_2d(img_rand)

        # ---- 保存 ----
        filename = f"{save_index:04d}.png"
        filepath = os.path.join(save_dir, filename)
        plt.imsave(filepath, normalize01(ac), cmap="gray")

        save_index += 1
        frame_count += 1

    cap.release()
    print(f"[{video_name}] 保存枚数: {save_index-1}")

print("\nすべての動画の処理が完了しました")
