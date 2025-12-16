import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. クラス定義 (学習時と同じ定義)
# ==========================================
class MultiVideoSpeckleDataset(Dataset):
    def __init__(self, label_dir, img_root_dir, transform=None):
        self.img_root_dir = img_root_dir
        self.transform = transform
        self.data_pairs = [] 

        # ラベルファイルを取得
        label_pattern = os.path.join(label_dir, "label_*.txt")
        label_files = glob.glob(label_pattern)
        
        if len(label_files) == 0:
            print(f"警告: {label_dir} にラベルファイル(label_*.txt)が見つかりません。")

        for l_file in label_files:
            filename = os.path.basename(l_file)
            # "label_video_A.txt" -> "video_A"
            video_name = filename.replace("label_", "").replace(".txt", "")
            
            with open(l_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    parts = line.split(',')
                    if len(parts) < 2: continue
                    try:
                        frame_num = int(parts[0])
                        label_val = int(parts[1])
                    except ValueError:
                        continue

                    # 画像パス生成
                    img_name = f"{frame_num:04d}.png" 
                    img_path = os.path.join(self.img_root_dir, video_name, img_name)
                    
                    target_label = label_val - 1
                    self.data_pairs.append((img_path, target_label))

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        img_path, label = self.data_pairs[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise FileNotFoundError(f"画像が見つかりません: {img_path}")

        image = cv2.resize(image, (256, 256))
        image = image.astype('float32') / 255.0
        image_tensor = torch.from_numpy(image).unsqueeze(0) 
        label_tensor = torch.tensor(label, dtype=torch.long)
        return image_tensor, label_tensor

class SpeckleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(SpeckleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ==========================================
# 2. 検証実行メイン部
# ==========================================
if __name__ == '__main__':
    # --- パス設定 (修正版) ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 検証用ラベル: src/test_labels_3class (ここはsrc内のまま想定)
    VAL_LABEL_DIR = os.path.join(BASE_DIR, "test_labels_3class")
    
    # ★修正点: 検証用画像も src の外 (一つ上の階層) にある場合
    # .../src/../test_autocorr_images
    VAL_IMAGE_ROOT = os.path.join(BASE_DIR, "..", "test_autocorr_images")
    
    # 学習済みモデル: src/speckle_model.pth
    MODEL_PATH = os.path.join(BASE_DIR, "speckle_model.pth")

    print(f"検証ラベルフォルダ: {VAL_LABEL_DIR}")
    print(f"検証画像フォルダ: {VAL_IMAGE_ROOT}")

    # --- 準備 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if os.path.exists(VAL_LABEL_DIR) and os.path.exists(VAL_IMAGE_ROOT):
        # データセット作成
        val_dataset = MultiVideoSpeckleDataset(VAL_LABEL_DIR, VAL_IMAGE_ROOT)
        
        if len(val_dataset) > 0:
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
            print(f"検証データ数: {len(val_dataset)}")

            # モデルロード
            model = SpeckleCNN(num_classes=3).to(device)
            if os.path.exists(MODEL_PATH):
                model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
                print("学習済みモデルを読み込みました。")
            else:
                print(f"エラー: モデルファイル({MODEL_PATH})が見つかりません。学習(train.py)を先に実行してください。")
                exit()

            # --- 推論実行 ---
            model.eval()
            all_preds = []
            all_labels = []

            print("検証中...")
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.numpy())

            # --- 結果表示 ---
            correct = sum([1 for p, l in zip(all_preds, all_labels) if p == l])
            total = len(all_labels)
            accuracy = 100 * correct / total
            
            print("-" * 30)
            print(f"検証結果 (Accuracy): {accuracy:.2f}% ({correct}/{total})")
            print("-" * 30)

            # 詳細レポート
            try:
                from sklearn.metrics import classification_report, confusion_matrix
                import seaborn as sns
                
                class_names = ["Empty(1)", "Occupied(2)", "Other(3)"]
                print(classification_report(all_labels, all_preds, target_names=class_names))

                cm = confusion_matrix(all_labels, all_preds)
                plt.figure(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.title("Confusion Matrix")
                plt.show()
                
            except ImportError:
                print("sklearnなどのライブラリがないため、詳細レポートはスキップします。")

        else:
            print("エラー: データセットが空です。ラベルファイルと画像が正しく紐付いているか確認してください。")
    else:
        print("エラー: 指定されたフォルダが見つかりません。パスを確認してください。")