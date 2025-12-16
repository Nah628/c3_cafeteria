import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import glob
import cv2
import numpy as np

# ==========================================
# 1. データセット定義 (動画別ラベルファイル対応版)
# ==========================================
class MultiVideoSpeckleDataset(Dataset):
    def __init__(self, label_dir, img_root_dir, transform=None):
        """
        Args:
            label_dir (string): ラベルテキストファイルが入っているフォルダ
            img_root_dir (string): 画像フォルダのルート (中にvideo_A, video_B...がある前提)
        """
        self.img_root_dir = img_root_dir
        self.transform = transform
        self.data_pairs = [] # (画像パス, ラベル) のリストをここに作る

        # 1. label_dir 内の全てのテキストファイルを取得
        # ※Windows環境でパス区切り文字の問題が起きないよう os.path.join を使用
        label_pattern = os.path.join(label_dir, "label_*.txt")
        label_files = glob.glob(label_pattern)
        
        print(f"発見したラベルファイル数: {len(label_files)}")
        if len(label_files) == 0:
            print(f"警告: {label_dir} に 'label_*.txt' が見つかりませんでした。")

        # 2. 各ファイルを読み込んでリスト化
        for l_file in label_files:
            # ファイル名から動画名を特定
            # 例: "label_video_test.txt" -> "video_test"
            filename = os.path.basename(l_file)
            video_name = filename.replace("label_", "").replace(".txt", "")
            
            with open(l_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    
                    # カンマ区切りでパース (フレーム番号, ラベル)
                    # 例: "1, 1" -> frame_num=1, label_val=1
                    parts = line.split(',')
                    if len(parts) < 2: continue
                    
                    try:
                        frame_num = int(parts[0])
                        label_val = int(parts[1])
                    except ValueError:
                        continue # 数値変換できない行はスキップ

                    # 画像パスを構築
                    # 例: .../autocorr_images/video_test/0001.png
                    # ※重要: フレーム番号の桁数(0埋め)は画像保存時の仕様に合わせてください(ここでは4桁)
                    img_name = f"{frame_num:04d}.png" 
                    img_path = os.path.join(self.img_root_dir, video_name, img_name)
                    
                    # ラベルを 0 始まりに変換 (例: 1,2,3 -> 0,1,2)
                    target_label = label_val - 1
                    
                    # リストに追加
                    self.data_pairs.append((img_path, target_label))

        print(f"総学習データ数: {len(self.data_pairs)}")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        img_path, label = self.data_pairs[idx]

        # 画像読み込み (グレースケール)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            # エラー等のため画像が見つからない場合
            raise FileNotFoundError(f"画像読み込みエラー: {img_path}")

        # 前処理 (リサイズと正規化)
        image = cv2.resize(image, (256, 256))
        image = image.astype('float32') / 255.0
        
        # Tensor化 (Channel, Height, Width)
        image_tensor = torch.from_numpy(image).unsqueeze(0) 

        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return image_tensor, label_tensor

# ==========================================
# 2. CNNモデル定義
# ==========================================
class SpeckleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(SpeckleCNN, self).__init__()
        
        # 特徴抽出部 (畳み込み層)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 256 -> 128
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 128 -> 64
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 64 -> 32
        )
        
        # 分類部 (全結合層)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, 256),
            nn.ReLU(),
            nn.Dropout(0.5), # 過学習抑制
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ==========================================
# 3. 実行メイン処理
# ==========================================
if __name__ == '__main__':
    # パス設定
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    LABEL_DIR = os.path.join(BASE_DIR, "labels_3class")           # ラベルフォルダ
    IMAGE_ROOT = os.path.join(BASE_DIR, "..", "autocorr_images")  # 画像フォルダ (srcの一つ上)

    print(f"Label Dir: {LABEL_DIR}")
    print(f"Image Root: {IMAGE_ROOT}")

    # フォルダ存在確認
    if os.path.exists(LABEL_DIR) and os.path.exists(IMAGE_ROOT):
        
        # --- データセット準備 ---
        dataset = MultiVideoSpeckleDataset(LABEL_DIR, IMAGE_ROOT)
        
        # データが空でないか確認
        if len(dataset) > 0:
            # データを分割 (学習用:検証用 = 8:2)
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

            # DataLoader作成 (num_workers=0 はWindowsでのエラー回避のため安全策)
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

            # --- モデル・学習設定 ---
            # GPUが使えるならGPUを使う
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"使用デバイス: {device}")

            model = SpeckleCNN(num_classes=3).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # --- 学習ループ ---
            num_epochs = 10
            print("学習を開始します...")

            for epoch in range(num_epochs):
                model.train() # 学習モード
                running_loss = 0.0
                correct_train = 0
                total_train = 0

                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)

                    optimizer.zero_grad()           # 勾配リセット
                    outputs = model(images)         # 推論
                    loss = criterion(outputs, labels) # 損失計算
                    loss.backward()                 # 逆伝播
                    optimizer.step()                # パラメータ更新

                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_train += labels.size(0)
                    correct_train += (predicted == labels).sum().item()

                train_acc = 100 * correct_train / total_train
                avg_train_loss = running_loss / len(train_loader)

                # --- 検証ループ (各エポック終了後) ---
                model.eval() # 評価モード
                correct_val = 0
                total_val = 0
                val_loss = 0.0
                
                with torch.no_grad(): # 勾配計算なし
                    for images, labels in val_loader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total_val += labels.size(0)
                        correct_val += (predicted == labels).sum().item()

                val_acc = 100 * correct_val / total_val
                avg_val_loss = val_loss / len(val_loader)

                print(f"Epoch [{epoch+1}/{num_epochs}] "
                      f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% | "
                      f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.2f}%")

            # --- モデル保存 ---
            save_path = os.path.join(BASE_DIR, "speckle_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"学習完了。モデルを保存しました: {save_path}")

        else:
            print("エラー: 学習データが読み込めませんでした。画像パスやラベルファイルを確認してください。")

    else:
        print("エラー: 指定されたフォルダが見つかりません。パスを確認してください。")