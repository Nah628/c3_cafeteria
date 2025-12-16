import numpy as np

# 保存したファイルを読み込む
labels_data = np.load('labels.npy')

# 中身を表示
print("--- 保存されたラベルデータの確認 ---")
print(f"全体の形状: {labels_data.shape}")

print("\n最初の5フレーム分のラベル:")
print(labels_data[:])

#print("\n最後の5フレーム分のラベル:")
#print(labels_data[-5:])