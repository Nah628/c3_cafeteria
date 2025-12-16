import os
import glob
import re

# ==========================================
# 設定エリア
# ==========================================
# スクリプトを 'src' フォルダ内に置く前提のパス設定
current_dir = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(current_dir, 'front_view')       # 入力: src/front_view
OUTPUT_DIR = os.path.join(current_dir, 'labels_3class')   # 出力: src/labels_3class

# クラス定義 (3クラス)
# 0: Empty (空席)
# 1: Action (動作中: 座る・立つ)
# 2: Sitting (着席静止)
CLASS_EMPTY = 0
CLASS_ACTION = 1
CLASS_SITTING = 2

# アクションID (OA18データセット)
ACTION_SIT_DOWN = 12
ACTION_STAND_UP = 13
# ==========================================

def parse_action_file(filepath):
    """テキストファイルを読み込み、アクションリストを取得"""
    actions = []
    max_frame = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            # 不要な文字を除去 (例: を削除)
            clean_line = re.sub(r'\[.*?\]\s*', '', line.strip())
            
            try:
                parts = clean_line.split(',')
                if len(parts) == 3:
                    aid = int(parts[0])
                    start = int(parts[1])
                    end = int(parts[2])
                    
                    actions.append({'id': aid, 'start': start, 'end': end})
                    
                    if end > max_frame:
                        max_frame = end
            except ValueError:
                continue
                
    # 開始フレーム順にソート
    actions.sort(key=lambda x: x['start'])
    return actions, max_frame

def generate_labels(actions, max_frame):
    labels = [CLASS_EMPTY] * (max_frame + 1)
    
    current_sit_end_frame = None
    is_sitting_phase = False

    for action in actions:
        aid = action['id']
        start = action['start']
        end = action['end']

        # --- 座る動作 (12) ---
        if aid == ACTION_SIT_DOWN:
            # ★修正点: すでに着席モードなら、前の「座る」～今の「座る」の間を埋める
            if is_sitting_phase and current_sit_end_frame is not None:
                 fill_start = current_sit_end_frame + 1
                 fill_end = start - 1
                 if fill_end >= fill_start:
                     for i in range(fill_start, fill_end + 1):
                         if i < len(labels): labels[i] = CLASS_SITTING

            # 動作中を Action(1) に設定
            for i in range(start, end + 1):
                if i < len(labels): labels[i] = CLASS_ACTION
            
            # フェーズ更新
            current_sit_end_frame = end
            is_sitting_phase = True

        # --- 立つ動作 (13) ---
        elif aid == ACTION_STAND_UP:
            if is_sitting_phase and current_sit_end_frame is not None:
                fill_start = current_sit_end_frame + 1
                fill_end = start - 1
                
                if fill_end >= fill_start:
                    for i in range(fill_start, fill_end + 1):
                        if i < len(labels): labels[i] = CLASS_SITTING
            
            for i in range(start, end + 1):
                if i < len(labels): labels[i] = CLASS_ACTION
            
            is_sitting_phase = False
            current_sit_end_frame = None

    # 末尾処理
    if is_sitting_phase and current_sit_end_frame is not None:
        fill_start = current_sit_end_frame + 1
        for i in range(fill_start, max_frame + 1):
            if i < len(labels): labels[i] = CLASS_SITTING

    return labels

def main():
    # 出力フォルダ作成
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"作成しました: {OUTPUT_DIR}")

    # f_000.txt 〜 f_039.txt を取得
    files = glob.glob(os.path.join(INPUT_DIR, '*.txt'))
    files.sort() # 名前順に整列

    print(f"対象ファイル数: {len(files)} 件 (src/front_view 内)")

    count = 0
    for filepath in files:
        filename = os.path.basename(filepath)
        
        # 解析とラベル生成
        actions, max_frame = parse_action_file(filepath)
        
        if max_frame == 0:
            print(f"Skipping empty or invalid file: {filename}")
            continue

        labels = generate_labels(actions, max_frame)
        
        # 保存 (例: label_f_000.txt)
        output_filename = f"label_{filename}"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as out_f:
            for i, label in enumerate(labels):
                out_f.write(f"{i},{label}\n")
        
        count += 1

    print(f"\n完了: {count} 個のラベルファイルを生成しました。")
    print(f"保存先: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()