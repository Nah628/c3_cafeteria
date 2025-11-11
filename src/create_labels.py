import cv2
import numpy as np
import os

# --- 設定 ---
VIDEO_FILE_PATH = 'blurred_video.mp4' # アノテーション対象の動画
NUM_SEATS = 6                        # 座席の総数
SKIP_FRAMES = 15                     # 15フレームごと（約0.5秒ごと）にラベル付け
OUTPUT_FILE = 'labels.npy'           # 出力するNumpyファイル名

WINDOW_NAME = 'Annotation Tool'

# ★★★ 2. 2番目のモニター設定 (ご自身の環境に合わせて調整) ★★★
# プライマリモニターの解像度が 1920x1080 の場合、
# その右側にセカンダリモニターがあれば X=1920, Y=0 となります。
SECOND_MONITOR_X = 1920  # ← ここの数値を調整してください (例: 1920 や 2560 など)
SECOND_MONITOR_Y = 0
# ---

def parse_label_input(input_str, num_seats):
    """
    ターミナルからの入力 (例: "1,0,0,1,0,0") を 
    [1, 0, 0, 1, 0, 0] というintのリストに変換する。
    """
    try:
        # カンマで分割し、空白を除去
        parts = [p.strip() for p in input_str.split(',')]
        
        # 座席数と一致するかチェック
        if len(parts) != num_seats:
            print(f"  [エラー] 座席数({num_seats})と一致しません。入力: {len(parts)}席")
            return None
            
        # 0か1かに変換
        labels = [int(p) for p in parts]
        for label in labels:
            if label not in [0, 1]:
                print(f"  [エラー] 0または1以外の値が含まれています。")
                return None
                
        return labels
        
    except ValueError:
        print(f"  [エラー] 数値への変換に失敗しました。")
        return None
    except Exception as e:
        print(f"  [エラー] 予期せぬエラー: {e}")
        return None
def main():
    # 動画ファイルが存在するかチェック
    if not os.path.exists(VIDEO_FILE_PATH):
        print(f"エラー: 動画ファイルが見つかりません: {VIDEO_FILE_PATH}")
        return

    cap = cv2.VideoCapture(VIDEO_FILE_PATH)
    if not cap.isOpened():
        print("エラー: 動画を開けませんでした。")
        return
    
    # ★★★ 3. ウィンドウの事前作成と設定 ★★★
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL) # リサイズ可能なウィンドウを作成
    cv2.moveWindow(WINDOW_NAME, SECOND_MONITOR_X, SECOND_MONITOR_Y) # 2番目のモニターへ移動
    
    # ★★★★★★★★★★

    labels_list = [] # 正解ラベルをここに蓄積する
    frame_count = 0
    annotated_frame_count = 0

    print("--- アノテーションを開始します ---")
    print(f"{SKIP_FRAMES}フレームごとに一時停止します。")
    print(f"座席が「在席=1」「空席=0」として、{NUM_SEATS}席分の状態をカンマ区切りで入力してください。")
    print(f"例: 1,0,0,1,0,0")
    print("途中で終了する場合は、ターミナルで 'q' と入力してください。")
    print("-" * 30)

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("動画の終わりに達しました。")
            break
            
        if frame_count % SKIP_FRAMES == 0:
            annotated_frame_count += 1
            
            display_text = f"Frame: {frame_count} (Annotation #{annotated_frame_count})"
            cv2.putText(frame, display_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
           


            cv2.imshow(WINDOW_NAME, frame)
            
            # input()で停止する前に、ウィンドウが描画・応答できるように
            # 1ミリ秒だけ待機（イベント処理）を行う
            cv2.waitKey(1)
            
            # --- ラベル入力 ---
            while True:
                # ターミナルからの入力を待つ
                user_input = input(f"[Frame {frame_count}] {NUM_SEATS}席の状態を入力 (qで終了): ")
                
                if user_input.lower() == 'q':
                    print("アノテーションを中断します。")
                    ret = False 
                    break
                
                parsed_labels = parse_label_input(user_input, NUM_SEATS)
                
                if parsed_labels is not None:
                    labels_list.append(parsed_labels)
                    break 
                else:
                    print("  [再試行] もう一度入力してください。")
            
            if not ret:
                break

        frame_count += 1
        
        # このwaitKeyは、SKIP_FRAMESの間を早送り再生するために必要
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("ウィンドウが閉じられたため終了します。")
            break

    # --- 終了処理 ---
    cap.release()
    cv2.destroyAllWindows()

    if not labels_list:
        print("アノテーションデータがありません。ファイルは保存されませんでした。")
        return

    # PythonリストをNumpy配列に変換
    # 形状は (アノテーションしたフレーム数, 座席数) になる
    final_labels_np = np.array(labels_list)
    
    # Numpyファイル (.npy) として保存
    np.save(OUTPUT_FILE, final_labels_np)
    
    print("-" * 30)
    print("アノテーションが完了しました。")
    print(f"ファイル '{OUTPUT_FILE}' として保存しました。")
    print(f"保存されたラベルの形状: {final_labels_np.shape}")
    print(f"（{final_labels_np.shape[0]} フレーム分, {final_labels_np.shape[1]} 席）")

if __name__ == "__main__":
    main()