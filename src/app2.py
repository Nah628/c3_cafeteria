import cv2
import numpy as np
import threading
import time
import json
import os
import sys
from flask import Flask, jsonify, Response, render_template, request 
from ultralytics import YOLO

# --- Global variables and initial settings ---
app = Flask(__name__)

# YOLO model
model = YOLO("yolo11n.pt") 

# Video source (Webcam: 0, Video file: "video.mp4", etc.)
CAMERA_SOURCE = "ue.MOV"

# 空席判定占有率
occupied_rate = 0.5

# Configurationファイル名
CONFIG_FILE_PATH = "seats_config.json"

def load_seat_configurations_initial(config_file=CONFIG_FILE_PATH):
    if not os.path.exists(config_file):
        print(f"警告: 設定ファイル '{config_file}' が見つかりません。新規作成されます。", file=sys.stderr)
        return {}

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Check if 'seat_definitions' key exists and is a dictionary
        if "seat_definitions" not in config or not isinstance(config["seat_definitions"], dict):
            raise ValueError("JSONファイルに 'seat_definitions' キーが存在しないか、形式が不正です。")

        # Check if coordinates for each seat are in the correct list format
        for seat_id, coords in config["seat_definitions"].items():
            if not (isinstance(coords, list) and len(coords) == 4 and all(isinstance(c, (int, float)) for c in coords)):
                raise ValueError(f"座席 '{seat_id}' の座標形式が不正です。 [x1, y1, x2, y2] の形式である必要があります。")

        print(f"設定ファイル '{config_file}' を正常に読み込みました。")
        return config["seat_definitions"]
    
    except json.JSONDecodeError as e:
        print(f"エラー: 設定ファイル '{config_file}' の読み込み中にJSONの解析エラーが発生しました: {e}", file=sys.stderr)
        print("ファイルの内容が正しいJSON形式であることを確認してください。", file=sys.stderr)
        return {} # On error, return empty dict
    except ValueError as e:
        print(f"エラー: 設定ファイル '{config_file}' の内容が不正です: {e}", file=sys.stderr)
        return {} # On error, return empty dict
    except Exception as e:
        print(f"エラー: 設定ファイル '{config_file}' の読み込み中に予期せぬエラーが発生しました: {e}", file=sys.stderr)
        return {} # On error, return empty dict

# ここで読み込む
seat_definitions_with_coords = load_seat_configurations_initial()

# 第一フレーム
first_frame_cached = None
first_frame_lock = threading.Lock()

# グローバル変数として最新のpersonボックスリストを保持
latest_person_boxes = []
boxes_lock = threading.Lock()

# APIで返す用の最終判定ステータス
current_seat_statuses = {seat_id: "unknown" for seat_id in seat_definitions_with_coords.keys()}

# 5秒間分の判定結果を蓄積する履歴辞書
seat_status_history = {seat_id: [] for seat_id in seat_definitions_with_coords.keys()}

# フレームごとの暫定判定（毎フレーム処理中に更新）
internal_temp_statuses = {seat_id: "unknown" for seat_id in seat_definitions_with_coords.keys()}

status_lock = threading.Lock()

# 物体検出ループ内で毎フレームの判定を更新
def detect_and_update_seats_thread():
    global internal_temp_statuses, seat_status_history
    while True:
        # 検出結果取得
        results = model(img_for_detection, verbose=False, imgsz=640)
        boxes = results[0].boxes
        names = results[0].names

        #temp_statuses = {seat_id: "empty" for seat_id in seat_definitions_with_coords.keys()}
        temp_statuses = {seat_id: "empty" for seat_id in current_defs.keys()}
        person_boxes_tmp = []

        for box_idx, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            label = names[cls_id]
            if label != "person":
                continue

        # 判定結果を暫定的に保存
        with status_lock:
            internal_temp_statuses.update(temp_statuses)
            # 履歴にも追加（5秒間分集計用）
            for seat_id, status in temp_statuses.items():
                if seat_id not in seat_status_history:
                    seat_status_history[seat_id] = []
                seat_status_history[seat_id].append(status)

            # バウンディングボックスの座標(フル解像度)
            px1, py1, px2, py2 = map(int, box.xyxy[0].cpu().numpy())
            person_boxes_tmp.append([px1, py1, px2, py2])

        # 最新のperson_boxesをスレッドロック付きで更新
        with boxes_lock:
            latest_person_boxes = person_boxes_tmp

# 5秒ごとに履歴を集計してcurrent_seat_statusesに反映
def update_current_seat_statuses_every_interval(interval_sec = 5):
    global current_seat_statuses, seat_status_history
    while True:
        time.sleep(interval_sec)
        with status_lock:
            for seat_id, history in seat_status_history.items():
                empty_count = history.count('empty')
                occupied_count = history.count('occupied')
                if empty_count > occupied_count:
                    current_seat_statuses[seat_id] = 'empty'
                elif occupied_count > empty_count:
                    current_seat_statuses[seat_id] = 'occupied'
                else:
                    # 同数の場合は、前回状態を維持
                    current_seat_statuses[seat_id] = current_seat_statuses.get(seat_id, 'unknown')
                
                # 履歴をクリアして次周期へ
                seat_status_history[seat_id] = []


# 初めの1フレーム取得 
def get_first_frame():
    global first_frame_cached
    with first_frame_lock:
        if first_frame_cached is not None:
            return first_frame_cached

        cap = cv2.VideoCapture(CAMERA_SOURCE)
        if not cap.isOpened():
            print(f"エラー: ビデオソース {CAMERA_SOURCE} を開けませんでした。", file=sys.stderr)
            return None

        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("エラー: 最初のフレームを読み込めませんでした。", file=sys.stderr)
            return None

        height, width, _ = frame.shape
        print(f"DEBUG: First frame resolution: Width={width}, Height={height}", file=sys.stdout) 

        first_frame_cached = frame.copy()
        return first_frame_cached

# Configファイルから取得
"""
#############
jsonファイルのフォーマット
返り値{"seat_id":[座席座標]}
#############
{
    "seat_definitions": {
        "seat_1": [
            x1,
            y1,
            x2,
            y2
        ],
        "seat_2": [
            x1,
            y1,
            x2,
            y2
        ]
    }
}
"""
def load_seat_configurations_initial(config_file=CONFIG_FILE_PATH):
    if not os.path.exists(config_file):
        print(f"警告: 設定ファイル '{config_file}' が見つかりません。新規作成されます。", file=sys.stderr)
        return {}

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Check if 'seat_definitions' key exists and is a dictionary
        if "seat_definitions" not in config or not isinstance(config["seat_definitions"], dict):
            raise ValueError("JSONファイルに 'seat_definitions' キーが存在しないか、形式が不正です。")

        # Check if coordinates for each seat are in the correct list format
        for seat_id, coords in config["seat_definitions"].items():
            if not (isinstance(coords, list) and len(coords) == 4 and all(isinstance(c, (int, float)) for c in coords)):
                raise ValueError(f"座席 '{seat_id}' の座標形式が不正です。 [x1, y1, x2, y2] の形式である必要があります。")

        print(f"設定ファイル '{config_file}' を正常に読み込みました。")
        return config["seat_definitions"]
    
    except json.JSONDecodeError as e:
        print(f"エラー: 設定ファイル '{config_file}' の読み込み中にJSONの解析エラーが発生しました: {e}", file=sys.stderr)
        print("ファイルの内容が正しいJSON形式であることを確認してください。", file=sys.stderr)
        return {} # On error, return empty dict
    except ValueError as e:
        print(f"エラー: 設定ファイル '{config_file}' の内容が不正です: {e}", file=sys.stderr)
        return {} # On error, return empty dict
    except Exception as e:
        print(f"エラー: 設定ファイル '{config_file}' の読み込み中に予期せぬエラーが発生しました: {e}", file=sys.stderr)
        return {} # On error, return empty dict

# Configファイル読み込み
seat_definitions_with_coords = load_seat_configurations_initial()

# 空席判定用
# カウンターを最初に初期化しておく（グローバルで）
seat_stay_counters = {seat_id: 0 for seat_id in seat_definitions_with_coords.keys()}
seat_leave_counters = {seat_id: 0 for seat_id in seat_definitions_with_coords.keys()}

STAY_THRESHOLD = 50   # 何フレーム以上重なっていれば「着席」
LEAVE_THRESHOLD = 50  # 何フレーム以上重ならなければ「空席」

##########################
### Webからのリクエストで返す値
"""
current_seat_statuses: {"seat_id":座席状態(unknown,occupied,empty)}
Webページからアクセスした時に返す


last_frame:
numpyの配列
Webページからアクセスした時にlast_frameをバッファしてリクエストされたときに動画として返す
"""

# 現在の空席情報
# 複数スレッドから競合をブロック
current_seat_statuses = {seat_id: "unknown" for seat_id in seat_definitions_with_coords.keys()}
status_lock = threading.Lock()

# フレームバッファ
# 複数スレッドから競合をブロック
last_frame = None
frame_lock = threading.Lock()

# 物体検出と空席判定関数(スレッドで実行)
def detect_and_update_seats_thread():
    global latest_person_boxes

    cap = None
    TARGET_FPS = 5
    TARGET_FRAME_TIME = 1.0 / TARGET_FPS

    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    logged_frame_size = False

    while True:
        start_loop_time = time.perf_counter()

        if cap is None or not cap.isOpened():
            print(f"ビデオソースを開こうとしています: {CAMERA_SOURCE}")
            cap = cv2.VideoCapture(CAMERA_SOURCE)
            if not cap.isOpened():
                print(f"エラー: ビデオソース {CAMERA_SOURCE} を開けませんでした。5秒後に再試行します...")
                time.sleep(5)
                continue
            else:
                print(f"ビデオソース {CAMERA_SOURCE} を正常に開きました")

        ret, frame = cap.read()
        if not ret:
            print("フレームの取得に失敗しました。カメラを解放して再初期化します...")
            cap.release()
            cap = None
            time.sleep(1)
            continue

        if not logged_frame_size:
            h, w, _ = frame.shape
            print(f"DEBUG: YOLO input frame size: width={w}, height={h}")
            logged_frame_size = True

        # YOLO処理用に縮小
        img_for_detection = cv2.resize(frame, (960, 540))

        # フル → 処理解像度へスケーリング比
        scale_x = 960 / 1280
        scale_y = 540 / 720

        # スレッド競合防止しつつ座席定義読み出し
        with status_lock:
            current_defs = seat_definitions_with_coords.copy()
            temp_statuses = {seat_id: "empty" for seat_id in current_defs.keys()}

        # スケーリングされた座席定義
        scaled_defs = {
            seat_id: [
                int(coords[0] * scale_x),
                int(coords[1] * scale_y),
                int(coords[2] * scale_x),
                int(coords[3] * scale_y)
            ]
            for seat_id, coords in current_defs.items()
        }

        results = model(img_for_detection, verbose=False, imgsz=640)
        boxes = results[0].boxes
        names = results[0].names

        person_boxes_tmp = []

        for box_idx, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            label = names[cls_id]
            if label != "person":
                continue

            px1, py1, px2, py2 = map(int, box.xyxy[0].cpu().numpy())
            person_boxes_tmp.append([px1, py1, px2, py2])

        # 座席ごとの判定ループ（ここが人物検出の外側）
        for seat_id_key, seat_coords in scaled_defs.items():
            sx1, sy1, sx2, sy2 = seat_coords

            is_overlapping = False

            for box in person_boxes_tmp:
                px1, py1, px2, py2 = box

                ix1 = max(sx1, px1)
                iy1 = max(sy1, py1)
                ix2 = min(sx2, px2)
                iy2 = min(sy2, py2)

                inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                seat_area = (sx2 - sx1) * (sy2 - sy1)

                overlap_ratio = inter_area / seat_area if seat_area > 0 else 0

                if overlap_ratio > occupied_rate:
                    is_overlapping = True
                    break

            if is_overlapping:
                seat_stay_counters[seat_id_key] += 1
                seat_leave_counters[seat_id_key] = 0
            else:
                seat_stay_counters[seat_id_key] = 0
                seat_leave_counters[seat_id_key] += 1

            # 状態更新判定
            if seat_stay_counters[seat_id_key] >= STAY_THRESHOLD:
                temp_statuses[seat_id_key] = "occupied"
            elif seat_leave_counters[seat_id_key] >= LEAVE_THRESHOLD:
                temp_statuses[seat_id_key] = "empty"
            else:
                temp_statuses[seat_id_key] = current_seat_statuses.get(seat_id_key, "unknown")

        with boxes_lock:
            latest_person_boxes = person_boxes_tmp

        with status_lock:
            current_seat_statuses.update(temp_statuses)

        # 描画用（元フル解像度 frame に対して scaled_defs を逆変換して描く）
        with frame_lock:
            global last_frame

            for seat_id, scaled_coords in scaled_defs.items():
                sx1 = int(scaled_coords[0] / scale_x)
                sy1 = int(scaled_coords[1] / scale_y)
                sx2 = int(scaled_coords[2] / scale_x)
                sy2 = int(scaled_coords[3] / scale_y)
                status = current_seat_statuses.get(seat_id, "unknown")

                color = (0, 255, 0) if status == "empty" else (0, 0, 255)
                thickness = 2 if status == "occupied" else 1

                cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), color, thickness)
                cv2.putText(frame, f"{seat_id}: {status}", (sx1 + 5, sy1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            for i, box in enumerate(person_boxes_tmp):
                px1 = int(box[0] / scale_x)
                py1 = int(box[1] / scale_y)
                px2 = int(box[2] / scale_x)
                py2 = int(box[3] / scale_y)
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {i+1}", (px1, py1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            last_frame = frame.copy()

        end_loop_time = time.perf_counter()
        processing_duration = end_loop_time - start_loop_time
        sleep_duration = TARGET_FRAME_TIME - processing_duration
        if sleep_duration > 0:
            time.sleep(sleep_duration)

    cap.release()




# Flaskのルート定義

@app.route('/api/person_boxes')
def get_person_boxes():
    with boxes_lock:
        return jsonify(latest_person_boxes)


# 空席情報を返すエンドポイント
@app.route('/api/seat_status')
def get_seat_status():
    with status_lock:
        return jsonify(current_seat_statuses)

# ライブ映像を生成する
def generate_frames():
    while True:
        with frame_lock:
            if last_frame is None:
                # Skip if frame is not yet available
                time.sleep(0.1)
                continue
            frame = last_frame.copy()

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# generate_frameに従ってライブ映像を返す
@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# メインWebページを返すルート
@app.route('/')
def index():
    # render_template looks for files in the 'templates' folder
    return render_template('index.html')

# 座席座標を指定するためのルート
@app.route('/define_seats_web')
def define_seats_web():
    """座席定義用のウェブUIを表示するルート"""
    return render_template('define_seats_web.html')

# New API endpoint to save seat definitions from the web page
@app.route('/api/save_seat_definitions', methods=['POST'])
def save_seat_definitions():
    data = request.get_json() # Get JSON data from request body
    if not data:
        return jsonify({"success": False, "message": "リクエストボディにJSONデータがありません。"}), 400

    new_seat_definitions = data.get("seat_definitions")

    if not isinstance(new_seat_definitions, dict):
        return jsonify({"success": False, "message": "無効なデータ形式です。'seat_definitions'は辞書である必要があります。"}), 400

    # Validate the format of received seat definitions
    validated_definitions = {}
    for seat_id, coords in new_seat_definitions.items():
        if not (isinstance(coords, list) and len(coords) == 4 and all(isinstance(c, (int, float)) for c in coords)):
            return jsonify({"success": False, "message": f"座席 '{seat_id}' の座標形式が不正です。 [x1, y1, x2, y2] の形式である必要があります。"}), 400
        validated_definitions[seat_id] = coords

    # Save to JSON file
    try:
        # Update seat definitions thread-safely
        with status_lock: # Use status_lock to protect seat definition updates as well
            global seat_definitions_with_coords # Update global variable
            seat_definitions_with_coords = validated_definitions
            # Reset current_seat_statuses based on the newly defined seats
            current_seat_statuses.clear()
            for seat_id in seat_definitions_with_coords.keys():
                current_seat_statuses[seat_id] = "unknown"

            config_data = {
                "seat_definitions": validated_definitions
            }
            with open(CONFIG_FILE_PATH, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=4, ensure_ascii=False)

        print(f"座席定義を '{CONFIG_FILE_PATH}' に更新しました。")
        return jsonify({"success": True, "message": "座席定義を正常に保存しました。"}), 200
    except Exception as e:
        print(f"エラー: 座席定義の保存中にエラーが発生しました: {e}", file=sys.stderr)
        return jsonify({"success": False, "message": f"座席定義の保存中にエラーが発生しました: {str(e)}"}), 500

# 現愛の座席座標を返すエンドポイント
@app.route('/api/get_all_seat_coords')
def get_all_seat_coords():
    """現在の座席定義（座標を含む）をJSON形式で返す"""
    return jsonify({"seat_definitions": seat_definitions_with_coords})

# 映像の1フレームを返すエンドポイント
@app.route('/first_frame_feed')
def first_frame_feed():
    """動画の最初のフレームをJPEG画像として提供するルート（解像度縮小付き）"""
    frame = get_first_frame()  # 元の高解像度フレームを取得
    if frame is None:
        return Response("Error: Could not load first frame", status=500, mimetype='text/plain')

    # 解像度を960x540に縮小（define_seats_web.html と一致）
    resized_frame = cv2.resize(frame, (960, 540))  # 幅×高さ
    ret, buffer = cv2.imencode('.jpg', resized_frame)
    frame_bytes = buffer.tobytes()
    return Response(frame_bytes, mimetype='image/jpeg')


# --- Application startup ---
if __name__ == '__main__':
    print("最初のフレームを事前にロードしています...")
    get_first_frame() 
    if first_frame_cached is None:
        print("警告: 最初のフレームのロードに失敗しました。定義ツールで問題が発生する可能性があります。", file=sys.stderr)
    else:
        print("最初のフレームのロードが完了しました。")

    # YOLOの物体検出スレッドの実行
    detector_thread = threading.Thread(target=detect_and_update_seats_thread)
    detector_thread.daemon = True
    detector_thread.start()

    # 5秒周期で集計スレッド開始
    updater_thread = threading.Thread(target=update_current_seat_statuses_every_interval, args=(5,))
    updater_thread.daemon = True
    updater_thread.start()

    # Start Flask application
    # host='0.0.0.0' allows external access
    # debug=True for development, set to False in production (affects performance)
    # threaded=True enables concurrent connections in development server
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
