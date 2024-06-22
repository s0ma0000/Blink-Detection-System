import numpy as np  # 数値計算
from collections import deque  # 効率的なキュー操作
from scipy.fftpack import fft, fftfreq  # フーリエ変換
import cv2  # 画像処理とコンピュータビジョン
import mediapipe as mp  # 顔のランドマーク検出

# データの可視化
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ランドマーク間の横方向の距離を計算する関数
def calculate_horizontal_distance(landmark1, landmark2, frame_width):
    x1 = int(landmark1.x * frame_width)
    x2 = int(landmark2.x * frame_width)
    return abs(x1 - x2)

# ランドマーク間の縦方向の距離を計算する関数
def calculate_vertical_distance(landmark1, landmark2, frame_height):
    y1 = int(landmark1.y * frame_height)
    y2 = int(landmark2.y * frame_height)
    return abs(y1 - y2)

# Figure の閉じるボタンがクリックされたときのコールバック関数
def on_figure_close(event):
    global running
    running = False

# カメラの設定: カメラのキャプチャを開始する
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("カメラが開けません")
    exit()

# 保存する動画の設定
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('dataset/output.mp4', fourcc, 20.0, (640, 480))  # 出力ファイル名とフレームサイズを指定

# MediaPipeのFace Meshモデルを初期化する．顔の高精細なランドマークを検出するために refine_landmarks を True に設定
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# 虹彩のX座標を格納するリスト
right_iris_x_coords = []
left_iris_x_coords = []

# 虹彩の相対的な位置を格納するデックの最大長を指定
N = 50

# 虹彩の相対的な位置を格納するデックを作成
right_iris_relative_pos = deque(maxlen=N)
left_iris_relative_pos = deque(maxlen=N)

# まばたき検出用の閾値とカウンター
blink_threshold = 5  # 縦の距離がこれ以下になった場合にまばたきと判断
right_blink_counter = 0
left_blink_counter = 0
right_blink_count = 0
left_blink_count = 0

# インタラクティブモードをオンにする
plt.ion()

# 虹彩のX座標をプロットするためのグラフを設定
fig_iris, (ax_iris_pos, ax_iris_fft) = plt.subplots(2, 1, figsize=(8, 6))
line_right_iris, = ax_iris_pos.plot(right_iris_relative_pos, label='Right Iris Position')
line_left_iris, = ax_iris_pos.plot(left_iris_relative_pos, label='Left Iris Position')
ax_iris_pos.legend()
ax_iris_pos.set_title('Relative Iris Position Over Time')
ax_iris_pos.set_xlabel('Frame')
ax_iris_pos.set_ylabel('Relative Position')
ax_iris_pos.set_ylim(0, 1)  # 縦軸の範囲を0から1に設定
ax_iris_pos.set_yticks(np.arange(0, 1.1, 0.1))  # 縦軸の目盛りを0.1刻みで設定

ax_iris_fft.set_title('Fourier Transform of Iris Position')
ax_iris_fft.set_xlabel('Frequency')
ax_iris_fft.set_ylabel('Amplitude')

# サブプロット間の隙間を調整
fig_iris.subplots_adjust(hspace=0.5)  # hspace はサブプロット間の垂直方向のスペースを指定

# matplotlibの設定
fig, ax = plt.subplots()  # プロット用のウィンドウを作成
scat = ax.scatter([], [], color='blue', s=1)  # 散布図の初期設定
ax.set_xlim(0, 480)  # X軸のリミットを設定（カメラの解像度に応じて調整）
ax.set_ylim(480, 0)  # Y軸のリミットを設定（カメラの解像度に応じて調整）
ax.invert_yaxis()  # Y軸を反転
ax.set_aspect('equal')  # 縦横比を1:1に設定

# 虹彩の円のパッチを初期化
right_iris_circle = Circle((0, 0), 0, color='red', fill=False)
left_iris_circle = Circle((0, 0), 0, color='blue', fill=False)
ax.add_patch(right_iris_circle)
ax.add_patch(left_iris_circle)

# ウィンドウ名
window_name = 'Eye Center and Landmarks Visualization'
cv2.namedWindow(window_name)

# Figure の閉じるボタンがクリックされたときのイベントを設定
fig.canvas.mpl_connect('close_event', on_figure_close)
fig_iris.canvas.mpl_connect('close_event', on_figure_close)

running = True

# 無限ループを開始してカメラからフレームを継続的に取得する
while running:
    # カメラからフレームを読み込む
    ret, frame = cap.read()
    if not ret:
        break
    # フレームを左右反転させ、自然な鏡のようなビューを提供する
    frame = cv2.flip(frame, 1)
    # フレームのカラーフォーマットをBGRからRGBに変換する．MediaPipeではRGB形式が必要
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Face Meshを実行して、結果を取得する
    output = face_mesh.process(rgb_frame)
    # 検出された顔のランドマークを取得する
    landmark_points = output.multi_face_landmarks

    # フレームの高さと幅を取得する
    frame_h, frame_w, _ = frame.shape
    
    if landmark_points:
        # 最初の(一人目の）顔のランドマークを取得する
        landmarks = landmark_points[0].landmark

        # 右目の周りのランドマークを可視化する
        right_eye_indexes = [33, 133, 160, 159, 158, 157, 173, 246, 7, 163, 144, 145, 153, 154, 155]
        for index in right_eye_indexes:
            landmark = landmarks[index]
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 2, (0, 140, 255), -1)
            # 橙色で右目のランドマークを表示
        
        # 右目の中心点を計算
        right_eye_x = sum([landmarks[idx].x for idx in right_eye_indexes]) / len(right_eye_indexes)
        right_eye_y = sum([landmarks[idx].y for idx in right_eye_indexes]) / len(right_eye_indexes)
        right_eye_x = int(right_eye_x * frame_w)
        right_eye_y = int(right_eye_y * frame_h)

        # 左目の周りのランドマークを可視化する
        left_eye_indexes = [362, 263, 387, 386, 385, 384, 398, 466, 249, 390, 373, 374, 380, 381, 382]
        for index in left_eye_indexes:
            landmark = landmarks[index]
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 2, (0, 140, 255), -1)
            # 橙色で左目のランドマークを表示

        # 左目の中心点を計算
        left_eye_x = sum([landmarks[idx].x for idx in left_eye_indexes]) / len(left_eye_indexes)
        left_eye_y = sum([landmarks[idx].y for idx in left_eye_indexes]) / len(left_eye_indexes)
        left_eye_x = int(left_eye_x * frame_w)
        left_eye_y = int(left_eye_y * frame_h)
        
        # 右目の虹彩の中心を表示
        right_iris_center = landmarks[468]
        right_iris_center_x = int(right_iris_center.x * frame_w)
        right_iris_center_y = int(right_iris_center.y * frame_h)
        cv2.circle(frame, (right_iris_center_x, right_iris_center_y), 5, (0, 0, 255), -1)  # 赤色

        # 右目の虹彩の半径を計算
        right_iris_left = landmarks[469]
        right_iris_right = landmarks[471]
        right_iris_radius = int(calculate_horizontal_distance(right_iris_left, right_iris_right, frame_w) / 2)

        # 右目の虹彩の円を描画
        cv2.circle(frame, (right_iris_center_x, right_iris_center_y), right_iris_radius, (0, 0, 255), 1)  # 赤色の円

        # 左目の虹彩の中心を取得
        left_iris_center = landmarks[473]
        left_iris_center_x = int(left_iris_center.x * frame_w)
        left_iris_center_y = int(left_iris_center.y * frame_h)

        # 左目の虹彩の半径を計算
        left_iris_left = landmarks[474]
        left_iris_right = landmarks[476]
        left_iris_radius = int(calculate_horizontal_distance(left_iris_left, left_iris_right, frame_w) / 2)
        cv2.circle(frame, (left_iris_center_x, left_iris_center_y), 5, (0, 0, 255), -1)  # 赤色

        # 左目の虹彩の円を描画
        cv2.circle(frame, (left_iris_center_x, left_iris_center_y), left_iris_radius, (255, 0, 0), 1)  # 青色の円

        # ランドマークデータを配列に格納
        x_coords = [landmark.x * frame_w for landmark in landmarks]
        y_coords = [(1 - landmark.y) * frame_h for landmark in landmarks]  # Y座標の反転を調整

        # プロット領域の調整
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        buffer = 50  # バッファを追加して顔の周囲に少し余裕を持たせる
        ax.set_xlim(x_min - buffer, x_max + buffer)
        ax.set_ylim(y_min - buffer, y_max + buffer)  # Y軸の範囲設定を反転

        # 虹彩の円のパッチを更新
        right_iris_circle.center = (right_iris_center_x, frame_h - right_iris_center_y)
        right_iris_circle.radius = right_iris_radius
        left_iris_circle.center = (left_iris_center_x, frame_h - left_iris_center_y)
        left_iris_circle.radius = left_iris_radius

        # 散布図のデータを更新
        scat.set_offsets(np.c_[x_coords, y_coords])
        fig.canvas.draw_idle()

        # 右目の虹彩の相対的な位置を計算
        right_eye_left_x = int(landmarks[33].x * frame_w)
        right_eye_right_x = int(landmarks[133].x * frame_w)
        right_iris_relative_pos.append((right_iris_center_x - right_eye_left_x) / (right_eye_right_x - right_eye_left_x))

        # 左目の虹彩の相対的な位置を計算
        left_eye_left_x = int(landmarks[362].x * frame_w)
        left_eye_right_x = int(landmarks[263].x * frame_w)
        left_iris_relative_pos.append((left_iris_center_x - left_eye_left_x) / (left_eye_right_x - left_eye_left_x))

        # プロットデータの更新
        line_right_iris.set_data(range(len(right_iris_relative_pos)), right_iris_relative_pos)
        line_left_iris.set_data(range(len(left_iris_relative_pos)), left_iris_relative_pos)

        # 軸の範囲を調整
        if len(right_iris_relative_pos) > N:
            ax_iris_pos.set_xlim(len(right_iris_relative_pos) - N, len(right_iris_relative_pos))
        else:
            ax_iris_pos.set_xlim(0, N)

        # フーリエ変換を適用 (毎フレーム更新)
        if len(right_iris_relative_pos) == N:
            right_fft = np.fft.fft(list(right_iris_relative_pos))
            left_fft = np.fft.fft(list(left_iris_relative_pos))
            freq = np.fft.fftfreq(N, d=1)  # `d` はサンプリング間隔、ここでは1フレームごと

            ax_iris_fft.clear()
            # FFT の結果は N の半分までが有用（ナイキスト周波数まで）
            freq_half = freq[:N//2]
            right_fft_half = np.abs(right_fft)[:N//2]
            left_fft_half = np.abs(left_fft)[:N//2]

            ax_iris_fft.plot(freq_half, right_fft_half, label='Right Iris FFT')
            ax_iris_fft.plot(freq_half, left_fft_half, label='Left Iris FFT')
            ax_iris_fft.legend()
            ax_iris_fft.set_title('Fourier Transform of Iris Position')
            ax_iris_fft.set_xlabel('Frequency')
            ax_iris_fft.set_ylabel('Amplitude')

            # 縦軸の範囲を調整
            max_amplitude = max(max(right_fft_half), max(left_fft_half))
            ax_iris_fft.set_ylim(0, max_amplitude * 0.1)

            fig_iris.canvas.draw_idle()

        # まばたきの検出
        right_eye_top = landmarks[159]
        right_eye_bottom = landmarks[145]
        left_eye_top = landmarks[386]
        left_eye_bottom = landmarks[374]

        right_eye_height = calculate_vertical_distance(right_eye_top, right_eye_bottom, frame_h)
        left_eye_height = calculate_vertical_distance(left_eye_top, left_eye_bottom, frame_h)

        # まばたき判定
        if right_eye_height < blink_threshold:
            right_blink_counter += 1
        else:
            if right_blink_counter > 0:
                right_blink_count += 1
                right_blink_counter = 0
        
        if left_eye_height < blink_threshold:
            left_blink_counter += 1
        else:
            if left_blink_counter > 0:
                left_blink_count += 1
                left_blink_counter = 0

        # まばたきのカウントを表示
        cv2.putText(frame, f"Right Blinks: {right_blink_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Left Blinks: {left_blink_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # グラフの再描画
        fig_iris.canvas.draw_idle()

    # フレームを保存
    out.write(frame)

    # ウィンドウにフレームを表示する
    cv2.imshow(window_name, frame)

    # 'Esc'キーまたは'Q'キーが押されたらループから抜けてプログラムを終了する
    if cv2.waitKey(1) & 0xFF in [27, ord('q')] or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        running = False

# カメラのリソースを解放し、オープンしたウィンドウを閉じる
cap.release()
out.release()  # 保存ファイルを閉じる
cv2.destroyAllWindows()
plt.close('all')
