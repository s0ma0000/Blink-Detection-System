import numpy as np
import cv2
import mediapipe as mp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# MediaPipeの初期化
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# カメラの初期化
cap = cv2.VideoCapture(0)

# 3Dデータを保存するためのリスト
frames = []

# プロットの初期化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# リアルタイムでフレームを処理
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 3D座標を取得
            landmarks = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark])
            frames.append(landmarks)
            
            # 3Dプロット
            ax.clear()
            ax.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2])
            plt.draw()
            plt.pause(0.001)
    
    cv2.imshow('Face Mesh', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# 3DデータをNumPy配列に変換して保存
frames = np.array(frames)
np.save('4d_face_data.npy', frames)

# 4Dデータの再生
loaded_frames = np.load('4d_face_data.npy')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def update(num, data, plot):
    plot._offsets3d = (data[num][:, 0], data[num][:, 1], data[num][:, 2])

ani = FuncAnimation(fig, update, frames=len(loaded_frames), fargs=(loaded_frames, ax), interval=50)
plt.show()
