import cv2
import mediapipe as mp
# 初始化 MediaPipe 手部偵測
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # "CPU 部分：MediaPipe 手部偵測"
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result_hands = hands.process(frame_rgb)

    keypoints = []

    if result_hands.multi_hand_landmarks:
        for hand_landmarks in result_hands.multi_hand_landmarks:
            # "繪製手部關鍵點"
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # "紀錄指定關鍵點的位置"
            keypoints.append([
                hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y,  # 拇指尖端
                hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y,  # 食指尖端
                hand_landmarks.landmark[12].x, hand_landmarks.landmark[12].y  # 中指尖端
            ])

    cv2.imshow('Hand Pose', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
cap.release()
cv2.destroyAllWindows()