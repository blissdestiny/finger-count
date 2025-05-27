import cv2
import mediapipe as mp
import numpy as np

# Инициализация медиапайп для распознавания рук
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

def count_fingers(landmarks):
    # Определяем, сколько пальцев поднято
    thumb_up = landmarks[mp_hands.HandLandmark.THUMB_TIP].y < landmarks[mp_hands.HandLandmark.THUMB_IP].y
    index_finger_up = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    middle_finger_up = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
    ring_finger_up = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y
    pinky_finger_up = landmarks[mp_hands.HandLandmark.PINKY_TIP].y < landmarks[mp_hands.HandLandmark.PINKY_PIP].y

    return sum([thumb_up, index_finger_up, middle_finger_up, ring_finger_up, pinky_finger_up])

def is_fist(landmarks):
    # Проверка сжатого кулака (если все пальцы близки друг к другу)
    distances = [
        np.linalg.norm(np.array([landmarks[mp_hands.HandLandmark.THUMB_TIP].x, landmarks[mp_hands.HandLandmark.THUMB_TIP].y]) - np.array([landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x, landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y])),
        np.linalg.norm(np.array([landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x, landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y]) - np.array([landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x, landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y])),
        np.linalg.norm(np.array([landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x, landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y]) - np.array([landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].x, landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y])),
        np.linalg.norm(np.array([landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].x, landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y]) - np.array([landmarks[mp_hands.HandLandmark.PINKY_TIP].x, landmarks[mp_hands.HandLandmark.PINKY_TIP].y])),
    ]

    return all(distance < 0.05 for distance in distances)  # Порог для сжатого кулака

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Обработка изображения
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    total_fingers = 0
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Рисование ключевых точек
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            landmarks = hand_landmarks.landmark
            num_fingers = count_fingers(landmarks)
            
            # Проверка сжатого кулака
            if is_fist(landmarks):
                num_fingers = 1

            total_fingers += num_fingers

    # Убедитесь, что общее количество пальцев не превышает 10
    total_fingers = min(total_fingers, 10)

    # Вывод числа пальцев на экран
    cv2.putText(frame, f'Fingers: {total_fingers}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Показываем изображение
    cv2.imshow('Finger Count', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Нажмите 'Esc' для выхода
        break

cap.release()
cv2.destroyAllWindows()