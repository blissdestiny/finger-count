import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
detector = mp_pose.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

video = cv2.VideoCapture(0)

def check_fingers(points, hand_type):
    count = 0
    right_hand = (hand_type == "Right")

    thumb_end = points[mp_pose.HandLandmark.THUMB_TIP]
    thumb_base = points[mp_pose.HandLandmark.THUMB_MCP]
    if right_hand:
        thumb_raised = thumb_end.x < thumb_base.x - 0.05
    else:
        thumb_raised = thumb_end.x > thumb_base.x + 0.05
    if thumb_raised:
        count += 1

    tips = [
        mp_pose.HandLandmark.INDEX_FINGER_TIP,
        mp_pose.HandLandmark.MIDDLE_FINGER_TIP,
        mp_pose.HandLandmark.RING_FINGER_TIP,
        mp_pose.HandLandmark.PINKY_TIP
    ]
    joints = [
        mp_pose.HandLandmark.INDEX_FINGER_PIP,
        mp_pose.HandLandmark.MIDDLE_FINGER_PIP,
        mp_pose.HandLandmark.RING_FINGER_PIP,
        mp_pose.HandLandmark.PINKY_PIP
    ]

    for tip, joint in zip(tips, joints):
        if points[tip].y < points[joint].y:
            count += 1

    return count

def check_closed(points):
    distances = [
        np.linalg.norm([points[mp_pose.HandLandmark.INDEX_FINGER_TIP].x - points[mp_pose.HandLandmark.INDEX_FINGER_MCP].x,
                        points[mp_pose.HandLandmark.INDEX_FINGER_TIP].y - points[mp_pose.HandLandmark.INDEX_FINGER_MCP].y]),
        np.linalg.norm([points[mp_pose.HandLandmark.MIDDLE_FINGER_TIP].x - points[mp_pose.HandLandmark.MIDDLE_FINGER_MCP].x,
                        points[mp_pose.HandLandmark.MIDDLE_FINGER_TIP].y - points[mp_pose.HandLandmark.MIDDLE_FINGER_MCP].y]),
        np.linalg.norm([points[mp_pose.HandLandmark.RING_FINGER_TIP].x - points[mp_pose.HandLandmark.RING_FINGER_MCP].x,
                        points[mp_pose.HandLandmark.RING_FINGER_TIP].y - points[mp_pose.HandLandmark.RING_FINGER_MCP].y]),
        np.linalg.norm([points[mp_pose.HandLandmark.PINKY_TIP].x - points[mp_pose.HandLandmark.PINKY_MCP].x,
                        points[mp_pose.HandLandmark.PINKY_TIP].y - points[mp_pose.HandLandmark.PINKY_MCP].y])
    ]
    return all(d < 0.1 for d in distances)

while video.isOpened():
    success, image = video.read()
    if not success:
        break

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = detector.process(image_rgb)
    finger_count = 0

    if results.multi_hand_landmarks:
        for i, hand in enumerate(results.multi_hand_landmarks):
            side = results.multi_handedness[i].classification[0].label
            landmarks = hand.landmark

            if check_closed(landmarks):
                current_count = 0
            else:
                current_count = check_fingers(landmarks, side)
            finger_count += current_count

            mp_draw.draw_landmarks(image, hand, mp_pose.HAND_CONNECTIONS)

    cv2.putText(image, f'Fingers: {finger_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Hand Tracking', image)

    if cv2.waitKey(1) & 0xFF == 27:
        break

video.release()
cv2.destroyAllWindows()