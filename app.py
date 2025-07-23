import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Initialize camera and hand detection
cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.75)
draw_utils = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()
smooth_x, smooth_y = 0, 0
smoothening = 5

# Cooldown for clicks
click_delay = 0.3
last_click_time = 0

# Mapping margins (adjust if needed)
margin = 20

def fingers_up(hand_landmarks):
    tips = [8, 12, 16, 20]
    up = []
    for tip in tips:
        up.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y)
    return up

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hand_detector.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            draw_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

            # Get key points
            index_finger = hand_landmarks.landmark[8]
            thumb = hand_landmarks.landmark[4]
            middle_finger = hand_landmarks.landmark[12]

            index_x = int(index_finger.x * width)
            index_y = int(index_finger.y * height)

            # Convert to screen coords with margin
            screen_x = np.interp(index_x, [margin, width - margin], [0, screen_width])
            screen_y = np.interp(index_y, [margin, height - margin], [0, screen_height])

            # Smooth movement
            smooth_x += (screen_x - smooth_x) / smoothening
            smooth_y += (screen_y - smooth_y) / smoothening
            pyautogui.moveTo(smooth_x, smooth_y)

            # Draw pointer
            cv2.circle(frame, (index_x, index_y), 15, (0, 255, 0), cv2.FILLED)

            # Gestures
            dist_index_thumb = np.hypot(index_finger.x - thumb.x, index_finger.y - thumb.y)
            dist_middle_thumb = np.hypot(middle_finger.x - thumb.x, middle_finger.y - thumb.y)
            fingers = fingers_up(hand_landmarks)
            current_time = time.time()

            # Left Click: Index + Thumb
            if dist_index_thumb < 0.05 and current_time - last_click_time > click_delay:
                pyautogui.click()
                last_click_time = current_time

            # Right Click: Middle + Thumb
            elif dist_middle_thumb < 0.05 and current_time - last_click_time > click_delay:
                pyautogui.rightClick()
                last_click_time = current_time

            # Scroll Up (2 fingers up)
            elif fingers[0] and fingers[1] and not fingers[2] and not fingers[3]:
                pyautogui.scroll(25)

            # Scroll Down (2 fingers down)
            elif not fingers[0] and not fingers[1] and fingers[2] and fingers[3]:
                pyautogui.scroll(-25)

    cv2.imshow("Virtual Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
