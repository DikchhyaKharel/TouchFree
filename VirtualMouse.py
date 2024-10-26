import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Setting up MediaPipe for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Initialize screen dimensions for cursor control
screen_width, screen_height = pyautogui.size()

# Track last double-click and right-click times to prevent multiple triggers
last_double_click_time = 0
last_right_click_time = 0
gesture_start_time = None  # To track how long a gesture has been held
gesture_hold_threshold = 1.0  # Time in seconds to hold most gestures
right_click_hold_threshold = 0.5  # Right-click hold threshold
right_click_cooldown = 1.0  # Cooldown for right-click in seconds

# Function to calculate the position of the cursor based on hand position
def get_cursor_position(landmark, frame_width, frame_height):
    x, y = int(landmark.x * frame_width), int(landmark.y * frame_height)
    screen_x = np.clip(np.interp(x, (0, frame_width), (0, screen_width)), 0, screen_width - 1)
    screen_y = np.clip(np.interp(y, (0, frame_height), (0, screen_height)), 0, screen_height - 1)
    return screen_x, screen_y

# Function to detect specific gestures
def detect_gesture(hand_landmarks):
    fingers = []
    # Thumb
    if hand_landmarks[4].x < hand_landmarks[3].x:
        fingers.append(1)
    else:
        fingers.append(0)
    # Other fingers
    for id in range(8, 21, 4):
        if hand_landmarks[id].y < hand_landmarks[id - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

# Main code to start video capture and gesture control
cap = cv2.VideoCapture(0)
pTime = 0
drag_active = False  # To track if drag mode is active
current_gesture = None  # To track the ongoing gesture

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip the image for a mirrored view
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape

    # Convert frame to RGB as MediaPipe requires RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Check if any hand landmarks are detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks on the hand
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmark positions
            landmarks = hand_landmarks.landmark

            # Get the current gesture based on finger positions
            fingers = detect_gesture(landmarks)

            # Identify gesture type and perform corresponding mouse action
            if fingers == [1, 1, 1, 1, 1]:
                # Neutral Position: Open Palm (all fingers up)
                # Releases drag if it was active
                gesture_type = "neutral"

            elif fingers == [0, 1, 0, 0, 0]:
                # Cursor Movement: Index Finger Up, Other Fingers Down
                # Moves the cursor based on the index finger's position
                gesture_type = "move_cursor"

            elif fingers == [0, 1, 1, 0, 0] and not drag_active:
                # Left Click: Index and Middle Fingers Up, Other Fingers Down
                # Simulates a left-click
                gesture_type = "left_click"

            elif abs(landmarks[4].x - landmarks[8].x) < 0.05:
                # Right Click: Thumb and Index Close Together
                # Simulates a right-click with a cooldown
                gesture_type = "right_click"

            elif fingers == [1, 0, 0, 0, 0]:
                # Drag and Drop: Only Thumb Up, All Other Fingers Down
                # Initiates drag mode by holding down the mouse button
                gesture_type = "drag_and_drop"

            elif abs(landmarks[4].x - landmarks[20].x) < 0.05:
                # Double Click: Thumb and Pinky Together
                # Simulates a double-click with a cooldown to prevent multiple triggers
                gesture_type = "double_click"

            else:
                gesture_type = None

            # Check if the gesture has changed
            if gesture_type != current_gesture:
                current_gesture = gesture_type
                gesture_start_time = time.time()  # Reset the start time for new gesture

            # If gesture is held long enough, execute action
            if gesture_type:
                hold_time = right_click_hold_threshold if gesture_type == "right_click" else gesture_hold_threshold

                # If gesture is held long enough, perform the action
                if time.time() - gesture_start_time > hold_time:
                    if gesture_type == "neutral":
                        # Release drag if it was active
                        if drag_active:
                            pyautogui.mouseUp()
                            drag_active = False

                    elif gesture_type == "move_cursor":
                        # Move the cursor based on the index finger's position
                        cursor_x, cursor_y = get_cursor_position(landmarks[8], frame_width, frame_height)
                        pyautogui.moveTo(cursor_x, cursor_y)

                    elif gesture_type == "left_click":
                        # Perform a left-click
                        pyautogui.click()

                    elif gesture_type == "right_click":
                        current_time = time.time()
                        # Check cooldown for right-click to prevent multiple triggers
                        if current_time - last_right_click_time > right_click_cooldown:
                            pyautogui.click(button='right')
                            last_right_click_time = current_time

                    elif gesture_type == "drag_and_drop":
                        # Initiate drag mode by holding down the mouse button
                        if not drag_active:
                            pyautogui.mouseDown()
                            drag_active = True

                    elif gesture_type == "double_click":
                        current_time = time.time()
                        # Prevent multiple double-clicks by adding a cooldown
                        if current_time - last_double_click_time > 0.5:
                            pyautogui.doubleClick()
                            last_double_click_time = current_time

    # Display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the output frame
    cv2.imshow("AI Virtual Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
