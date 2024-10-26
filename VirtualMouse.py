import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# -----------------------------------------------------------
# Explanation:
# This script uses OpenCV, MediaPipe, and PyAutoGUI to enable 
# gesture-based control of the mouse and system volume. It 
# captures video from the webcam, detects hand landmarks using 
# MediaPipe, interprets specific hand gestures to perform 
# actions like moving the cursor, left-click, right-click, 
# dragging, and volume control. Pycaw is used to manage 
# system volume, allowing for volume adjustments via thumb 
# and index finger gestures.
# 
# OpenCV handles video input/output, 
# MediaPipe detects hand landmarks, and 
# PyAutoGUI maps gestures to mouse actions. 
# -----------------------------------------------------------

# Setting up MediaPipe for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Initialize screen dimensions for cursor control
screen_width, screen_height = pyautogui.size()

# Track last click times to prevent multiple triggers
last_double_click_time = 0
last_right_click_time = 0
last_left_click_time = 0
gesture_start_time = None  # To track how long a gesture has been held
gesture_hold_threshold = 1.0  # Time in seconds to hold most gestures
click_cooldown = 0.3  # Cooldown for clicks in seconds
right_click_delay = 0.5  # Delay after right click to prevent multiple triggers
drag_active = False  # To track if drag-and-drop is active
multiple_select_active = False  # To track if multiple selection is active

# Initialize Pycaw for volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# -----------------------------------------------------------
# Function: get_cursor_position
# Description:
# This function calculates the on-screen cursor position 
# based on the coordinates of the detected hand landmark.
# It maps the webcam's frame dimensions to the screen 
# resolution to ensure smooth cursor movement.
# -----------------------------------------------------------
def get_cursor_position(landmark, frame_width, frame_height):
    x, y = int(landmark.x * frame_width), int(landmark.y * frame_height)
    screen_x = np.clip(np.interp(x, (0, frame_width), (0, screen_width)), 0, screen_width - 1)
    screen_y = np.clip(np.interp(y, (0, frame_height), (0, screen_height)), 0, screen_height - 1)
    return screen_x, screen_y

# -----------------------------------------------------------
# Function: detect_gesture
# Description:
# This function determines which fingers are extended or 
# folded based on the hand landmarks detected by MediaPipe.
# It returns an array indicating the state of each finger, 
# which is then used to identify specific gestures.
# -----------------------------------------------------------
def detect_gesture(hand_landmarks):
    fingers = []
    # Thumb detection
    if hand_landmarks[4].x < hand_landmarks[3].x:
        fingers.append(1)
    else:
        fingers.append(0)
    # Other fingers detection
    for id in range(8, 21, 4):
        if hand_landmarks[id].y < hand_landmarks[id - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

# Main code to start video capture and gesture control
cap = cv2.VideoCapture(0)
pTime = 0
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

    # -----------------------------------------------------------
    # Processing and Recognizing Gestures:
    # If hand landmarks are detected, the script identifies 
    # different gestures based on the positions of the fingers 
    # and executes actions like clicking, dragging, volume 
    # control, and cursor movement. It ensures smooth and 
    # responsive interactions by checking gesture changes and 
    # adding delays to prevent multiple triggers.
    # -----------------------------------------------------------
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
                gesture_type = "neutral"

            elif fingers == [0, 1, 0, 0, 0]:
                # Cursor Movement: Index Finger Up, Other Fingers Down
                gesture_type = "move_cursor"
            
            elif fingers == [0, 1, 1, 0, 0]:
                # Left Click: Index and Middle Fingers Up, Other Fingers Down
                gesture_type = "left_click"

            elif fingers == [0, 0, 0, 0, 1]:
                # Right Click: Pinky Up, All Other Fingers Down
                gesture_type = "right_click"

            elif fingers == [0, 0, 0, 0, 0]:
                # Multiple Select: Fist Gesture (All Fingers Down)
                gesture_type = "multiple_select"

            # Drag-and-Drop: V Gesture with Index and Middle Bent to Start Drag
            # And V Gesture with Both Straight Up to Drop
            elif (fingers == [0, 1, 1, 0, 0] and 
                  landmarks[8].y > landmarks[6].y and landmarks[12].y > landmarks[10].y):
                # Both index and middle bent down
                gesture_type = "drag_and_drop_start"
            elif (fingers == [0, 1, 1, 0, 0] and 
                  landmarks[8].y < landmarks[6].y and landmarks[12].y < landmarks[10].y):
                # Both index and middle straight up
                gesture_type = "drag_and_drop_end"

            # Volume Control: Thumb and Index (Close decreases, Far increases)
            elif abs(landmarks[4].x - landmarks[8].x) < 0.05:
                gesture_type = "volume_control_decrease"
            elif abs(landmarks[4].x - landmarks[8].x) > 0.05:
                gesture_type = "volume_control_increase"

            else:
                gesture_type = None

            # Check if the gesture has changed
            if gesture_type != current_gesture:
                current_gesture = gesture_type
                gesture_start_time = time.time()  # Reset the start time for new gesture

            # If gesture is held long enough, execute action
            if gesture_type:
                # Handle Volume Control immediately
                if gesture_type == "volume_control_decrease":
                    current_volume = volume.GetMasterVolumeLevelScalar()
                    volume.SetMasterVolumeLevelScalar(max(current_volume - 0.02, 0.0), None)

                elif gesture_type == "volume_control_increase":
                    current_volume = volume.GetMasterVolumeLevelScalar()
                    volume.SetMasterVolumeLevelScalar(min(current_volume + 0.02, 1.0), None)

                else:
                    # Hold time for the gestures
                    hold_time = click_cooldown if gesture_type in ["right_click", "multiple_select", "drag_and_drop_start"] else gesture_hold_threshold

                    if time.time() - gesture_start_time > hold_time:
                        if gesture_type == "neutral":
                            # Release any active actions
                            if drag_active:
                                pyautogui.mouseUp()
                                drag_active = False
                            if multiple_select_active:
                                pyautogui.mouseUp()
                                multiple_select_active = False

                        elif gesture_type == "move_cursor":
                            # Move the cursor based on the index finger's position
                            cursor_x, cursor_y = get_cursor_position(landmarks[8], frame_width, frame_height)
                            pyautogui.moveTo(cursor_x, cursor_y)

                        elif gesture_type == "left_click":
                            current_time = time.time()
                            if current_time - last_left_click_time > click_cooldown:
                                pyautogui.click()
                                last_left_click_time = current_time

                        elif gesture_type == "right_click":
                            current_time = time.time()
                            if current_time - last_right_click_time > right_click_delay:
                                pyautogui.click(button='right')
                                last_right_click_time = current_time

                        elif gesture_type == "multiple_select":
                            # Start selecting when you make a fist and the cursor moves with your hand and release when done selecting items  
                            cursor_x, cursor_y = get_cursor_position(landmarks[8], frame_width, frame_height)
                            pyautogui.moveTo(cursor_x, cursor_y)
                            if not drag_active:
                                pyautogui.mouseDown()
                                drag_active = True
                            elif drag_active and fingers == [1, 1, 1, 1, 1]:
                                pyautogui.mouseUp()
                                drag_active = False

                        elif gesture_type == "drag_and_drop_start":
                            # Start drag when V gesture is bent down
                            cursor_x, cursor_y = get_cursor_position(landmarks[8], frame_width, frame_height)
                            pyautogui.moveTo(cursor_x, cursor_y)
                            if not drag_active:
                                pyautogui.mouseDown()
                                drag_active = True

                        elif gesture_type == "drag_and_drop_end":
                            # Drop the dragged item when V gesture is straight up
                            if drag_active:
                                pyautogui.mouseUp()
                                drag_active = False

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
