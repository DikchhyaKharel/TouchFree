import keyboard_mouse as km
import cv2
import mediapipe as mp

# Initialize the video capture and hand-tracking modules
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

initHand = mp.solutions.hands
mainHand = initHand.Hands()
draw = mp.solutions.drawing_utils

keys = [["A", "Z", "E", "R", "T", "Y", "U", "I", "O", "P", "^", "$"],
        ["Q", "S", "D", "F", "G", "H", "J", "K", "L", "M", "%", "*"],
        ["W", "X", "C", "V", "B", "N", ",", ";", ":", "!", ".", "?"]]

finalText = ""
clicked = False

def handLandmarks(colorImg):
    """Detect hand landmarks using MediaPipe."""
    landmarkList = []
    landmarkPositions = mainHand.process(colorImg)
    landmarkChek = landmarkPositions.multi_hand_landmarks
    if landmarkChek:
        for hand in landmarkChek:
            for index, landmark in enumerate(hand.landmark):
                landmarkList.append([index, int(landmark.x * 1280), int(landmark.y * 720)])
    return landmarkList


def drawAll(img, buttonList):
    """Draw all buttons on the keyboard."""
    for button in buttonList:
        x, y = button.pos
        w, h = button.size

        if button.text in ["Space", "Delete", "Switch to Mouse"]:
            cv2.rectangle(img, button.pos, (x + w, y + h), (64, 64, 64), cv2.FILLED)
            text_x = x + int(w * 0.35) - 50
            text_y = y + int(h * 0.65)
            cv2.putText(img, button.text, (text_x, text_y), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
        else:
            cv2.rectangle(img, button.pos, (x + w, y + h), (64, 64, 64), cv2.FILLED)
            cv2.putText(img, button.text, (x + 25, y + 60), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
    return img


class Button:
    """Define a button for the virtual keyboard."""
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.text = text
        self.size = size

def keyboard():
    """Run the virtual keyboard application."""
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    initHand = mp.solutions.hands
    mainHand = initHand.Hands()
    draw = mp.solutions.drawing_utils

    keys = [["A", "Z", "E", "R", "T", "Y", "U", "I", "O", "P", "^", "$"],
            ["Q", "S", "D", "F", "G", "H", "J", "K", "L", "M", "%", "*"],
            ["W", "X", "C", "V", "B", "N", ",", ";", ":", "!", ".", "?"]]

    finalText = ""
    clicked = False
    buttonList = []

    # Create keyboard buttons
    for i in range(len(keys)):
        for j, key in enumerate(keys[i]):
            buttonList.append(Button([100 * j + 50, 100 * i + 50], key))

    # Add "Space", "Delete", and "Switch to Mouse" buttons
    buttonList.append(Button([50, 350], "Space", [885, 85]))
    buttonList.append(Button([950, 350], "Delete", [285, 85]))
    buttonList.append(Button([50, 450], "Switch to Mouse", [885, 85]))  # New button

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        lmlist = handLandmarks(img)
        img = drawAll(img, buttonList)

        if lmlist:
            for button in buttonList:
                x, y = button.pos
                w, h = button.size

                if x < lmlist[8][1] < x + w and y < lmlist[8][2] < y + h:
                    if button.text in ["Space", "Delete", "Switch to Mouse"]:
                        cv2.rectangle(img, button.pos, (x + w, y + h), (128, 128, 128), cv2.FILLED)
                        text_x = x + int(w * 0.35) - 50
                        text_y = y + int(h * 0.65)
                        cv2.putText(img, button.text, (text_x, text_y), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                    else:
                        cv2.rectangle(img, button.pos, (x + w, y + h), (128, 128, 128), cv2.FILLED)
                        cv2.putText(img, button.text, (x + 25, y + 60), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)

                    # Detect click
                    if lmlist[8][2] < lmlist[7][2] and lmlist[12][2] < lmlist[11][2] and not clicked:
                        if button.text == "Space":
                            finalText += " "
                        elif button.text == "Delete":
                            finalText = finalText[:-1]
                        elif button.text == "Switch to Mouse":
                            cap.release()
                            cv2.destroyAllWindows()
                            km.call_mouse()
                            return 
                        else:
                            finalText += button.text
                        clicked = True

                    if lmlist[8][2] < lmlist[7][2] and lmlist[12][2] > lmlist[11][2]:
                        clicked = False

        # Display the typed text
        cv2.rectangle(img, (50, 580), (1235, 680), (64, 64, 64), cv2.FILLED)
        cv2.putText(img, finalText, (60, 645), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)

        # Show the keyboard
        cv2.imshow('Virtual Keyboard', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Run the virtual keyboard
    keyboard()

