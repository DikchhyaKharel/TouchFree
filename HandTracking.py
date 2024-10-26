import cv2
import mediapipe as mp
import time
from mediapipe import solutions

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon  # Keep as float for MediaPipe
        self.trackCon = trackCon  # Keep as float for MediaPipe
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)

        if self.result.multi_hand_landmarks:
            for handLms in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
    
    def findPos(self, img, handNo = 0, draw = True):
        lmList = []
        if self.result.multi_hand_landmarks:
            myHand = self.result.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                        
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                       
                        lmList.append([id,cx,cy])
                        if draw:
                            cv2.circle(img, (cx,cy),15,(255,0,0),cv2.FILLED)
        return lmList

def main():
    pTime = 0
    cTime = 0
    vid = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = vid.read()
        img = cv2.flip(img, 1)  # Optional: Flip image to avoid mirroring
        img = detector.findHands(img) #gets the hand 
        lmList = detector.findPos(img)
        if len(lmList) !=0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
