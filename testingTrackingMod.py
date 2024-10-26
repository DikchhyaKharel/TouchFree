import cv2
import mediapipe as mp
import time
import HandTracking as htm

pTime = 0
cTime = 0
vid = cv2.VideoCapture(0)
detector = htm.handDetector()

while True:
    success, img = vid.read()
    img = cv2.flip(img, 1)  #Flipping image to avoid mirroring
    img = detector.findHands(img) #finds the hand 
    lmList = detector.findPos(img) #finds the position of each point on hand
    if len(lmList) !=0:
        print(lmList[4])

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    