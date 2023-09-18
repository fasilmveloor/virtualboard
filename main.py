import cv2
import time
import mediapipe as mp
import HandTrackingModule as htm
import numpy as np
import imutils


wCam, hCam = 1080, 720

cap = cv2.VideoCapture(0)
cap.set(3, 1080)
cap.set(4, 720)
cap.set(2, 1080)

pTime = 0
cTime = 0

header = cv2.imread('drawer.png')

detector = htm.HandDetector(detectionCon=0.75)

count = 0
dir = 0
drawColor = (255,0,255)
brush = 15
eraser = 50
xp , yp = 0,0

imgCanvas = np.zeros((607, 1080, 3), np.uint8)

while True:
    #1. Import image
    success, img = cap.read()
    img = imutils.resize(img, width=1080)
    img = cv2.flip(img, 1)
    
    #2Find hand landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:

        #Tip of index finger
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        fingers = detector.fingersUp()

        if fingers[1] and fingers[2] and fingers.count(1)==2:
            cv2.rectangle(img, (x1, y1-25), (x2, y2+25), drawColor, cv2.FILLED)
            xp, yp = 0, 0
        elif fingers[1] and fingers[2]==False: 
            cv2.circle(img, (x1,y1), 15, drawColor, cv2.FILLED)
            if xp==0 and yp == 0:
                xp, yp = x1, y1
            cv2.line(img, (xp,yp), (x1,y1), drawColor, brush)
            cv2.line(imgCanvas, (xp,yp), (x1,y1), drawColor, brush)
            xp, yp = x1, y1

        elif fingers[1] and fingers[2] and fingers[3]:
            if xp==0 and yp == 0:
                xp, yp = x1, y1

            eraserColor = (0,0,0)
            cv2.line(img, (xp,yp), (x1,y1), eraserColor, eraser)
            cv2.line(imgCanvas, (xp,yp), (x1,y1), eraserColor, eraser)
            xp, yp = x1, y1


    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)


    
    
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS:{int(fps)}', (40, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
    img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5,0)
    cv2.imshow("Image",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break