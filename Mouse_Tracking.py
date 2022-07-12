import cv2
import HandTrackingModule as htm
import numpy as np
import time
import autopy


frameWidth, frameHeight = 640, 420
frameR = 100
smoothening = 15

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()


while True:
    # Find hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # Get the tip of the index and middle fingers
    if len(lmList)!=0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        # print(x1, y1, x2, y2)
    # Check which fingers are up
    fingers = detector.fingersUp()
    cv2.rectangle(img, (frameR, frameR), (frameWidth-frameR, frameHeight-frameR), (255, 0, 0), 2)

    # Only Index Finger : Moving Mode
    if fingers[1] == 1 and fingers[2] == 0:
        x3 = np.interp(x1, (frameR, frameWidth-frameR), (0, wScr))
    # Convert Coordinates
        y3 = np.interp(y1, (frameR, frameHeight-frameR), (0, hScr))

    # Smoothen values
        clocX = plocX + (x3 - plocX) / smoothening
        clocY = plocY + (y3 - plocY) / smoothening

    # Move mouse
        autopy.mouse.move(wScr - clocX, clocY)
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        plocX, plocY = clocX, clocY
    # Both index and middle fingers are up : Clicking Mode
    if fingers[1] == 1 and fingers[2] == 1:
        length, img, lineInfo = detector.findDistance(8, 12, img)

    # Find Distance b/w fingers
    # Click mouse if distance short
        if length < 40:
            cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
            autopy.mouse.click()
    # Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20,50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)
    # Display

        # for handLms in results.multi_hand_landmarks:
        #     mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    cv2.imshow("Image", img)

    cv2.waitKey(1)