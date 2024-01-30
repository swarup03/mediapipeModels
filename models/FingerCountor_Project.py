import cv2
import os
import time
import HandTracking_model as htm

wCam,hCam = 1080,920

ptime = 0
ctime = 0
cap = cv2.VideoCapture(1)
cap.set(3,wCam)
cap.set(4,hCam)

#gatimg image and their location
folderPath = "FingerImages"
myList = os.listdir(folderPath)
overlayList = []
for imPath in myList:
    image = cv2.imread((f'{folderPath}/{imPath}'))
    overlayList.append(image)


detector = htm.handDetecter(detectionCon=0.75)
tipInds = [4,8,12,16,20]
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if lmList is not None:
        if len(lmList) != 0:
            fingersStatus = []
            # in open cv mx height is 0, so we will use inverse of the graph
            if lmList[tipInds[0]][1] > lmList[tipInds[0] - 1][1]:
                fingersStatus.append(1)
            else:
                fingersStatus.append(0)
            for id in range(1,len(tipInds)):
                if lmList[tipInds[id]][2] < lmList[tipInds[id]-2][2]:
                    fingersStatus.append(1)
                else:
                    fingersStatus.append(0)


            # print(fingersStatus)
            totalFingers = fingersStatus.count(1)
            h,w,c = overlayList[totalFingers].shape
            img[0:h,0:w]=overlayList[totalFingers-1]

            cv2.rectangle(img,(20,255),(170,425),(0,255,0),cv2.FILLED)
            cv2.putText(img,str(totalFingers),(45,375),cv2.FONT_HERSHEY_PLAIN,10,(255,0,0),25)





    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(img,"FPS: "+str(int(fps)), (450, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)