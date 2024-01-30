import cv2
import time
import numpy as np
import HandTracking_model as htm
import os

folderPath = "paintHeader"
myList = os.listdir(folderPath)
overlayList=[]
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

drawColor = (255,255,255)
header=overlayList[0]

ptime = 0
ctime = 0
cap = cv2.VideoCapture(1)
xp,yp=0,0
detecter = htm.handDetecter(detectionCon=0.7)

imgCanvas = np.zeros((720,1278,3),np.uint8)
while True:
    # Import image
    success, img = cap.read()
    img = cv2.resize(img,(1278,720))
    img = cv2.flip(img,1)

    # find hand landmarks
    img=detecter.findHands(img)
    lmList = detecter.findPosition(img,draw=False)

    if lmList is not None:
        if len(lmList) != 0:
            # accessing the points of middle finger and index finger
            _,x1,y1 = lmList[8]
            _,x2,y2 = lmList[12]
            # print(lmList)

            #chacking whether the finger are up
            fingerStatus=detecter.fingerUp()
            # print(fingerStatus)

            # selection mode:- two finger are up (choose color)
            if fingerStatus[1] and fingerStatus[2]:
                xp,yp =0,0
                # print("selection")
                if y1<126:
                    if 222<x1<425:
                        header = overlayList[0]
                        drawColor = (255,255,255)
                    elif 425<x1<620:
                        header = overlayList[1]
                        drawColor=(0,255,0)
                    elif 620<x1<810:
                        header = overlayList[2]
                        drawColor = (255,0,0)
                    elif 820<x1<1000:
                        header = overlayList[3]
                        drawColor = (0,0,255)
                    elif 1030<x1<1220:
                        header = overlayList[4]
                        drawColor = (0,0,0)
                cv2.rectangle(img,(x1,y1-20),(x2,y2+20),drawColor,cv2.FILLED)



            # drawing mde:= only index finger up (draw random)
            if fingerStatus[1] and fingerStatus[2] == False:
                cv2.circle(img,(x1,y1),15,drawColor,cv2.FILLED)
                # print("drawing mode")
                if xp == 0 and yp ==0:
                    xp,yp = x1,y1

                if drawColor ==(0,0,0):
                    cv2.line(img, (xp, yp), (x1, y1), drawColor, 50)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, 50)
                else:
                    cv2.line(img,(xp,yp),(x1,y1),drawColor,15)
                    cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,15)

                xp,yp =x1,y1

    imgGray = cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)

    # Adding the header image
    img[0:126,0:1278]=header
    # img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 0), 2)
    cv2.imshow("Image", img)
    cv2.imshow("ImageCanvas", imgCanvas)

    cv2.waitKey(1)