import cv2
import numpy as np
import time
import HandTracking_model as htm
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wCam,hCam = 720,560

ptime = 0
ctime = 0
cap = cv2.VideoCapture(1)
cap.set(3,wCam)
cap.set(4,hCam)


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

volRange=volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol=0
volBar=300
volPer = 0

detector = htm.handDetecter(detectionCon=0.7)
while True:
    success, img = cap.read()
    img=detector.findHands(img)
    lmlist = detector.findPosition(img,draw=False)
    if lmlist is not None:
        # num = int(input('enter which finger'))
        if len(lmlist) != 0:
            # print(lmlist[4],lmlist[8])

            x1,y1 = lmlist[4][1],lmlist[4][2]
            x2,y2 = lmlist[8][1],lmlist[8][2]
            cx,cy = (x1+x2)//2,(y1+y2)//2

            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.line(img,(x1,y1),(x2,y2),(0,255,0),3)
            cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

            # cal
            length = math.hypot(x2-x1,y2-y1)
            # print(length)

            # Hand range 25 - 200
            # volume range -65.25 - 0.0

            vol = np.interp(length,[25,190],[minVol,maxVol])
            volBar = np.interp(length,[25,190],[400,150])
            volPer = np.interp(length,[25,190],[1,100])

            print(vol)
            volume.SetMasterVolumeLevel(vol, None)

            if length<=25:
                cv2.circle(img, (cx, cy), 10, (0,255,0), cv2.FILLED)

        cv2.rectangle(img,(50,150),(85,400),(0,255,0),3)
        cv2.rectangle(img,(50,int(volBar)),(85,400),(0,255,0),cv2.FILLED)
        cv2.putText(img,f'{int(volPer)}%',(40,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),3)





    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 0), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)