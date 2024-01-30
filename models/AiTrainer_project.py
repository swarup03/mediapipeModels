import cv2
import time
import numpy as np
import PoseEstimation_model as pem

cap = cv2.VideoCapture("gymVideos/1.mp4")
# cap = cv2.VideoCapture(1)
pTime = 0
detector = pem.poseDetector()
count = 0
direction = 0
while True:
    success, img = cap.read()

    # img = cv2.imread("gymVideos/k1.png")

    img =detector.findPose(img,draw=False)
    lmList = detector.findPosition(img,draw = False)
    if lmList is not None:
        if len(lmList) != 0:
            angle = detector.fingAngle(img,12,14,16,draw=False)
            per = np.interp(angle,(60,160),(0,100))
            # print(per)

            if per == 100:
                if direction==0:
                    count += .5
                    direction =1
            if per == 0:
                if direction == 1:
                    count += .5
                    direction = 0
            print(count)

            cv2.putText(img,str(int(count)),(50,150),cv2.FONT_HERSHEY_PLAIN,5,(255,0,0),5)



    cTime = time.time()
    fps = 1 /(cTime -pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)) ,(50 ,60) ,cv2.FONT_HERSHEY_PLAIN ,3 ,(255 ,255 ,0) ,3)

    # code which will show the video
    cv2.imshow("image",img)
    cv2.waitKey(1)