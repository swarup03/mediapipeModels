import cv2
import mediapipe as mp
import time
import math


class poseDetector():
    def __init__(self, mode=False,numPoses =1,  smooth=True, modelComplex=False,
                 detectionCon = 0.5,personCon=0.5 ,trackCon = 0.5):
        self.mode =mode
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.modelComplex=modelComplex
        self.numPoses=numPoses
        self.personCon = personCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode,self.numPoses,self.smooth,self.modelComplex,
                                     self.detectionCon,self.personCon,self.trackCon)
    def findPose(self ,img ,draw = True):
        imgRGB = cv2.cvtColor(img ,cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        #   print(results.pose_landmarks) # this will print x, y and z codiates
        if self.results.pose_landmarks:
            if draw == True:
                self.mpDraw.draw_landmarks(img ,self.results.pose_landmarks ,self.mpPose.POSE_CONNECTIONS)
        return img
    def findPosition(self ,img ,draw = True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id ,lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                #             print(id,lm)
                cx ,cy =int(lm. x *w) ,int(lm. y *h)
                self.lmList.append([id ,cx ,cy])
                if draw:
                    cv2.circle(img ,(cx ,cy) ,10 ,(255 ,0 ,255) ,cv2.FILLED)
        return self.lmList
    def fingAngle(self,img,p1,p2,p3,draw = True):
        #gating the land marks
        x1,y1 = self.lmList[p1][1:]
        x2,y2 = self.lmList[p2][1:]
        x3,y3 = self.lmList[p3][1:]

        #calculate the angle
        angle = math.degrees(math.atan2(y3-y2,x3-x2)-math.atan2(y1-y2,x1-x2))

        if angle <0:
            angle +=360


        if draw:
            cv2.line(img,(x1,y1),(x2,y2),(255,255,255),3)
            cv2.line(img,(x3,y3),(x2,y2),(255,255,255),3)
            cv2.circle(img, (x1, y1), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img,(x1,y1),10,(0,0,255))
            cv2.circle(img, (x2, y2), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img,(x2,y2),10,(0,0,255))
            cv2.circle(img, (x3, y3), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img,(x3,y3),10,(0,0,255))
            cv2.putText(img,str(int(angle)),(x2-50,y2+50),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)

        return angle


def main():
    cap = cv2.VideoCapture(1)
    pTime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img =detector.findPose(img)
        lmList = detector.findPosition(img,draw = False)

        cTime = time.time()
        fps = 1 /(cTime -pTime)
        pTime = cTime

        cv2.putText(img ,str(float(fps)) ,(70 ,50) ,cv2.FONT_HERSHEY_COMPLEX ,3 ,(255 ,255 ,0) ,3)

        # code which will show the video
        cv2.imshow("image" ,img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()