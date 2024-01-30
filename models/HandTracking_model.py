import cv2
import mediapipe as mp
import time

class handDetecter():
    def __init__ (self,mode=False,maxHands=2,modelComplex=False,detectionCon=0.5,trackCon=0.5):
        self.mode=mode
        self.maxHands=maxHands
        self.detectionCon=detectionCon
        self.trackCon=trackCon
        self.modelComplex=modelComplex

        self.mpHands = mp.solutions.hands
        self.hands = self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, self.detectionCon, self.trackCon)

        self.mpDraw = mp.solutions.drawing_utils
        self.tipInds = [4,8,12,16,20]


    def findHands(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
    def findPosition(self,img,handNo=0, draw=True):

        self.lmList=[]
        if self.results.multi_hand_landmarks:
            myhands = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myhands.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                self.lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            return self.lmList
    def fingerUp(self):
        fingersStatus = []
        # in open cv mx height is 0, so we will use inverse of the graph
        if self.lmList[self.tipInds[0]][1] < self.lmList[self.tipInds[0] - 1][1]:
            fingersStatus.append(1)
        else:
            fingersStatus.append(0)
        for id in range(1, len(self.tipInds)):
            if self.lmList[self.tipInds[id]][2] < self.lmList[self.tipInds[id] - 2][2]:
                fingersStatus.append(1)
            else:
                fingersStatus.append(0)
        return fingersStatus

def main():
    ptime = 0
    ctime = 0
    cap = cv2.VideoCapture(1)
    detecter = handDetecter()
    while True:
        success, img = cap.read()
        img=detecter.findHands(img)
        lmList = detecter.findPosition(img,draw=False)
        if lmList is not None:
            # num = int(input('enter which finger'))
            if len(lmList) != 0:
                print(lmList[4])


        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 0), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__=="__main__":
    main()