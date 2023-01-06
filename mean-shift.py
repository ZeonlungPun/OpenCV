import numpy as np
import cv2,pickle



#指定追踪目标
#ret,frame=cap.read()
#print(frame.shape)
# r=250
# c=250
# h=180
# w=100
# win=(c,r,w,h)
# roi=frame[r:r+h,c:c+w]

try:
    with open('PeoplePos','rb') as f:
        posList=pickle.load(f)
        print('coordinate:',posList)
except:
    posList = []


class getTag:
    #可以利用鼠标事件指定追踪对象
    def __init__(self):
        self.cap = cv2.VideoCapture('E:\\YOLOV5+DeepSORT\\yolov5-deepsort\\video\\test_person.mp4')
    @staticmethod
    def mouseClick(events, x, y, flags, parms):
        leftTopList = []
        rightBottomList = []

        """
        Args:
            events: pass automatically
            x: click the point
            y: click the point
        Returns: position of car park:top left point; bottom right point
        """
        if events == cv2.EVENT_LBUTTONDOWN:
            leftTopList.append(x)
            leftTopList.append(y)
            posList.append(leftTopList)
            print('successfully save left top coordinate')
        if events == cv2.EVENT_RBUTTONDOWN:
            rightBottomList.append(x)
            rightBottomList.append(y)
            posList.append(rightBottomList)

            print('successfully save right bottom coordinate')
        with open('PeoplePos', 'wb') as f:
            pickle.dump(posList, f)

    def set_coordinate(self):
        while True:
            _, img =self.cap.read()
            cv2.imshow("show",img)
            cv2.setMouseCallback('show',self.mouseClick)
            key = cv2.waitKey()
            if key == 27:
                break


    def show_box(self):
        while True:
            _, img =self.cap.read()

            leftTopList = posList[0]
            rightBottomList = posList[1]

            self.h = int(rightBottomList[1]) - int(leftTopList[1])
            self.w = int(rightBottomList[0]) - int(leftTopList[0])
            self.r = leftTopList[1]  # y1
            self.c = leftTopList[0]  # x1
            cv2.rectangle(img, (self.c, self.r), (self.c + self.w, self.r + self.h), 255, 2)

            cv2.imshow("show",img)

            key = cv2.waitKey()
            if key == 27:
                break


    def catch_target(self):
        #计算直方图
        ret, frame = self.cap.read()
        win=(self.c,self.r,self.w,self.h)
        roi=frame[self.r:self.r+self.h,self.c:self.c+self.w]
        hsv_roi=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
        roi_hist=cv2.calcHist([hsv_roi],[0],None,[180],[0,180])
        cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
        #目标追踪
        #迭代停止条件
        term=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,10,1)

        while True:
            ret, frame = self.cap.read()
            if ret==True:
                hst = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                #反向投影
                dst=cv2.calcBackProject([hst],[0],roi_hist,[0,180],1)

                ret,win=cv2.meanShift(dst,win,term)
                x,y,w1,h1=win
                img2=cv2.rectangle(frame,(x,y),(x+w1,y+h1),255,2)
                cv2.imshow('result',img2)
                key=cv2.waitKey(60)

                if (key == 27):  # 键盘按ESC，则退出
                    break
        self.cap.release()
        cv2.destroyAllWindows()


GET_TAG=getTag()
GET_TAG.set_coordinate()
GET_TAG.show_box()
GET_TAG.catch_target()












