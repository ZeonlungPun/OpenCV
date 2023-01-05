import cv2
import numpy as np


cap=cv2.VideoCapture("E:\\people.mp4")

#去除前景算法
bgsubmog=cv2.bgsegm.createBackgroundSubtractorMOG()
#形态学kernel
kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

min_w=30
min_h=30
#检测线位置,可尽量靠近视频底部
line_height=500

def center(x,y,w,h):
    x1=int(w/2)
    y1=int(h/2)
    cx=x+x1
    cy=y+y1

    return cx,cy
#存放有效特征图
cars=[]
#偏移量
offset=3
carnum=0

while True:
    ret,frame=cap.read()

    if (ret==True):
        #灰度化
        cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #print(frame.shape)
        #高斯去噪
        blur=cv2.GaussianBlur(frame,(3,3),5)
        #去除背景
        mask=bgsubmog.apply(blur)
        #腐蚀,去掉图中的小斑块
        erode=cv2.erode(mask,kernel)
        #膨胀，还原放大
        dilate=cv2.dilate(erode,kernel,iterations=2)
        #闭：去掉物体内部的小块
        close=cv2.morphologyEx(dilate,cv2.MORPH_CLOSE,kernel)
        #close = cv2.morphologyEx(close, cv2.MORPH_CLOSE, kernel)
        #查找轮廓
        cnts,h=cv2.findContours(close,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        #画一条基准线
        cv2.line(frame,(10,line_height),(1400,line_height),(255,255,0),3)

        #画识别出来的每个图像的外接矩形
        for (i,c) in enumerate(cnts):
            #获取坐标
            (x,y,w,h)=cv2.boundingRect(c)
            # 对识别的图像进行过滤
            isValid = ((w >= min_w) and (h >= min_h))
            if (not isValid):
                continue
            #画矩形
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)


            #得到车的中心点
            cpoints=center(x,y,w,h)
            cars.append(cpoints)
            #对有效车辆进行计数：中心点是否经过两条线之间
            for (x,y) in cars:
                if (y>line_height-offset) and (y<line_height+offset):
                    carnum+=1
                    cars.remove((x,y))
                    print(carnum)

        cv2.putText(frame,"count:"+str(carnum),(500,60),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),5)
        cv2.imshow('vedio',frame)

    key=cv2.waitKey(100)
    if (key==27):#键盘按ESC，则退出
        break

cap.release()
cv2.destroyAllWindows()

