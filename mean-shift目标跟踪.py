import numpy as np
import cv2

cap=cv2.VideoCapture('E:\\YOLOV5+DeepSORT\\yolov5-deepsort\\video\\test_person.mp4')

#指定追踪目标
ret,frame=cap.read()
print(frame.shape)
r=250
c=250
h=180
w=100

win=(c,r,w,h)
roi=frame[r:r+h,c:c+w]


#计算直方图
hsv_roi=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
roi_hist=cv2.calcHist([hsv_roi],[0],None,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
#目标追踪
#迭代停止条件
term=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,10,1)

while True:
    ret, frame = cap.read()
    if ret==True:
        hst = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #反向投影
        dst=cv2.calcBackProject([hst],[0],roi_hist,[0,180],1)

        ret,win=cv2.meanShift(dst,win,term)
        x,y,w1,h1=win
        img2=cv2.rectangle(frame,(x,y),(x+w1,y+h1),255,2)
        cv2.imshow('result',img2)
        cv2.waitKey(60)






