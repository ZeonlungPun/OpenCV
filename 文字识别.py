import cv2
import numpy as np
from PIL import Image
import pytesseract
#创建Haar级联器
plate=cv2.CascadeClassifier("E:\opencv\sources\data\haarcascades\haarcascade_russian_plate_number.xml")

img=cv2.imread(r'E:\\test3.jpg')

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#检测位置
plates=plate.detectMultiScale(gray,1.1,3)
for (x,y,w,h) in plates:
   #print(x,y,w,h)
   cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
#提取ROI
roi=gray[y:y+h,x:x+w]
#二值化
ret,roi_bin=cv2.threshold(roi,100,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print(ret)
ans=pytesseract.image_to_string(roi,lang='eng',config='--psm 8 --oem 3')
print(ans)
cv2.imshow('1',roi_bin)
cv2.waitKey()
