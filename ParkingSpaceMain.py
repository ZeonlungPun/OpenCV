import cv2
import pickle
import numpy as np

with open('E:\\CarParkProject\\CarParkPos', 'rb') as f:
    posList=pickle.load(f)
width,height=107,48

cap=cv2.VideoCapture('E:\\CarParkProject\\carPark.mp4')

def checkParkingSpace(imgProcessed):
    space_counter=0
    for pos in posList:
        x,y=pos
        imgCrop=imgProcessed[y:y+height,x:x+width]
        #judge whether the parking space has car through non-zero pixel number
        count=cv2.countNonZero(imgCrop)
        cv2.putText(img,str(count),(x,y+height-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,color=(255,0,255),thickness=1)

        if count<800:
            color=(0,255,0)
            thickness=5
            space_counter+=1
        else:
            color = (0, 0,255)
            thickness = 2
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)
    cv2.putText(img, f'free:{space_counter}/{len(posList)}', (100,50), cv2.FONT_HERSHEY_SIMPLEX,2, color=(0, 0, 255), thickness=5)


while True:
    #play the video for loop
    if cap.get(cv2.CAP_PROP_POS_FRAMES)==cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    success,img=cap.read()
    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur=cv2.GaussianBlur(imgGray,(3,3),1)
    imgThreshold=cv2.adaptiveThreshold(imgBlur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,25,16)
    imgMedian=cv2.medianBlur(imgThreshold,5)
    kernel=np.ones((3,3),np.int8)
    imgDilate=cv2.dilate(imgMedian,kernel,iterations=1)


    checkParkingSpace(imgDilate)


    cv2.imshow("image",img)
    key=cv2.waitKey(10)

    if (key==27):#键盘按ESC，则退出
        break
cap.release()
cv2.destroyAllWindows()
