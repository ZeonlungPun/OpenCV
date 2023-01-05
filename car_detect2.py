import cv2


cap=cv2.VideoCapture("E:\\people.mp4")


def filter_img(frame):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray,(5,5),0)
    _,thresh=cv2.threshold(blur,20,255,cv2.THRESH_BINARY)
    dilated=cv2.dilate(thresh,None,iterations=3)
    return dilated

def center(x,y,w,h):
    x1=int(w/2)
    y1=int(h/2)
    cx=x+x1
    cy=y+y1
    return cx,cy

#帧差分

ret1,frame1=cap.read()
ret2,frame2=cap.read()
print(frame1.shape)
min_area=200
num=0
cars=[]
line_base=700
offset=1
while True:

    if (ret1==True and ret2==True):
        diff=cv2.absdiff(frame1,frame2)
        mask=filter_img(diff)

        contours,_=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        # 画一条基准线
        cv2.line(frame1, (line_base, 0), (line_base, 1272), (255, 255, 0), 3)
        dect=[]
        for contour in contours:
            (x,y,w,h)=cv2.boundingRect(contour)
            if cv2.contourArea(contour)< min_area:
                continue
            cv2.rectangle(frame1,pt1=(x,y),pt2=(x+w,y+h),color=(0,255,0),thickness=2)
            dect.append([x,y,w,h])

            #得到中心点
            points=center(x,y,w,h)


            cars.append(points)
            for (x,y) in cars:
                if (x>line_base-offset) and (x<line_base+offset):
                    num+=1
                    cars.remove((x,y))
                    print(num)


        for bid in box_ids:
            x,y,w,h,id=bid
            cv2.putText(frame1,str(id),(x,y-15),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

        #show
        cv2.putText(frame1,"count:"+str(num),(500,60),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),5)
        cv2.imshow("frame",frame1)

        frame1=frame2
        ret2, frame2 = cap.read()



        if cv2.waitKey(50)==27:
            break
