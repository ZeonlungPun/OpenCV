import cv2
import pickle



width,height=107,48
try:
    with open('CarParkPos','rb') as f:
        posList=pickle.load(f)
except:
    posList=[]

def mouseClick(events,x,y,flags,parms):
    """
    Args:
        events: pass automatically
        x: click the point
        y: click the point
        flags:
        parms:

    Returns: position of car park:top left point
    """
    if events==cv2.EVENT_LBUTTONDOWN:
        posList.append((x,y))
    if events==cv2.EVENT_RBUTTONDOWN:
        for i,pos in enumerate(posList):
            x1,y1=pos
            if x1<x<x1+width and y1<y<y1+height:
                posList.pop(i)

    #write the position information into document
    with open('CarParkPos', 'wb') as f:
        pickle.dump(posList,f)




while True:
    img = cv2.imread('E:\\CarParkProject\\carParkImg.png')
    for pos in posList:
        cv2.rectangle(img,pos,(pos[0]+width,pos[1]+height),(255,0,255),2)
    cv2.imshow("result",img)
    cv2.setMouseCallback('result',mouseClick)
    cv2.waitKey(1)