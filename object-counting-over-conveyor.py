##### ASUMPTIONS ######
#
#   Only one item is passing through the frame at a time
#
#   Conveyer belt is moving from right to left 
#
#	If item goes from right to left then increase in counter or opposite then decrease in counter (in both case by 1)
#
#   Detecting object by adjusting HSV color… I’m taking rectangle object but for taking particular "cylindrical" object we can take ratio between height and width. 
#
########################

import numpy as np
import matplotlib.pyplot as plt
from collections import deque 
import cv2

cap = cv2.VideoCapture(0)

#Ignoring function body since we don't have to perform any action when value changes
def nothing(a):
    pass

#Web cam's video window resize
cv2.namedWindow("Frame",cv2.WINDOW_NORMAL)
cv2.resizeWindow("Frame",200,150)

#Creating new window to adjust HSV min, max value to detect object live
cv2.namedWindow('Adjust-Mask')
cv2.createTrackbar("LH","Adjust-Mask",0,255,nothing)
cv2.createTrackbar("LS","Adjust-Mask",0,255,nothing)
cv2.createTrackbar("LV","Adjust-Mask",190,255,nothing)
cv2.createTrackbar("UH","Adjust-Mask",255,255,nothing)
cv2.createTrackbar("US","Adjust-Mask",100,255,nothing)
cv2.createTrackbar("UV","Adjust-Mask",255,255,nothing)

#loc = deque(maxlen=20)
loc = 0 #To record last position of object
count = 0 #Object counter

while True:

    rtn, frame = cap.read()
    #Filliping frame to remove mirror effect
    frame = cv2.flip(frame,1)
        
    #_,t_frame = cv2.threshold(frame,127,255,0)
    
    #Blur helps in removing noise
    g_blur = cv2.GaussianBlur(frame,(5,5),0)
    
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
    #Taking values from window for lower and upper range
    LH = cv2.getTrackbarPos("LH","Adjust-Mask")
    LS = cv2.getTrackbarPos("LS","Adjust-Mask")
    LV = cv2.getTrackbarPos("LV","Adjust-Mask")
    UH = cv2.getTrackbarPos("UH","Adjust-Mask")
    US = cv2.getTrackbarPos("US","Adjust-Mask")
    UV = cv2.getTrackbarPos("UV","Adjust-Mask")
    lower_r = np.array([LH,LS,LV])
    upper_r = np.array([UH,US,UV])

    #Filtering image through color range to highlight object
    mask = cv2.inRange(g_blur,lower_r,upper_r)

    #Sharping the edges by taking only max value in kernal size
    mask = cv2.erode(mask,None,iterations=2)
    #Sharping the edges by assigning the max value
    mask = cv2.dilate(mask,None,iterations=2)
    
    contrs,hirarchy = cv2.findContours(mask.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
   
    #Process if any contour found
    if len(contrs) > 0:
        #taking contour with largest area
        cntr = max(contrs,key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(cntr)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        #checking directing of moving, assuming object moved if x value is changes more then 50pixel
        loc = x-abs(loc) if abs(x-abs(loc))>50 else loc
        
    elif loc is not 0:
        dir = 'left' if loc < 0 else 'right' 
        #Counter 
        count += 1 if dir == 'left' else -1 
        print(count)
        loc = 0

    cv2.imshow("Frame",frame)
    cv2.imshow("Mask",mask)
    
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
        
    

