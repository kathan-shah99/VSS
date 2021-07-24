import cv2
import numpy as np

video = cv2.VideoCapture("/home/kathan/Desktop/Cruise_control/Media/video1.avi")
#video = cv2.VideoCapture("/dev/video0")
detector = cv2.createBackgroundSubtractorMOG2(history=500,varThreshold=60,detectShadows=False)
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2));

if not video.isOpened():
    print("Could not open video")
    exit()

while True:
   #capture frame by frame
   ret , frame =  video.read()
   print(frame.shape)
   #cv2.resize(frame, (480, 480),interpolation = cv2.INTER_NEAREST)
   
   # if frame is read correctly ret is True
   if not ret:
       print("Can't receive frame")
       break
   
   gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
   fgmask = detector.apply(gray)
   #fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel);

   contours,_ = cv2.findContours(fgmask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
   
   for cnt in contours:
       if cv2.contourArea(cnt) > 100:
           rect = cv2.minAreaRect(cnt)
           box = cv2.boxPoints(rect)
           box = np.int0(box)
           cv2.rectangle(frame,(box[0][0],box[0][1]),(box[2][0],box[2][1]),(0,0,255),2)
           
    

   cv2.imshow("frame",frame)
   cv2.imshow("mask",fgmask)
   
   if cv2.waitKey(60) == ord('q'):
       break
    
video.release()    	#release the captured frame object at end
cv2.destroyAllWindows()  
