# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import numpy as np
import pickle as pk

#Initialzing the Face Cascades
face_cascade=cv2.CascadeClassifier('cascades\data\haarcascade_frontalface_alt2.xml')
eye_cascade=cv2.CascadeClassifier('cascades\data\haarcascade_eye.xml')
smile_cascade=cv2.CascadeClassifier('cascades\data\haarcascade_smile.xml')
#Initializing the Recognizer
recognizer=cv2.face.LBPHFaceRecognizer_create() 
recognizer.read("trainer.yml")#Training the model

labels={"Name" : 1}
                
with open("labels.pickle",'rb') as fl:
    nlabels=pk.load(fl)
    labels={v:k for k,v in nlabels.items()}

cap =cv2.VideoCapture(0)

while(True):
    #capture frame by frame
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) 
    faces= face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
     
        
    for (x,y,w,h) in faces:
        #q9print(x,y,w,h)
        roi_g=gray[y:y+h,x:x+w]#(ycord_start,ycord_end)
        roi_color=frame[y:y+h,x:x+w]
        id_,conf=recognizer.predict(roi_g)
        if conf>=45 and conf<=85:
            print(id_)
            print(labels[id_])
            font=cv2.FONT_HERSHEY_SIMPLEX
            name=labels[id_]
            color=(0,255,100)
            stroke=1
            cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)
            
        cv2.imwrite('my-image.jpg',roi_g)
        
        color=(255,0,128)
        stroke=2
        #Draw rectangle around faces
        cv2.rectangle(frame,(x,y),(x+w,y+h),color,stroke)
        eyes=eye_cascade.detectMultiScale(roi_g)
        for(a,b,c,d) in eyes:
            cv2.rectangle(roi_color,(a,b),(a+c,b+d),(0,13,255),2)
    
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break;
    
        
#release the cap

cap.release()
cv2.destroyAllWindows()