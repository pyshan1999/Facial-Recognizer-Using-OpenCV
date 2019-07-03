# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 00:09:05 2019

@author: Akarshan Srivastava
"""

import os
import cv2
from PIL import Image
import numpy as np
import pickle

face_cascade=cv2.CascadeClassifier('cascades\data\haarcascade_frontalface_alt2.xml')
recognizer=cv2.face.LBPHFaceRecognizer_create() 

base_dir=os.path.dirname(os.path.abspath(__file__))
image_dir=os.path.join(base_dir,"images")

# print(base_dir,image_dir)

current_id=0
label_ids={}
y_label=[]
x_train=[]
 
for root,dirs,files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path=os.path.join(root,file)
            label=os.path.basename(root).replace(" ","-").lower()
            #print(label,path)
            
            if not label in label_ids:
                label_ids[label]=current_id
                current_id+=1
            
            id_=label_ids[label]#print(label_ids)
            
            pil_img=Image.open(path).convert("L")
            size=(550,550)
            f_img=pil_img.resize(size,Image.ANTIALIAS)
            img_arr=np.array(f_img,'uint8')
            #print(img_arr)
            faces= face_cascade.detectMultiScale(img_arr,scaleFactor=1.5,minNeighbors=5)
   
            for(x,y,w,h) in faces:
                roi=img_arr[y:y+h,x:x+w]
                x_train.append(roi) 
                y_label.append(id_) 
                
with open("labels.pickle",'wb') as fl:
    pickle.dump(label_ids,fl)
    
recognizer.train(x_train,np.array(y_label))
recognizer.save("trainer.yml")
                