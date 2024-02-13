import cv2
import pickle
import numpy as np
import os
facesDetect=cv2.CascadeClassifier('data\haarcascade_frontalface_default.xml')
video=cv2.VideoCapture(0)
facesData=[]
i=0
name=input("enter your name:")

while True:
    ret,frame=video.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=facesDetect.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cropImage=frame[y:y+h,x:x+w,:]
        resizedImage=cv2.resize(cropImage,[50,50])
        if len(facesData)<=100 and i%10==0:
            facesData.append(resizedImage)
        i+=1
        cv2.putText(frame,str(len(facesData)),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(50,50
        ))
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),1)
    cv2.imshow("frame",frame)
    k=cv2.waitKey(1)
    if(k==ord('q') or len(facesData)==50):
        break
video.release()
cv2.destroyAllWindows()
facesData=np.asarray(facesData)
facesData=facesData.reshape(100,-1)

if 'names.pkl' not in os.listdir('data/'):
    names=[name]*100
    with open ('data/names.pkl','wb') as f:
        pickle.dump(names,f)
else:
    with open ('data/names.pkl','rb') as f:
        names=pickle.load(f)
    names=names+[name]*100
    with open ('data/names.pkl','wb') as f:
        pickle.dump(names,f)

if 'facesData.pkl' not in os.listdir('data/'):
    
    with open ('data/facesData.pkl','wb') as f:
        pickle.dump(facesData,f)
else:
    with open ('data/facesData.pkl','rb') as f:
        faces=pickle.load(f)
    faces=np.append(faces,facesData,axis=0)
    with open ('data/facesData.pkl','wb') as f:
        pickle.dump(names,f)