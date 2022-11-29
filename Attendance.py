from operator import index
import cv2
import numpy as np
import face_recognition
import os


from sqlalchemy import true
from datetime import datetime,timedelta
from time import sleep

path="ImagesAttendance"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_face.mp4",fourcc,6.0,(1280,720))
images=[]
classNames=[]
myList=os.listdir(path)

print(myList)

for cl in myList:
    curImg=cv2.imread(f"{path}/{cl}")
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
    
#print(classNames)

def findEncodings(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList    




def markAttendance(name):
    with open("Attendance.csv","r+")as f:
        myDataList=f.readlines()
        nameList=[]
        now=datetime.now()
        dtString=now.strftime("%d/%m/%y")
        timeString=now.strftime("%H:%M:%S")
            
        for line in myDataList:
            entry=line.split(",")
            nameList.append(entry[0])
        if name not in nameList:
            f.writelines(f"\n{name},{dtString},{timeString}")
       
           

encodeListKnown=findEncodings(images)
print("Kodovaní dokončeno")

cap=cv2.VideoCapture(0)

while true:
    success,img=cap.read()
    imgs=cv2.resize(img,(0,0),None,0.25,0.25)
    imgs=cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)

    facesCurFrame=face_recognition.face_locations(imgs)
    encodeCurFrame=face_recognition.face_encodings(imgs,facesCurFrame)

    for encodeFace,faceLoc in zip(encodeCurFrame,facesCurFrame):
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDist=face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDist)
        matchIndex=np.argmin(faceDist)

        if matches[matchIndex]:
            name=classNames[matchIndex].upper()
            #print(name)
            
            y1,x2,y2,x1=faceLoc
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            position = (x2- x1)//4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255.0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,0.85,(255,255,255),2)
            markAttendance(name) 
            
    if cv2.waitKey(1)&0xFF==ord(" "):
        break
    cv2.imshow("webcam",img)
    out.write(img)
    #cv2.waitKey(1)
    






