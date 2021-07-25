#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
from PIL import Image
import sys


# In[2]:


print("Enter cascade file Location:")
cascade_file=input()


# In[3]:


cascade_file


# In[4]:


face_cascade = cv2.CascadeClassifier(cascade_file)
#'D:/All_AI_Related/facedetect_config/haarcascade_frontalface_default.xml'


# In[5]:


cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, img = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face=cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face_img=Image.fromarray(face)
        #resized_face_image=np.array(face_img.resize((64,64),resample=Image.NEAREST))
        #resized_face_image=np.expand_dims(resized_face_image,axis=0)
        #print(resized_face_image.shape)
        #print(localModel.predict(resized_face_image))
    # Display
    cv2.imshow('img', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


# In[ ]:




