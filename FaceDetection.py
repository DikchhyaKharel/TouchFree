#!/usr/bin/env python
# coding: utf-8

# In[11]:


get_ipython().system('pip3 install opencv-python')


# In[1]:


get_ipython().system('pip install opencv-contrib-python')


# In[2]:


import os
os.sys.path


# In[5]:


pip install opencv-python


# In[1]:


pip list


# In[1]:


import cv2


# In[2]:


import os
file_isfound = os.path.isfile("penguin.jpg")
if(file_isfound is True):
    print("File is Present")
else:
    print("File is not present")


# In[3]:


img_grayscale = cv2.imread('images/penguin.jpg',0)


# In[4]:


img_grayscale.shape


# In[5]:


print(img_grayscale)


# In[6]:


cv2.imshow("Grayscale Image",img_grayscale)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[7]:


img_rgb = cv2.imread('images/penguin.jpg')


# In[8]:


img_rgb.shape


# In[9]:


print(img_rgb)


# In[10]:


cv2.imshow("Grayscale Image",img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[11]:


img_resized = cv2.resize(img_rgb,(int(img_rgb.shape[1]/2),int(img_rgb.shape[0]/2)))


# In[12]:


cv2.imshow("Resized Image",img_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[13]:


img_resized.shape


# In[14]:


import pkg_resources


# In[15]:


haar_xml = pkg_resources.resource_filename('cv2', 'data/haarcascade_frontalface_default.xml')


# In[16]:


face_cascade = cv2.CascadeClassifier(haar_xml)


# In[17]:


img = cv2.imread('images/harry-potter.jpg')


# In[18]:


cv2.imshow("Input image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[19]:


gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# In[20]:


cv2.imshow("Grayscale Image", gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[21]:


faces = face_cascade.detectMultiScale(gray_img, scaleFactor = 1.05, minNeighbors = 10)


# In[22]:


print(type(faces))


# In[23]:


print(faces)


# In[24]:


for x,y,w,h in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)


# In[25]:


cv2.imshow("Rectangled Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[34]:


import time


# In[27]:


import cv2,time
video = cv2.VideoCapture(0)
check, frame = video.read()
print(check)
print(frame)
time.sleep(3)
video.release()


# In[28]:


import cv2,time
video = cv2.VideoCapture(0)
check, frame = video.read()

time.sleep(3)

cv2.imshow('Capturing',frame)
cv2.waitKey(0)

video.release()
cv2.destroyAllWindows()


# In[30]:


import cv2,time
video = cv2.VideoCapture(0)
a = 1
while True:
    a = a+1
    check, frame = video.read()
    print(frame)

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    cv2.imshow('Capturing',gray)
    key = cv2.waitKey(1)
    
    if key == ord('q'):
        break

print(a)
video.release()
cv2.destroyAllWindows()

