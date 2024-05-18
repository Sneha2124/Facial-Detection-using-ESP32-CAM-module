import cv2
import urllib.request
import numpy as np


face_cascade=cv2.CascadeClassifier("C:\\Users\piyush30\AppData\Local\Programs\Python\Python310\Lib\site-packages\cv2\haarcascade_frontalface_default.xml")

url='http://192.168.43.239/cam-hi.jpg'
cv2.namedWindow("gotcha",cv2.WINDOW_AUTOSIZE)

while True:
    imgRespose=urllib.request.urlopen(url)

    #Save image locally
    # filename+=1
    # urllib.request.urlretrieve(url, f"./data/file.jpg")
    imgnp=np.array(bytearray(imgRespose.read()),dtype=np.uint8)
    img=cv2.imdecode(imgnp,-1)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face=face_cascade.detectMultiScale(img,scaleFactor=1.1,minNeighbors=5)
    for x,y,w,h in face:
        img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
        cv2.imwrite('./data/file.png',img)

    cv2.imshow("gotcha",img)
    key=cv2.waitKey(5)
    if key==ord('q'):
        break

cv2.destroyAllWindows

