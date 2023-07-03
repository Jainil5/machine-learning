import os
import tensorflow as tf
import uuid
import time
import cv2

IMAGES_PATH = os.path.join("faceDetection\data","images")
number_images = 20

cap = cv2.VideoCapture(0)
for imgnum in range(number_images):
    print("Collecting Images{}".format(imgnum))
    ret,frame = cap.read()
    imgname = os.path.join(IMAGES_PATH,f"{str(uuid.uuid1())}.jpg")
    cv2.imwrite(imgname,frame)
    cv2.imshow("frame",frame)
    time.sleep(1)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()

cv2.destroyAllWindows()    

