import cv2
from PIL import Image
import numpy as np

image = cv2.imread("Image Processing/pic1.jpg")
img = cv2.resize(image,(480,480))
print(image.shape)

# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# print(gray.shape)
# cv2.imshow("Gray",gray)


# Resize
# res = cv2.resize(gray,(128,128))
# #cv2.imshow("Resize",res)
# # Resize

# res2 = cv2.resize(gray,(64,64))
# #cv2.imshow("Resize2",res2)

# res3 = cv2.resize(res,(32,32))
# #cv2.imshow("Resize 3",res3)

# print(res3.shape)

# blur = cv2.GaussianBlur(img,(7,7),cv2.BORDER_DEFAULT)
# cv2.imshow("Blur",blur)



# #Edge Cascade by canny edge detection
# canny = cv2.Canny(res,125,175)
# cv2.imshow("Canny",canny)

# kernel = np.ones((2,2),np.uint8)

# #Erosion
# erode = cv2.erode(canny,kernel, iterations=1)
# cv2.imshow("Erode",erode)

# #Dilation
# dilated = cv2.dilate(canny,kernel,iterations=1)
# cv2.imshow("Dilate",dilated)


#Denoising image.  Adjusts pixel by taking mean pizels in a particular area

#denoise = cv2.fastNlMeansDenoisingColored(img,None,20,20,7,15)
#cv2.imshow("Denoise",denoise)

# Finding color / COLOR DETECTION

hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV_FULL)

lower_blue = np.array([65,0,0])
upper_blue = np.array([110,255,255])

mask = cv2.inRange(hsv,lower_blue,upper_blue)

cv2.imshow("mask",mask)

cv2.waitKey(0)
