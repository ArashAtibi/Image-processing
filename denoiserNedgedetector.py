import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("images/BSE_Google_noisy.jpg",1) #1 for color, 0 black and white
kernel = np.ones((3,3), np.float32)/9#or 5,5 and then /2 ; can also define gaussian filter instead of ones

filte_2D = cv2.filter2D(img, -1, kernel)#just a Convolution
blur = cv2.blur(img,(3,3))#5,5 is the size of kernel
gaussian_blur = cv2.GaussinBlur(img, (5,5), 0)#so far linear Convolution 
median_blur = cv2.medanBlur(img, 3) #good for preserving edges
bilateral_blur = cv2.BilateralFIlter(img, 9, 75, 75) #best for preserving edges
#nonlocal means filter(nlm) is also a very good filter for microscopic images


img2 = cv2.imread("images/neuron.jpg",0) #1 for color, 0 black and white
edges = cv2.Canny(img2, 100, 200)#canny function and its minimum and maximum values

cv2.imshow("Original", img)
#cv2.imshow("2D custom filter", filte_2D)
#cv2.imshow("2D Blur filter", blur)
#cv2.imshow("2D Gaussian filter", gaussian_blur)
cv2.imshow("2D Median filter", gaussian_blur)
cv2.imshow("2D Bilateral filter", bilateral_blur)


cv2.imshow("Original", img2)
cv2.imshow("edges", edges)

cv2.waitKey(0)#always with imshow
cv2.destroyAllWindows()


