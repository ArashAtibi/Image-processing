# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 01:47:02 2022

@author: Arash
Import 2 images
ORB detector
keypoints and their description 
Match keypoints - Brute force matcher
extract good points -- RANSAC (reject unworthy keypoints)
registration (using homplogy)


"""

import cv2
import numpy as np

path1 = r'C:\Users\Arash\Pictures\Gnoisy_distorted.jpg'

path2 = r'C:\Users\Arash\Pictures\Gnoisy.png'
im1 = cv2.imread(path1,1) #reference image 1 for color, 0 black and white
im2 = img = cv2.imread(path2,1) # to be registered 1 for color, 0 black and white

img1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)# convert to gray level images
img2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
img1 = cv2.medianBlur(img1, 3) #good for preserving edges
img2 = cv2.medianBlur(img2, 3) #good for preserving edges


#img2 = cv2.imread("images/neuron.jpg",0) #1 for color, 0 black and white
#edges = cv2.Canny(img2, 100, 200)#canny function and its minimum and maximum values
orb = cv2.ORB_create(90)

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)


matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

#MATCH DESCRIPTORS

matches = matcher.match(des1, des2, None)

matches = sorted(matches, key = lambda x:x.distance )


points1 = np.zeros((len(matches), 2), dtype=np.float32)
points2 = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
    points1[i,:] = kp1[match.queryIdx].pt
    points2[i,:] = kp2[match.trainIdx].pt

h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
#use homography

height, width, channels = im2.shape

im1Reg = cv2.warpPerspective(im1, h, (width, height))




img5 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None )




#img3 = cv2.drawKeypoints(img1, kp1, None, flags=None)
#img4 = cv2.drawKeypoints(img2, kp2, None, flags=None)




cv2.imshow("match", img5)

cv2.imshow("Original", img2)
cv2.imshow("to be registered", im1Reg)
#cv2.imshow("2D Blur filter", blur)
#cv2.imshow("2D Gaussian filter", gaussian_blur)
#cv2.imshow("2D Median filter", gaussian_blur)
#cv2.imshow("2D Bilateral filter", bilateral_blur)


#cv2.imshow("Original", img2)
#cv2.imshow("edges", edges)

cv2.waitKey(0)#always with imshow
cv2.destroyAllWindows()