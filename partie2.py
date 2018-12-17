# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 18:33:50 2018

"""
import numpy as np
import cv2
from matplotlib import pyplot as plt

def detectComputeDescriptorsAndMatch(img1, img2):
    return orbDetector(img1, img2)

def siftDetector(img1, img2):
     # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    return kp1, kp2, good

def orbDetector(img1, img2):
    # Initiate ORB detector
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    
    return kp1, kp2, matches

img = cv2.imread('C:\\Users\\praxis\\Documents\\Cours\\2018-2019\\VR-RA\\correct\\box.png',0)
imgTest = cv2.imread('C:\\Users\\praxis\\Documents\\Cours\\2018-2019\\VR-RA\\correct\\box_in_scene.png',0)
cv2.imshow('frameRef', img)

#Calcule les points d'intérêt, les descripteurs des 2 images et les matche
kp1, kp2, matches = detectComputeDescriptorsAndMatch(img, imgTest)

# Dessiner les 10 premiers matches.
img3 = cv2.drawMatches(img, kp1, imgTest, kp2, matches[:50], None, flags=2)

plt.imshow(img3), plt.show()

#Crée 2 tableaux en utilisant les points d'intéret trouvés et triés dans les
#2images
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

#Calcule l'homographie
h, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

# Dimensions de l'image
height, w = img.shape

#On transforme les 4 coins de l'image source avec l'homographie pour dessiner la 
#boite englobante
pts = np.float32([[0, 0], [0, height - 1], [w - 1, height - 1], [w - 1, 0]]).reshape(-1, 1, 2)
dst = cv2.perspectiveTransform(pts, h)  
# On connecte les 4 coins avec des lignes
imgTest = cv2.polylines(imgTest, [np.int32(dst)], True, 255, 3, cv2.LINE_AA) 
cv2.imshow('frame', imgTest)

#Application de l'homographie à une image quelconque
imgReplace = cv2.imread('C:\\Users\\praxis\\Documents\\Cours\\2018-2019\\VR-RA\\correct\\carte de visite.jpg',0)
#Redimensionnement pour qu'elle ait la même taille que l'image d'origine
imgReplace = cv2.resize(imgReplace,None,fx=np.float32(w) / imgReplace.shape[1] , fy=np.float32(height) / imgReplace.shape[0], interpolation = cv2.INTER_CUBIC)
cv2.imshow('frame2', imgReplace)
imgReplace = cv2.warpPerspective(imgReplace, h, (imgTest.shape[1], imgTest.shape[0]))
cv2.imshow('frame3', imgReplace)

dstInt = np.int32(dst)

for i in range(0, imgTest.shape[0]):
    for j in range(0, imgTest.shape[1]):
        if imgReplace[i,j] != 0:
            imgTest[i,j] = imgReplace[i,j]

cv2.imshow('frame', imgTest)
cv2.waitKey(0)