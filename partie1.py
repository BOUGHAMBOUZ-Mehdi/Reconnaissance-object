# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 18:00:43 2018

"""
import numpy as np
import cv2
from matplotlib import pyplot as plt

imgSrc = cv2.imread('C:\\Users\\praxis\\Documents\\Cours\\2018-2019\\VR-RA\\correct\\animauxFantastiques.jpg',cv2.IMREAD_UNCHANGED )
imgDest = cv2.imread('C:\\Users\\praxis\\Documents\\Cours\\2018-2019\\VR-RA\\correct\\times_square.jpg',cv2.IMREAD_UNCHANGED )

# Dimensions de l'image
height, w, channel = imgSrc.shape
#Coins de l'image
src_pts = np.float32([[0, 0], [0, height - 1], [w - 1, height - 1], [w - 1, 0]])
#Coins de la figure dans l'image de destination
dst_pts = np.float32([[1045, 134], [1086, 323], [1224, 297], [1170, 80]])

#Calcul de l'homographie
h, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

#Création d'un masque blanc de la taille de l'image source
mask = np.zeros((height, w, channel), dtype = np.float32) + 1.0

#Application de l'homographie à l'image source
imgReplace = cv2.warpPerspective(imgSrc, h, (imgDest.shape[1], imgDest.shape[0]), flags=cv2.INTER_LANCZOS4, borderValue=(0, 0, 0))#bbox[2], bbox[3]))

#Application de l'homographie à l'image masque, qui est blanche (=1.) uniquement à l'emplacement de l'affiche
#Les bords de l'affiche masque sont automatiquement interpolés pour éviter la pixellisation
mask = cv2.warpPerspective(mask, h, (imgDest.shape[1], imgDest.shape[0]), flags=cv2.INTER_LANCZOS4, borderValue=(0., 0., 0.))#bbox[2], bbox[3]))

plt.imshow(mask)
plt.imshow(imgReplace)
                
#Alpha blending entre les 2 images en utilisant l'image masque comme carte alpha
imgDest = imgDest * (1. - mask) + imgReplace * mask
#Passage flottant -> entier
imgDest = imgDest.astype(np.uint8)

#Ou plus lent, plus classique et responsable d'artéfacts :
#for i in range(0, imgDest.shape[0]):
#    for j in range(0, imgDest.shape[1]):
#        for k in range(0, 3):
#            if not(imgReplace[i,j,0] == 0 and imgReplace[i,j,1] == 0 and imgReplace[i,j,2] == 0):
#                imgDest[i,j,k] = imgReplace[i,j,k]

plt.imshow(imgDest[:,:,::-1])
cv2.waitKey(0)