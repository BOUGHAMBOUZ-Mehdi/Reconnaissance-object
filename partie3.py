# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 18:48:10 2018

"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
from ObjLoader import *

ORB = 1
SIFT = 2

def detector(img, algo):
    if algo == ORB:
        return orbDetector(img)
    else:
        return siftDetector(img)
    
def siftDetector(img):
     # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp, desc = sift.detectAndCompute(img, None)
    
    return kp, desc

def orbDetector(img):
    # Initiate ORB detector
    orb = cv2.ORB_create()
    kp, desc = orb.detectAndCompute(img, None)

    return kp, desc

def match(desc1, desc2, algo):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    if algo == ORB:
        # Match descriptors.
        matches = bf.match(desc1, desc2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        
        return matches
    else:
            
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc1,desc2, k=2)
        
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
                
        return good

                        
def render(img, obj, projection, scale, model, color = (130, 25, 200)):
    vertices = obj.vertices
    scale_matrix = np.eye(3) * scale
    scale_matrix[1] = -scale_matrix[1]
    scale_matrix[2] = -scale_matrix[2]
    h, w, channel = model.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        cv2.fillConvexPoly(img, imgpts, color)

    return img
        

def projectionMatrix(camera_parameters, homography):
    #Matrice des paramètres extrinsèques
    M = np.dot(np.linalg.inv(K), homography)

    _lambda1 = np.linalg.norm(M[:, 0])
    _lambda2 = np.linalg.norm(M[:, 1])
    _lambda = (_lambda1 + _lambda2) / np.float32(2)

    M[:, 0] = M[:, 0] / _lambda1
    M[:, 1] = M[:, 1] / _lambda2
    t = M[:, 2] / _lambda
    R = np.c_[M[:, 0], M[:, 1], np.cross(M[:, 0], M[:, 1])]
    #R = np.c_[np.cross(R[:, 1], R[:, 2]), R[:, 1], R[:, 2]]
    
    if cv2.determinant(R) < 0:
        R[:, 2] = R[:, 2] * (-1)

    W, U, Vt = cv2.SVDecomp(R)
    R = U.dot(Vt);
            
    extr = np.float32([[R[0,0], R[0,1], R[0,2], t[0]],[R[1,0], R[1,1], R[1,2], t[1]], [R[2,0], R[2,1], R[2,2], t[2]]])
            
    return np.dot(camera_parameters, extr)    

imgRef = cv2.imread('C:\\Users\\praxis\\Documents\\Cours\\2018-2019\\VR-RA\\video\\ref2Corrigee.jpg',cv2.IMREAD_UNCHANGED )
vid = cv2.VideoCapture('C:\\Users\\praxis\\Documents\\Cours\\2018-2019\\VR-RA\\video\\20181009_140438.mp4')
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
videoWriter=cv2.VideoWriter('C:\\Users\\praxis\\Documents\\Cours\\2018-2019\\VR-RA\\video\\videoooo.avi',fourcc,30,(1280, 720))
obj = OBJ('C:\\Users\\praxis\\Documents\\Cours\\2018-2019\\VR-RA\\correct\\fox2.obj', swapyz=False)

MIN_MATCHES = 10

algo = SIFT
firstFrame = True
drawMatches = False
drawBorder = True

#Paramètres intrinsèques de la caméra du Samsung GS6
#Attention, les images des vidéos (720p ici) n'ont pas la même taille que les 
#images photos et il est donc nécessaire modifier artificiellement la taille pixel
#Valeurs théoriques pour la photo, qui diffèrent dans les séquences vidéo
imageRefSizeX = 5312
imageRefSizeY = 2988
pixelSizeMm = 0.00112 
focalLengthMm = 4.3

K = None

#Points d'intérêt
kpRef = None
descRef = None

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # read the current frame
    ret, frame = vid.read()
    if not ret:
        print "Unable to capture video"
        break 

    if firstFrame:
        kpRef, descRef = detector(imgRef, algo)
        firstFrame = False
        
        #On redimensionne la taille pixel en fonction de la taille des frames vidéo
        pixelSizeMm = pixelSizeMm * (imageRefSizeX / np.float32(frame.shape[1]))
        fpixel = focalLengthMm / pixelSizeMm
        K = np.float32([[fpixel, 0, frame.shape[1] / np.float32(2)],[0, fpixel, frame.shape[0] / np.float32(2)], [0,0,1]])
        
    kpFrame, descFrame = detector(frame, algo)
    matches = match(descRef, descFrame, algo)
        
    # On calcule l'homographie si on trouve assez de matches
    if len(matches) > MIN_MATCHES:        
        src_pts = np.float32([kpRef[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kpFrame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        h, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)#,5.0)
        
        if h is not None:
            # Taille de l'image
            height, w, channel = imgRef.shape
            pts = np.float32([[0, 0], [0, height - 1], [w - 1, height - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            
            #Dessine la boîte englobante du plan de référence détecté
            if drawBorder:
                dst = cv2.perspectiveTransform(pts, h)
                frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA) 
                
            #Dessine les points de matching calculés
            if drawMatches:
                # Dessine les 50 premiers matches
                frame = cv2.drawMatches(imgRef,kpRef,frame,kpFrame,matches[:50], None, flags=2)
                frame = cv2.resize(frame,(1280,720))
            
            #Calcule la matrice de projection
            h2 = projectionMatrix(K, h)
            
            #Projette le modèle 3D dans l'image
            frame = render(frame, obj, h2, imgRef)
            cv2.imshow('frame3', frame)
            cv2.waitKey(5)
#            plt.imshow(frame[:,:,::-1])
#            plt.pause(0.05)
            
            videoWriter.write(frame)
            
vid.release()
videoWriter.release()
cv2.destroyAllWindows()
print "Fini" 