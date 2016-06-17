#################################################################
#                                                               #
#   By Jesper H. Christensen                                    #
#   jesper@haahrchristensen.dk                                  #
#                                                               #
#   Script is developed to detect and locate the Leader robot   #
#   and the tracking point in an image.                         #
#                                                               #
#   Developed as a part of the B.Eng project:                   #
#   Vision Based Control of Collaboring Mobile robots           #
#                                                               #
#################################################################

import numpy as np
import cv2
from matplotlib import pyplot as plt

# load source image
src = cv2.imread('/home/jesper/Dropbox/Diplom Elektro/7. semester/Bachelor/Images/run1/raw28.png')

# get shape from image
height, width = src.shape[:2]

# print shape
print "src height: ", height
print "src width: ", width

# show source image
cv2.imshow('1',src)
cv2.waitKey(0)

# resizing source to fourth the size
resized = cv2.resize(src,(width/4,height/4))

# show resized image
cv2.imshow('2',resized)
cv2.waitKey(0)


# convert to HSV
hsv = cv2.cvtColor(resized.copy(),cv2.COLOR_BGR2HSV)

# show HSV image
cv2.imshow('3',hsv)
cv2.waitKey(0)

# threshold HSV image
thresh = cv2.inRange(hsv, (0,120,90),(255,255,255))

# show thresholded image
cv2.imshow('4',thresh)
cv2.waitKey(0)

# get structuring element for opening
structure = cv2.getStructuringElement(cv2.MORPH_RECT, (resized.shape[0]/7,1))

# open image
open = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,structure)

# show open image
cv2.imshow('5',open)
cv2.waitKey(0)

# get structuring element for closing
structure = cv2.getStructuringElement(cv2.MORPH_RECT, (resized.shape[0]/5,1))

# close image
closed = cv2.morphologyEx(open,cv2.MORPH_CLOSE,structure)

# show closed image
cv2.imshow('6',closed)
cv2.waitKey(0)

# get contours on closed image
im2, cnts, hierachy = cv2.findContours(closed.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

m = 0       # variable used for storing the V-mean-value
i = 0       # counter
mIdx = 0    # used for storing index of highest V-mean-value contour

# loop through contours
for c in cnts:

    # get bounding rectangle of contour
    (x,y,w,h) = cv2.boundingRect(c)

    # throw away is width is too small
    if w > 50:

        # crop the HSV image using bounding rectangle of contour
        croppedContour = hsv[y:y+h,x:x+w]

        # iterate till largest mean value has been found
        if cv2.mean(croppedContour)[2] > m:

            # update with a new higher value
            m = cv2.mean(croppedContour)[2]

            # update index to correspond with m
            mIdx = i

    i = i + 1

# get the bounding rectangle of the contour with highest V-mean-value
(x,y,w,h) = cv2.boundingRect(cnts[mIdx])

# draw rectangle on full sized image
cv2.rectangle(src, (x*4,y*4),(x*4+w*4,y*4+h*4), (0,255,0),2,8,0)

# show source image with rectangle
cv2.imshow('7',src)
cv2.waitKey(0)

# crop full sized image using the bounding rectangle of the used contour
cropped = src[y*4:(y+h)*4,x*4:(x+w)*4]

# show cropped image (should contain the red band)
cv2.imshow('8',cropped)
cv2.waitKey(0)

# get the shape of the cropped image
croppedHeight, croppedWidth = cropped.shape[:2]

# convert cropped image to HSV
redHSV = cv2.cvtColor(cropped,cv2.COLOR_BGR2HSV)

# show HSV image
cv2.imshow('9',redHSV)
cv2.waitKey(0)

# threshold image
redThresh = cv2.inRange(redHSV,(0,120,90),(255,255,255))

# show image
cv2.imshow('10',redThresh)
cv2.waitKey(0)

# get structuring element to open image
structure = cv2.getStructuringElement(cv2.MORPH_RECT, (croppedHeight/5,croppedHeight/5))

# open image
open = cv2.morphologyEx(255-redThresh,cv2.MORPH_OPEN,structure)

# show open image
cv2.imshow('11',open)
cv2.waitKey(0)

# find contours on image
im2, cnts, hierachy = cv2.findContours(255-open.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# sort contours by area
cnts = sorted(cnts, key = cv2.contourArea, reverse=True)

# throw away largest (will be the red band contour)
cnts = cnts[1:len(cnts)]

isSquare = False    # variable showing if the tracking point has been deteceted
count = 0           # counter

# loop through contours
for c in cnts:

    # approximate contours using as few points as possible
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.05*peri, True)

    # get how many points is used
    length = len(approx)

    # throw away contour if area is too small
    if cv2.contourArea(c) > 50:

        # check if rectangle
        if length == 4:

            # get image moment of contour
            M = cv2.moments(c)

            # use moments to calculate mass centres
            centX = int(M["m10"] / M["m00"])+x*4
            centY = int(M["m01"] / M["m00"])+y*4

            # get bounding rectangle of contour
            xR, yR, wR, hR = cv2.boundingRect(c)

            # draw bounding rectangle on source image
            cv2.rectangle(src, (xR+x*4,yR+y*4), ((xR+x*4)+wR,(yR+y*4)+hR),(0,255,0),1,8,0)

            # draw mass centre on source image
            cv2.circle(src, (centX, centY), 2, (0,255,0),-1)

            # set tracking point flag
            isSquare = True
            break

    count = count + 1

# print the result
if isSquare == True:
    print "Square!"
else:
    print "No Square!"

# show source image
cv2.imshow('12',src)
cv2.waitKey(0)


