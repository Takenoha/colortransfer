import cv2
import sys
import os

#im = cv2.imread(sys.argv[1])
#filename = os.path.splitext(os.path.basename(sys.argv[1]))[0]
filename = 'thumbnail.webp'
im = cv2.imread(filename)
im_rgbgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
cv2.imwrite(filename+'rgb.jpg', im_rgbgray)

im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
hsvh,hsvs,hsvv = cv2.split(im_hsv)
cv2.imwrite(filename+'hsv.jpg', hsvv)

im_lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
labl,laba,labb = cv2.split(im_lab)
cv2.imwrite(filename+'lab.jpg', labl)
