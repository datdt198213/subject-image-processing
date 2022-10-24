import cv2 as cv
import numpy as np
import csv

image = cv.imread('image.jpg')
cv.imshow('Image', image)

float_img = image.astype("float")
array_double = cv.normalize(float_img, None, 0.0, 1.0, cv.NORM_MINMAX)

print(*array_double[:,:,0])
print(array_double[:,:,1])
print(array_double[:,:,2])

width, height, channels = image.shape

print('Image revolution: ', width, 'x', height, '\tColor: ', channels)

cv.waitKey(0)