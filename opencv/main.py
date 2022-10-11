#import the required libraries
import numpy as np 
import matplotlib.pyplot as plt
import cv2

# %matplotlib inline
image = cv2.imread('index.jpg')

#converting image to Gray scale 
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#showing the grayscale image
#cv2.imshow('gray_image', gray_image)

#converting image to HSV format
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#showing the HSV image
#cv2.imshow('hsv_image', hsv_image)

#image saving
cv2.imwrite('gray_image.jpg', gray_image)
cv2.imwrite('hsv_image.jpg', hsv_image)

#converting image to size (300, 300, 3)
smaller_image = cv2.resize(image, (300,300),cv2.INTER_AREA)

# print('Resized dimensions: ', smaller_image.shape)
# cv2.imshow('smaller_image',smaller_image)

scale_percent = 20
width = int(image.shape[0] * scale_percent / 100)
height = int(image.shape[1] * scale_percent / 100)
dimension = (width, height)

_20_percent_image = cv2.resize(image, dimension, cv2.INTER_CUBIC)
# cv2.imshow('_20_percent_image', _20_percent_image)
# cv2.imwrite('20_percent_image.jpg', _20_percent_image)

#converting image to size ratio 60%
scale_percent = 40
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dimension = (width, height)

_40_percent_image =  cv2.resize(image, dimension,cv2.INTER_CUBIC)

# print('40 percent of dimensions: ', _40_percent_image.shape)
# cv2.imshow('_40_percent_image', _40_percent_image)
# cv2.imwrite('40_percent_image.jpg', _40_percent_image)

#image rotation
rows,cols = image.shape[:2] 
#(col/2,rows/2) is the center of rotation for the image 
# M is the cordinates of the center 
M = cv2.getRotationMatrix2D((cols/2,rows/2),180,0.5) 
dst = cv2.warpAffine(image,M,(cols,rows)) 
# cv2.imshow('dst', dst)

# image translation
M = np.float32([[1,0,100], [0, 1, 50]]) 
# 100 is width translation, 50 is height translation
_image_translation = cv2.warpAffine(image, M, (cols, rows))

# cv2.imshow('img', dst)
# cv2.imwrite('image translation.jpg', _image_translation)

# Affine transformation
rows, cols, ch = image.shape

pts1 = np.float32([[50,50], [200,50], [50,200]])
pts2 = np.float32([[10,100], [200,50], [100,250]])

M = cv2.getAffineTransform(pts1, pts2)

image_affine = cv2.warpAffine(image, M, (cols, rows))

# Perspective Transform
pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

M = cv2.getPerspectiveTransform(pts1, pts2)

image_perspective = cv2.warpPerspective(image, M, (300, 300	))

# plt.subplot(121), plt.imshow(image), plt.title('Input')
# plt.subplot(122), plt.imshow(image_affine), plt.title('Output')
# plt.show()

# thresholding 
ret, thresh1 = cv2.threshold(image, 127,255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(image, 127,255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(image, 127,255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(image, 127,255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(image, 127,255, cv2.THRESH_TOZERO_INV)

titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [image, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(6):
	plt.subplot(2,3,i+1), plt.imshow(images[i],'gray', vmin = 0, vmax = 255)
	plt.title(titles[i])
	plt.xticks([]),plt.yticks([])

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()