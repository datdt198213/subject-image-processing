#import the required libraries
import numpy as np 
import matplotlib.pyplot as plt
import cv2

# %matplotlib inline
image = cv2.imread('index.jpg')

#####################################################################
# converting color
#converting image to Gray scale 
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#showing the grayscale image
#cv2.imshow('gray_image', gray_image)

#converting image to HSV format
# hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#showing the HSV image
#cv2.imshow('hsv_image', hsv_image)

#####################################################################
#image saving
# cv2.imwrite('gray_image.jpg', gray_image)
# cv2.imwrite('hsv_image.jpg', hsv_image)

#####################################################################
#converting image to size (300, 300, 3)
# smaller_image = cv2.resize(image, (300,300),cv2.INTER_AREA)

# print('Resized dimensions: ', smaller_image.shape)
# cv2.imshow('smaller_image',smaller_image)

#####################################################################
# converting 20 percent
# scale_percent = 20
# width = int(image.shape[0] * scale_percent / 100)
# height = int(image.shape[1] * scale_percent / 100)
# dimension = (width, height)

# _20_percent_image = cv2.resize(image, dimension, cv2.INTER_CUBIC)
# cv2.imshow('_20_percent_image', _20_percent_image)
# cv2.imwrite('20_percent_image.jpg', _20_percent_image)

#####################################################################
# converting 40 percent
# scale_percent = 40
# width = int(image.shape[1] * scale_percent / 100)
# height = int(image.shape[0] * scale_percent / 100)
# dimension = (width, height)

# _40_percent_image =  cv2.resize(image, dimension,cv2.INTER_CUBIC)

# print('40 percent of dimensions: ', _40_percent_image.shape)
# cv2.imshow('_40_percent_image', _40_percent_image)
# cv2.imwrite('40_percent_image.jpg', _40_percent_image)

#####################################################################
#image rotation
# rows,cols = image.shape[:2] 
#(col/2,rows/2) is the center of rotation for the image 
# M is the cordinates of the center 
# M = cv2.getRotationMatrix2D((cols/2,rows/2),180,0.5) 
# dst = cv2.warpAffine(image,M,(cols,rows)) 
# cv2.imshow('dst', dst)

#####################################################################
# image translation
# M = np.float32([[1,0,100], [0, 1, 50]]) 
# 100 is width translation, 50 is height translation
# _image_translation = cv2.warpAffine(image, M, (cols, rows))

# cv2.imshow('img', dst)
# cv2.imwrite('image translation.jpg', _image_translation)

#####################################################################
# Affine transformation
# rows, cols, ch = image.shape

# pts1 = np.float32([[50,50], [200,50], [50,200]])
# pts2 = np.float32([[10,100], [200,50], [100,250]])

# M = cv2.getAffineTransform(pts1, pts2)

# image_affine = cv2.warpAffine(image, M, (cols, rows))

#####################################################################
# Perspective Transform
# pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
# pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

# M = cv2.getPerspectiveTransform(pts1, pts2)

# image_perspective = cv2.warpPerspective(image, M, (300, 300	))

# plt.subplot(121), plt.imshow(image), plt.title('Input')
# plt.subplot(122), plt.imshow(image_affine), plt.title('Output')
# plt.show()

######################################################################
# thresholding 
# ret, thresh1 = cv2.threshold(image, 127,255, cv2.THRESH_BINARY)
# ret, thresh2 = cv2.threshold(image, 127,255, cv2.THRESH_BINARY_INV)
# ret, thresh3 = cv2.threshold(image, 127,255, cv2.THRESH_TRUNC)
# ret, thresh4 = cv2.threshold(image, 127,255, cv2.THRESH_TOZERO)
# ret, thresh5 = cv2.threshold(image, 127,255, cv2.THRESH_TOZERO_INV)

# titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
# images = [image, thresh1, thresh2, thresh3, thresh4, thresh5]

# for i in range(6):
# 	plt.subplot(2,3,i+1), plt.imshow(images[i],'gray', vmin = 0, vmax = 255)
# 	plt.title(titles[i])
# 	plt.xticks([]),plt.yticks([])

# plt.show()

#####################################################################
# Adaptive Thresholding
# image = cv2.imread('20_percent_image.jpg')
# image = cv2.medianBlur(image, 5)

# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ret, th1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# th2 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11,2)
# th3 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11,2)

# titles = ['Original Image', 'Global Thresholding (v = 127)', 'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
# images = [image, th1, th2, th3]

# cv2.imshow('image', image)

# for i in range(4):
# 	plt.subplot(2,2,i+1), plt.imshow(images[i],'gray')
# 	plt.title(titles[i])
# 	plt.xticks([]), plt.yticks([])

# plt.show()

#####################################################################
# Otsu's Binarization
image = cv2.imread('index.jpg')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# global thresholding
ret1,th1 = cv2.threshold(gray_image,127,255,cv2.THRESH_BINARY)

# Otsu's thresholding 
ret2,th2 = cv2.threshold(gray_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#Otsu's thresholding after Gaussian filtering
# blur = cv2.GaussianBlur(gray_image, (5,5) , 0)
# ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# images = [image, 0, th1, image, 0, th2, blur, 0, th3]
# titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v = 127)',
# 'Original Noisy Image', 'Histogram', "Otsu's Thresholding",
# 'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding"]

# for i in range(3):
# 	plt.subplot(3,3,i*3+1), plt.imshow(images[i*3], 'gray')
# 	plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
# 	plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
# 	plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
# 	plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
# 	plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
# plt.show()

#####################################################################
# avaraging
# image = cv2.imread('blur_image.jfif')

# scale_percent = 50
# width =int(image.shape[0] * scale_percent / 100) 
# height = int(image.shape[1] * scale_percent/ 100)
# dimension = (width, height)
# percent_image = cv2.resize(image, dimension, cv2.INTER_CUBIC)
# kernel = np.ones((5,5), np.float32)/25
# dst = cv2.filter2D(percent_image, -1, kernel)

# plt.subplot(121), plt.imshow(percent_image), plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(dst), plt.title('Averaging')
# plt.xticks([]), plt.yticks([])
# plt.show()
#####################################################################
# blured
image = cv2.imread('landscape-color.jpg')
# blur = cv2.blur(image, (5,5))
blur = cv2.GaussianBlur(image, (5,5),0)

plt.subplot(121), plt.imshow(image), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(blur),
# plt.title('Blured')
plt.title('Blured Gauss')
plt.xticks([]), plt.yticks([])
plt.show()

#####################################################################

#####################################################################
# key end cv2 running
cv2.waitKey(0)
cv2.destroyAllWindows()