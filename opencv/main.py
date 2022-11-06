from multipledispatch import dispatch
from mpl_toolkits.mplot3d import Axes3D
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from collections import Counter
import cv2

image = cv2.imread('landscape-color.jpg')
#image saving
def saveImage(image, file_name):
	print("Save successfully!")
	cv2.imwrite(file_name, image)

# saveImage(image, 'Test.jpg')

#convert image color
def convertColorImage(image, channel):
	if(channel == "GRAY"):
		cvt_color_image = cv2.cvtColor(src = image, code = cv2.COLOR_BGR2GRAY)	
	if(channel == "HSV"):
		cvt_color_image = cv2.cvtColor(src = image, code = cv2.COLOR_BGR2HSV)
	if(channel == "RGB"):
		cvt_color_image = cv2.cvtColor(src = image, code = cv2.COLOR_BGR2RGB)
	return cvt_color_image

# gray_image = convertColorImage(image, "HSV")
# cv2.imshow("gray_image", gray_image)

# Resize image
@dispatch(np.ndarray, int)
def resizeImage(image, percent):
	width = int(image.shape[0] * percent / 100)
	height = int(image.shape[1] * percent / 100)
	dimension = (width, height)
	resized_image = cv2.resize(src = image,dsize = dimension,interpolation = cv2.INTER_CUBIC)
	return resized_image

@dispatch(np.ndarray, int, int)
def resizeImage(image, width, height):
	dimension = (width, height)
	resized_image = cv2.resize(src = image,dsize = dimension,interpolation = cv2.INTER_CUBIC)
	return resized_image

# image = resizeImage(image, 200, 200)
# cv2.imshow("image 20 percen", image)

#image rotation
def rotateImage(image, angle, scale):
	width, height = image.shape[:2]
	dimension = (width, height)
	center = (width/2, height/2)
	rotate_matrix = cv2.getRotationMatrix2D(center = center, angle = angle, scale = scale) 
	rotated_image = cv2.warpAffine(src = image, M = rotate_matrix, dsize = dimension)
	return rotated_image

# image_rotation = rotateImage(image, 45, 0.5)
# cv2.imshow("image_rotation", image_rotation)

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
# image_perspective = cv2.warpPerspective(image, M, (300, 300))

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
# image = cv2.imread('index.jpg')

# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# global thresholding
# ret1,th1 = cv2.threshold(gray_image,127,255,cv2.THRESH_BINARY)

# Otsu's thresholding 
# ret2,th2 = cv2.threshold(gray_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

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

# blured
def displayImage(image, subplot, title):
	cvt_color_image = convertColorImage(image, "RGB")
	plt.subplot(subplot), plt.imshow(cvt_color_image), plt.title(title)
	plt.xticks([]), plt.yticks([])

def displayTwiceImages(image1, subplot1, title1, image2, subplot2, title2):
	displayImage(image1, subplot1, title1)
	displayImage(image2, subplot2, title2)	
	plt.show()

def displayImages(images = [], subplots = [], titles = []):
	for i in range(len(images)):
		displayImage(images[i], subplots[i], titles[i])
	plt.show()

def drawSpectrum(image, subplot, title):
	image_tmp = convertColorImage(image, "GRAY")
	array = []
	for i in range(0,image_tmp.shape[0]):
		for j in range(0,image_tmp.shape[1]):
			pixel = image_tmp.item(i, j)
			array.append(pixel)
	labels, values = zip(*Counter(array).items())
	width = 1
	plt.subplot(subplot)
	plt.bar(labels, values, width)
	# plt.xlabel('Pixel values', fontsize=10)
	# plt.ylabel('Quantity', fontsize=10)
	plt.title(title, fontsize=10)

def drawSpectrums(images = [], subplots = [], titles = []):
	for i in range(len(images)):
		drawSpectrum(images[i], subplots[i], titles[i])
	plt.show()

def visualizeRGB(image, title):
	r, g, b = cv2.split(image)
	fig = plt.figure()
	axis = fig.add_subplot(1, 1, 1, projection="3d")

	pixel_colors = image.reshape((np.shape(image)[0]*np.shape(image)[1], 3))
	norm = colors.Normalize(vmin=-1.,vmax=1.)
	norm.autoscale(pixel_colors)
	pixel_colors = norm(pixel_colors).tolist()

	axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
	axis.set_xlabel("Red")
	axis.set_ylabel("Green")
	axis.set_zlabel("Blue")
	plt.title(title)

def visualizeRGBs(images = [], titles = []):
	for i in range(len(images)):
		visualizeRGB(images[i], titles[i])
	plt.show()

def main():
	image = cv2.imread('image-noise.jpg')
	width,height = image.shape[:2]
	
	kernel_size = (3,3)
	kernel = np.ones(kernel_size, np.float32)/(kernel_size[0] * kernel_size[1])
	gaussian_kernel = cv2.getGaussianKernel(ksize = 9, sigma = 5)

	gaussian_pyramid = cv2.pyrDown(src = image)
	blur = cv2.blur(src = image, ksize = kernel_size)
	median_blur = cv2.medianBlur(src = image, ksize = 5)
	laplacian_filter = cv2.Laplacian(src = image, ddepth = -1)
	filter2D = cv2.filter2D(src = image, ddepth = -1, kernel = kernel)
	box_filter = cv2.boxFilter(src = image, ddepth = -1, ksize = kernel_size, normalize=False)	
	gaussian_blur = cv2.GaussianBlur(src = image, ksize = kernel_size, sigmaX = 0, sigmaY = 0)
	bilateral_filter = cv2.bilateralFilter(src = image, d = 5,  sigmaColor = 75, sigmaSpace = 75)

	images = [image, blur, gaussian_blur, median_blur, box_filter, bilateral_filter, filter2D, laplacian_filter, gaussian_pyramid]
	subplots = [331,332,333,334,335,336,337,338,339]
	titles = ["Origin Image", "Blur", "Gaussian blur", "Median blur", "Box filter", "Bilateral filter", "Filter2D", "Laplacian", "Gaussian pyramid"]

	# displayImages(images, subplots, titles)
	# drawSpectrums(images, subplots, titles)
	
	# images = [image, blur]
	# titles = ["Origin", "Blur"]
	# visualizeRGBs(images, titles)

	# difference = cv2.subtract(image, blur)
	# print(*difference[:,:,0])

main()

# key end cv2 running
cv2.waitKey(0)
cv2.destroyAllWindows()