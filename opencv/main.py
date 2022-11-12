from multipledispatch import dispatch
from mpl_toolkits.mplot3d import Axes3D
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from collections import Counter
import cv2

# Save Image
def saveImage(image, file_name):
	print("Save successfully!")
	cv2.imwrite(file_name, image)

# Resize image
@dispatch(np.ndarray, int)
def resizeImage(image, percent):
	width = int(image.shape[1] * percent / 100)
	height = int(image.shape[0] * percent / 100)
	dimension = (width, height)
	resized_image = cv2.resize(src = image,dsize = dimension,interpolation = cv2.INTER_CUBIC)
	return resized_image

@dispatch(np.ndarray, tuple)
def resizeImage(image, dimension):
	resized_image = cv2.resize(src = image,dsize = dimension,interpolation = cv2.INTER_CUBIC)
	return resized_image

#convert image color
def convertColorImage(image, channel):
	if(channel == "GRAY"):
		cvt_color_image = cv2.cvtColor(src = image, code = cv2.COLOR_BGR2GRAY)	
	elif(channel == "HSV"):
		cvt_color_image = cv2.cvtColor(src = image, code = cv2.COLOR_BGR2HSV)
	elif(channel == "RGB"):
		cvt_color_image = cv2.cvtColor(src = image, code = cv2.COLOR_BGR2RGB)
	elif(channel == "YCC"):
		cvt_color_image = cv2.cvtColor(src = image, code = cv2.COLOR_BGR2YCrCb)
	elif(channel == "HLS"):
		cvt_color_image = cv2.cvtColor(src = image, code = cv2.COLOR_BGR2HLS)
	elif(channel == "LUV"):
		cvt_color_image = cv2.cvtColor(src = image, code = cv2.COLOR_BGR2Luv)
	return cvt_color_image

def displayImage(image, subplot, title):
	cvt_color_image = convertColorImage(image, "RGB")
	plt.subplot(subplot), plt.imshow(cvt_color_image), plt.title(title)
	plt.xticks([]), plt.yticks([])

def displayImages(images = [], subplots = [], titles = []):
	for i in range(len(images)):
		displayImage(images[i], subplots[i], titles[i])
	plt.show()

def displayColorImagesConversion(image):
	gray_image = convertColorImage(image, "GRAY")
	hsv_image = convertColorImage(image, "HSV")
	rgb_image = image
	ycc_image = convertColorImage(image, "YCC")
	hls_image = convertColorImage(image, "HLS")
	luv_image = convertColorImage(image, "LUV")

	images = [gray_image, hsv_image, rgb_image, ycc_image, hls_image, luv_image]
	subplots = [231,232,233, 234, 235, 236]
	titles = ["GRAY", "HSV", "RGB", "YCC", "HLS", "LUV"]
	displayImages(images, subplots, titles)

def displayResizeImages(image, percent, dimension, title1, title2):
	image_resized_p = resizeImage(image, percent)
	image_resized_d = resizeImage(image, dimension)

	cv2.imshow(title1, image_resized_p)
	cv2.imshow(title2, image_resized_d)

def displayHistogram(image, subplot, title):
	hist,bins = np.histogram(image.flatten(),256,[0,256])
	cdf = hist.cumsum()
	cdf_normalized = cdf * float(hist.max()) / cdf.max()
	
	plt.subplot(subplot), plt.title(title)
	plt.plot(cdf_normalized, color = 'b')
	plt.hist(image.flatten(),256,[0,256], color = 'r')
	plt.xlim([0,256])
	plt.legend(('cdf','histogram'), loc = 'upper left')

def displayHistograms(images = [], subplots = [], titles = []):
	for i in range(len(images)):
		displayHistogram(images[i], subplots[i], titles[i])
	plt.show()

def displayEqualizedHistogram(image):
	image = convertColorImage(image, "GRAY")
	dst = cv2.equalizeHist(image)
	
	images = [image, dst]
	subplots = [121, 122]
	titles = ["Original Image Histogram", "Equalized Image Histogram"]
	displayHistograms(images, subplots, titles)

def displayHistogramMatching(image):
	colors = ('b','g','r')
	for i,color in enumerate(colors):
		histr = cv2.calcHist([image],[i],None,[256],[0,256])
		plt.plot(histr,color = color)
		plt.xlim([0,256])
	plt.show()

def lowPassFilter(image, ksize):
	dst_blur = cv2.blur(src = image, ksize = ksize)
	dst_gaussian = cv2.GaussianBlur(src = image, ksize = ksize, sigmaX = 0, sigmaY = 0)

	images = [image, dst_blur, dst_gaussian]
	subplots = [131, 132, 133]
	titles = ["Original Image" ,"Median filter", "Gaussian filter"]
	displayImages(images, subplots, titles)

def medianFilter(image, ksize):
	dst_median = cv2.medianBlur(src = image, ksize = ksize)

	images = [image, dst_median]
	subplots = [121, 122]
	titles = ["Original Image" ,"Median Filter Image"]
	displayImages(images, subplots, titles)

def highPassFilter(image, kernel):
	dst_sharpen=cv2.filter2D(src = image,ddepth = -1,kernel = kernel)

	images = [image, dst_sharpen]
	subplots = [121, 122]
	titles = ["Original Image" ,"Sharpened Filter Image"]
	displayImages(images, subplots, titles)

def filterMinAndMax(image, kernel):
	dst_erosion = cv2.erode(src = image, kernel = kernel, iterations=1)
	dst_dilation = cv2.dilate(src = image, kernel = kernel, iterations=1)

	images = [image, dst_erosion, dst_dilation]
	subplots = [131, 132, 133]
	titles = ["Original Image" ,"Min Filter Image", "Max Filter Image"]
	displayImages(images, subplots, titles)

def filterCloseAndOpen(image, kernel):
	dst_opening = cv2.morphologyEx(src = image, op = cv2.MORPH_OPEN, kernel = kernel)
	dst_closing = cv2.morphologyEx(src = image, op = cv2.MORPH_CLOSE, kernel = kernel)

	images = [image, dst_opening, dst_closing]
	subplots = [131, 132, 133]
	titles = ["Original Image" ,"Opening Filter Image", "Closing Filter Image"]
	displayImages(images, subplots, titles)

def main():
	image = cv2.imread("car.jpg")
	
	# displayColorImagesConversion(image)
	# displayResizeImages(image, 50, (200,200), "Scale 50%", "200x200")
	# displayEqualizedHistogram(image)
	# displayHistogramMatching(image)
	# lowPassFilter(image, (5,7))
	# medianFilter(image, 7)

	# kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
	# highPassFilter(image, kernel)

	kernel = np.ones((5,5))
	# filterMinAndMax(image, kernel)
	filterCloseAndOpen(image, kernel)
main()

# key end cv2 running
cv2.waitKey(0)
cv2.destroyAllWindows()


#image rotation
def rotateImage(image, angle, scale):
	width, height = image.shape[:2]
	dimension = (width, height)
	center = (width/2, height/2)
	rotate_matrix = cv2.getRotationMatrix2D(center = center, angle = angle, scale = scale) 
	rotated_image = cv2.warpAffine(src = image, M = rotate_matrix, dsize = dimension)
	return rotated_image

# blured
def blur(image):
	width,height = image.shape[:2]
	kernel_size = (3,3)
	kernel = np.ones(kernel_size, np.float32)/(kernel_size[0] * kernel_size[1])
	gaussian_kernel = cv2.getGaussianKernel(ksize = 9, sigma = 5)

	gaussian_pyramid = cv2.pyrDown(src = image)
	blur = cv2.blur(src = image, ksize = kernel_size)
	laplacian_filter = cv2.Laplacian(src = image, ddepth = -1)
	filter2D = cv2.filter2D(src = image, ddepth = -1, kernel = kernel)
	box_filter = cv2.boxFilter(src = image, ddepth = -1, ksize = kernel_size, normalize=False)	
	dst_bilater = cv2.bilateralFilter(src = image, d = 5,  sigmaColor = 200, sigmaSpace = 200)

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
