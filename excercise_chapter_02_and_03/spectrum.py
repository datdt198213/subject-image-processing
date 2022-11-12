import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import cv2 as cv

image = cv.imread('image.jpg')

gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

array = []

for i in range(0,gray_image.shape[0]):
	for j in range(0,gray_image.shape[1]):
		pixel = gray_image.item(i, j)
		array.append(pixel)

#key: value, key: x, value: y, access into counter.key and 
#access value counter.value
labels, values = zip(*Counter(array).items())

# Width of column in plt
width = 1

plt.bar(labels, values, width)
plt.xlabel('Pixel values')
plt.ylabel('Quantity')
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
	plt.title(title, fontsize=10)

def drawSpectrums(images = [], subplots = [], titles = []):
	for i in range(len(images)):
		drawSpectrum(images[i], subplots[i], titles[i])
	plt.show()