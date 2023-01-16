from multipledispatch import dispatch
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from collections import Counter
import cv2


# Save Image
def save_image(image, file_name):
    print("Save successfully!")
    cv2.imwrite(file_name, image)


# Resize image
@dispatch(np.ndarray, int)
def resize_image(image, percent):
    width = int(image.shape[1] * percent / 100)
    height = int(image.shape[0] * percent / 100)
    dimension = (width, height)
    resized_image = cv2.resize(src=image, dsize=dimension, interpolation=cv2.INTER_CUBIC)
    return resized_image


@dispatch(np.ndarray, tuple)
def resize_image(image, dimension):
    resized_image = cv2.resize(src=image, dsize=dimension, interpolation=cv2.INTER_CUBIC)
    return resized_image


# convert image color
def convert_color(image, channel):
    if channel == "GRAY":
        cvt_color_image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
    elif channel == "HSV":
        cvt_color_image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2HSV)
    elif channel == "RGB":
        cvt_color_image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)
    elif channel == "YCC":
        cvt_color_image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2YCrCb)
    elif channel == "HLS":
        cvt_color_image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2HLS)
    elif channel == "LUV":
        cvt_color_image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2Luv)
    return cvt_color_image


def display_image(image, subplot, title):
    cvt_color_image = convert_color(image, "RGB")
    plt.subplot(subplot), plt.imshow(cvt_color_image), plt.title(title)
    plt.xticks([]), plt.yticks([])


def display_images(images=[], subplots=[], titles=[]):
    for i in range(len(images)):
        display_image(images[i], subplots[i], titles[i])
    plt.show()


def display_color_images_conversion(image):
    gray_image = convert_color(image, "GRAY")
    hsv_image = convert_color(image, "HSV")
    rgb_image = image
    ycc_image = convert_color(image, "YCC")
    hls_image = convert_color(image, "HLS")
    luv_image = convert_color(image, "LUV")

    images = [gray_image, hsv_image, rgb_image, ycc_image, hls_image, luv_image]
    subplots = [231, 232, 233, 234, 235, 236]
    titles = ["GRAY", "HSV", "RGB", "YCC", "HLS", "LUV"]
    display_images(images, subplots, titles)


def display_resize_images(image, percent, dimension, title1, title2):
    image_resized_p = resize_image(image, percent)
    image_resized_d = resize_image(image, dimension)

    cv2.imshow(title1, image_resized_p)
    cv2.imshow(title2, image_resized_d)


def display_negative_image(image):
    dst = cv2.bitwise_not(image)

    images = [image, dst]
    subplots = [121, 122]
    titles = ["Original Image", "Negative Image"]
    display_images(images, subplots, titles)


def display_histogram(image, subplot, title):
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()

    plt.subplot(subplot), plt.title(title)
    plt.plot(cdf_normalized, color='b')
    plt.hist(image.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')


def display_histograms(images=[], subplots=[], titles=[]):
    for i in range(len(images)):
        display_histogram(images[i], subplots[i], titles[i])
    plt.show()


def display_equalized_histogram(image):
    image = convert_color(image, "GRAY")
    dst = cv2.equalizeHist(image)

    images = [image, dst]
    subplots = [121, 122]
    titles = ["Original Image Histogram", "Equalized Image Histogram"]
    display_histograms(images, subplots, titles)


def display_histogram_matching(image):
    colours = ('b', 'g', 'r')
    for i, color in enumerate(colours):
        histr = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(histr, colours=color)
        plt.xlim([0, 256])


# plt.show()

# def displayConcatImage(image1, image2):
# 	image = cv2.hconcat(image1, image2)
# 	cv2.imshow("Concat image", image)

def enhance_linear_contrast(image, alpha, beta):
    dst_image = np.zeros(image.shape, image.dtype)

    for h in range(image.shape[0]):
        for w in range(image.shape[1]):
            for c in range(image.shape[2]):
                dst_image[h, w, c] = np.clip(alpha * image[h, w, c] + beta, 0, 255)

    images = [image, dst_image]
    subplots = [121, 122]
    titles = ["Original Image", "Linear Contrast Image"]
    display_images(images, subplots, titles)


def enhance_gamma_power_law(image, c, gamma):
    img_float32 = np.float32(image)
    dst_image = np.array(c * (img_float32 / 255) ** gamma)

    images = [image, dst_image]
    subplots = [121, 122]
    titles = ["Original Image", "Gamma Power Law Image"]
    display_images(images, subplots, titles)


def enhance_piecewise_linear(image, x1, x2, k1, k2, y1):
    img_float32 = np.float32(image)
    dst = img_float32
    cond_list = [(dst < x1), dst >= x2]
    func_list = [lambda dst: k1 * dst + y1 - k1 * x1, lambda dst: k2 * dst + y1 + k2 * x2]
    dst = np.piecewise(dst, cond_list, func_list)

    images = [image, dst]
    subplots = [121, 122]
    titles = ["Original Image", "Piecewise Linear Image"]
    display_images(images, subplots, titles)


def enhance_non_linear_contrast(image, threshold, grid_size):
    gray = convert_color(image, "GRAY")

    clahe = cv2.createCLAHE(threshold, grid_size)
    equalize = clahe.apply(gray)

    images = [gray, equalize]
    subplots = [121, 122]
    titles = ["Gray image", "Adaptive histogram equalization"]
    display_images(images, subplots, titles)


def low_pass_filter(image, ksize):
    dst_blur = cv2.blur(src=image, ksize=ksize)
    dst_gaussian = cv2.GaussianBlur(src=image, ksize=ksize, sigmaX=0, sigmaY=0)

    images = [image, dst_blur, dst_gaussian]
    subplots = [131, 132, 133]
    titles = ["Original Image", "Median filter", "Gaussian filter"]
    display_images(images, subplots, titles)


def median_filter(image, ksize):
    dst_median = cv2.medianBlur(src=image, ksize=ksize)

    images = [image, dst_median]
    subplots = [121, 122]
    titles = ["Original Image", "Median Filter Image"]
    display_images(images, subplots, titles)


def high_pass_filter(image, kernel):
    dst_sharpen = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

    images = [image, dst_sharpen]
    subplots = [121, 122]
    titles = ["Original Image", "Sharpened Filter Image"]
    display_images(images, subplots, titles)


def filter_min_max(image, kernel):
    dst_erosion = cv2.erode(src=image, kernel=kernel, iterations=1)
    dst_dilation = cv2.dilate(src=image, kernel=kernel, iterations=1)

    images = [image, dst_erosion, dst_dilation]
    subplots = [131, 132, 133]
    titles = ["Original Image", "Min Filter Image", "Max Filter Image"]
    display_images(images, subplots, titles)


def filter_close_open(image, kernel):
    dst_opening = cv2.morphologyEx(src=image, op=cv2.MORPH_OPEN, kernel=kernel)
    dst_closing = cv2.morphologyEx(src=image, op=cv2.MORPH_CLOSE, kernel=kernel)

    images = [image, dst_opening, dst_closing]
    subplots = [131, 132, 133]
    titles = ["Original Image", "Opening Filter Image", "Closing Filter Image"]
    display_images(images, subplots, titles)


def display_affine_transform(image, pts1, pts2):
    rows, cols, ch = image.shape
    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(image, M, (cols, rows))

    images = [image, dst]
    subplots = [121, 122]
    titles = ["Original Image", "Affine Transform Image"]
    display_images(images, subplots, titles)


def display_perspective_transform(image, pts1, pts2):
    rows, cols, ch = image.shape
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(image, M, (cols, rows))

    images = [image, dst]
    subplots = [121, 122]
    titles = ["Original Image", "Perspective Transform Image"]
    display_images(images, subplots, titles)


def display_adaptive_thresholding(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, th1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    th3 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    titles = ['Original Image', 'Global Thresholding (v = 127)', 'Adaptive Mean Thresholding',
              'Adaptive Gaussian Thresholding']
    images = [image, th1, th2, th3]
    subplots = [221, 222, 223, 224]
    display_images(images, subplots, titles)


def display_thresholding(image):
    ret, thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(image, 127, 255, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO_INV)

    titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
    images = [image, thresh1, thresh2, thresh3, thresh4, thresh5]
    subplots = [231, 232, 233, 234, 235, 236]

    display_images(images, subplots, titles)


def main():
    image = cv2.imread("car.jpg")

    # display_color_images_conversion(image)
    # display_resize_images(image, 50, (200,200), "Scale 50%", "200x200")
    # display_negative_image(image)
    display_equalized_histogram(image)
    display_histogram_matching(image)

    # enhance_linear_contrast(image, 2, -5)
    # enhance_non_linear_contrast(image, 40.0, (8,8))
    # enhance_gamma_power_law(image, 100, 2.0)
    # enhance_piecewise_linear(image, 20, 30, 50, 60, 70)

    # low_pass_filter(image, (5,7))
    # median_filter(image, 7)

    # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # high_pass_filter(image, kernel)

    # kernel = np.ones((5,5))
    # filter_min_max(image, kernel)
    # filter_close_open(image, kernel)

    # pts1 = np.float32([[50,50], [200,50], [50,200]])
    # pts2 = np.float32([[10,100], [200,50], [100,250]])
    # display_affine_transform(image, pts1, pts2)

    # pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
    # pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
    # display_perspective_transform(image, pts1, pts2)

    # display_thresholding(image)
    # display_adaptive_thresholding(image)

    # x7_image = cv2.imread("x7_image.png")
    # enhance_gamma_power_law(x7_image, 0.1, 1.0)
    # display_histogram(x7_image, "x7_image", "title")


main()

# key end cv2 running
cv2.waitKey(0)
cv2.destroyAllWindows()


# image rotation
def rotate_image(image, angle, scale):
    width, height = image.shape[:2]
    dimension = (width, height)
    center = (width / 2, height / 2)
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=scale)
    rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=dimension)
    return rotated_image


# blured
def blur(image):
    width, height = image.shape[:2]
    kernel_size = (3, 3)
    kernel = np.ones(kernel_size, np.float32) / (kernel_size[0] * kernel_size[1])
    gaussian_kernel = cv2.getGaussianKernel(ksize=9, sigma=5)

    gaussian_pyramid = cv2.pyrDown(src=image)
    blur = cv2.blur(src=image, ksize=kernel_size)
    laplacian_filter = cv2.Laplacian(src=image, ddepth=-1)
    filter2D = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    box_filter = cv2.boxFilter(src=image, ddepth=-1, ksize=kernel_size, normalize=False)
    dst_bilater = cv2.bilateralFilter(src=image, d=5, sigmaColor=200, sigmaSpace=200)


# image_rotation = rotateImage(image, 45, 0.5)
# cv2.imshow("image_rotation", image_rotation)

#####################################################################
# image translation
# image = cv2.imread('car.jpg')
# rows, cols, channels = image.shape
# M = np.float32([[1,0,100], [0, 1, 50]]) 
# 100 is width translation, 50 is height translation
# dst = cv2.warpAffine(image, M, (cols, rows))

# cv2.imshow('img', dst)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

#####################################################################
# Otsu's Binarization
# image = cv2.imread('index.jpg')

# def Otsu_binarization(image):
# 	img_float32 = np.float32(image)
# 	gray_image = cv2.cvtColor(img_float32, cv2.COLOR_BGR2GRAY)

# 	# global thresholding
# 	ret1,th1 = cv2.threshold(gray_image,127,255,cv2.THRESH_BINARY)

# 	# Otsu's thresholding 
# 	ret2,th2 = cv2.threshold(gray_image,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 	#Otsu's thresholding after Gaussian filtering
# 	blur = cv2.GaussianBlur(gray_image, (5,5) , 0)
# 	ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 	images = [image, 0, th1, image, 0, th2, blur, 0, th3]
# 	titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v = 127)',
# 	'Original Noisy Image', 'Histogram', "Otsu's Thresholding",
# 	'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding"]
# 	subplots = [331, 332, 333, 334, 335, 336, 337, 338, 339]
# 	display_images(images, subplots, titles)

# image = cv2.imread('car.jpg')
# Otsu_binarization(image)

def visualizeRGB(image, title):
    r, g, b = cv2.split(image)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")

    pixel_colors = image.reshape((np.shape(image)[0] * np.shape(image)[1], 3))
    norm = colors.Normalize(vmin=-1., vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Red")
    axis.set_ylabel("Green")
    axis.set_zlabel("Blue")
    plt.title(title)


def visualizeRGBs(images=[], titles=[]):
    for i in range(len(images)):
        visualizeRGB(images[i], titles[i])
    plt.show()
