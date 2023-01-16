"""
@file sobel_demo.py
@brief Sample code using Sobel and/or Scharr OpenCV functions to make a simple Edge Detector
"""
import cv2 as cv


def main():
    scale = 1
    delta = 0
    depth = cv.CV_16S

    # Load the image
    src = cv.imread("..\source\car.jpg")

    src = cv.GaussianBlur(src, (3, 3), 0)

    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    # Gradient-X
    # grad_x = cv.Scharr(gray,depth,0,1)
    grad_x = cv.Sobel(gray, depth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)

    print(depth)

    # Gradient-Y
    # grad_y = cv.Scharr(gray,depth,0,1)
    grad_y = cv.Sobel(gray, depth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)

    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)

    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    cv.imshow("window_name", grad)
    cv.waitKey(0)

    return 0


if __name__ == "__main__":
    main()
