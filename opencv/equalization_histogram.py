"""
To equalize image histogram
"""

import cv2
import matplotlib.pyplot as plt


def color_histogram(image):
    colours = ('b', 'g', 'r')
    for i, color in enumerate(colours):
        histogram = cv2.calcHist(image, [i], None, [256], [0, 256])
        plt.plot(histogram, color)
        plt.xlim([0, 256])
    plt.show()


def main():
    source = cv2.imread("../source/car.jpg")
    source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    destination = cv2.equalizeHist(source)

    # cv2.imshow("Original image", source)
    # cv2.imshow("Equalize histogram", destination)

    color_histogram(source)
    color_histogram(destination)


if __name__ == "__main__":
    main()

cv2.waitKey(0)
cv2.destroyAllWindows()
