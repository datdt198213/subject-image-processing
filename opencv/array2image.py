# Python program to convert
# numpy array to image

# import required libraries
import numpy as np
from PIL import Image as im


def main():
    array = np.array([[55, 55, 61, 255, 70, 61, 64, 73],
                      [63, 59, 55, 255, 109, 85, 69, 72],
                      [62, 59, 68, 255, 114, 104, 66, 73],
                      [63, 58, 71, 255, 154, 106, 70, 69],
                      [67, 61, 68, 255, 126, 88, 68, 70],
                      [79, 65, 60, 255, 77, 68, 58, 75],
                      [85, 71, 64, 255, 55, 61, 65, 83],
                      [87, 79, 69, 255, 65, 76, 78, 94]])

    print(type(array))

    print(array.shape)
    array = np.reshape(array, (8, 8))

    print(array.shape)

    print(array)

    data = im.fromarray(array)

    data.save('x7_image.png')


if __name__ == "__main__":
    main()
