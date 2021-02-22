import math

import cv2 as cv
import numpy as np


def box_filter(img, size):
    height = img.shape[0]
    width = img.shape[1]
    total = 0
    num = 1
    filtered = np.zeros((height, width))
    radius = int(np.floor(size // 2))
    temp = []
    div = size ** 2

    for row in range(radius, height - radius - size - 1):
        for col in range(radius, width - radius - size - 1):
            temp = []
            for i in range(size):
                for j in range(size):
                    total = total + img[row + i - 1][col + j - 1][0]



                mean = (int)(total / div)
                print(mean)
                filtered[row, col] = mean
                total = 0

            # print(temp)
            # temp = np.sort(temp)
            # length = len(temp)
            # val = int(temp[length // 2])
            # print(val)
            # filtered[row, col] = int(val)

    return filtered


def median_filter(img, size):
    height = img.shape[0]
    width = img.shape[1]

    filtered = np.zeros((height, width))
    radius = int(np.floor(size // 2))
    temp = []

    for row in range(radius, height - radius - size - 1):
        for col in range(radius, width - radius - size - 1):
            temp = []
            for i in range(size):
                for j in range(size):
                    print(img[row + i - 1][col + j - 1][0])
                    temp.append(img[row + i - 1][col + j - 1][0])


            # print(temp)
            temp = np.sort(temp)
            length = len(temp)
            val = int(temp[length // 2])
            filtered[row, col] = int(val)


    return filtered



def gaussian_kernel(size, sigma):
    kernel = np.zeros((size, size), np.float32)

    for i in range(size):
        for j in range(size):
            norm = math.pow(i - 1, 2) + pow(j - 1, 2)
            kernel[i, j] = math.exp(-norm / (2 * math.pow(sigma, 2)))  # Gaussian convolution
            summation = np.sum(kernel)  # summation
            final_kernel = kernel / summation  # normalization
    print(final_kernel)
    return final_kernel


def gaussian_filter(img, size, sigma):
    img_height = img.shape[0]
    img_width = img.shape[1]

    filtered = np.zeros((img_height, img_width), np.uint8)

    kernel = gaussian_kernel(size, sigma)

    for row in range(img_height - 1):
        for col in range(img_width - 1):
            summation = 0
            for x in range(-1, 2):
                for y in range(-1, 2):
                    summation += img[row + x, col + y] * kernel[x + 1][y + 1]
                    # print(summation)
            filtered[row, col] = summation[0]

    return filtered


def gradient_operations(img, kernelx, kernely):
    x = gradient_x(img, kernelx)
    y = gradient_y(img, kernely)

    cv.imshow('Gradient x', x)
    cv.imshow('Gradient y', y)


def gradient_x(img, kernelx):
    [width, height] = np.shape(img)
    filtered = np.zeros(shape=(width, height))

    for row in range(width - 2):
        for col in range(height - 2):
            sobel_x = np.sum(np.multiply(kernelx, img[row: row + 3, col: col + 3]))

            filtered[row + 1, col + 1] = sobel_x

    return filtered


def gradient_y(img, kernely):
    [width, height] = np.shape(img)
    filtered = np.zeros(shape=(width, height))

    for row in range(width - 2):
        for col in range(height - 2):
            sobel_y = np.sum(np.multiply(kernely, img[row: row + 3, col: col + 3]))
            filtered[row + 1, col + 1] = sobel_y

    return filtered


def sobel_filter(img, kernelx, kernely):
    [width, height, temp] = np.shape(img)
    filtered = np.zeros(shape=(width, height))

    for row in range(width - 2):
        for col in range(height - 2):
            sobel_x = np.sum(np.multiply(kernelx, img[row: row + 3, col: col + 3]))
            sobel_y = np.sum(np.multiply(kernely, img[row: row + 3, col: col + 3]))
            result = np.sqrt((sobel_x ** 2) + (sobel_y ** 2))
            filtered[row + 1, col + 1] = result

    return filtered


kernel_x = np.array([
    [1.0, 0.0, -1.0],
    [2.0, 0.0, -2.0],
    [1.0, 0.0, -1.0]])

kernel_y = np.array([
    [1.0, 2.0, 1.0],
    [0.0, 0.0, 0.0],
    [-1.0, -2.0, -1.0]])

image = cv.imread('images/image1.png')
cv.imshow('image', image)

# image = gaussian_filter(image, 3, 1)
# image = sobel_filter(image, kernel_x, kernel_y)
# cv.imshow('sobel', image)
# gradient_operations(image, kernel_x, kernel_y)
# image = cv.imread('images/image1.png')
# image = median_filter(image, 7)
# cv.imshow('7x7 median', image)
image = box_filter(image, 3)
# image = median_filter(image, 5)
cv.imshow('3x3 box', image)
# image = median_filter(image, 7)
# cv.imshow('7x7 median', image)
cv.waitKey(0)
cv.destroyAllWindows()
