import math

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


# Apply filter 1 dimension from notes
def apply_filter_1D(img, kernel):
    img = img.astype(np.float64)
    [height, width] = img.shape

    filtered_x = np.zeros((height, width))
    filtered_y = np.zeros((height, width))

    radius = int(np.floor(kernel.shape[0] / 2.))

    kernel = kernel[::-1]

    for row in range(radius, height - radius):
        for col in range(radius, width - radius):
            patch_x = img[row, col - radius:col + radius + 1]
            patch_y = img[row - radius:row + radius + 1, col]

            conv_x = np.sum(patch_x * kernel)
            conv_y = np.sum(patch_y * kernel)

            filtered_x[row, col] = conv_x
            filtered_y[row, col] = conv_y

    return filtered_x, filtered_y


# Convolution equation (taken from openCV)
def convolution(img, kernel, average=False):
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    img_row, img_col = img.shape
    kernel_row, kernel_col = kernel.shape

    output = np.zeros(img.shape)

    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)

    padded_image = np.zeros((img_row + (2 * pad_height), img_col + (2 * pad_width)))

    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = img

    for row in range(img_row):
        for col in range(img_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1]

    return output


# Question 1
def box_filter(img, size):
    height = img.shape[0]
    width = img.shape[1]
    total = 0
    num = 1
    filtered = np.zeros((height, width))
    radius = int(np.floor(size // 2))
    temp = []
    div = size ** 2

    # cycle through every pixel and every pixel in kernel
    for row in range(radius, height - radius - size - 1):
        for col in range(radius, width - radius - size - 1):
            temp = []
            for i in range(size):
                for j in range(size):
                    # add every pixel value together
                    total = total + img[row + i - 1][col + j - 1]

                #  Find mean of the kernel
                mean = (int)(total / div)
                # assign mean to pixel
                filtered[row, col] = mean
                # reset
                total = 0

    return filtered


# Question 2
def median_filter(img, size):
    height = img.shape[0]
    width = img.shape[1]

    filtered = np.zeros((height, width))
    radius = int(np.floor(size // 2))
    temp = []

    # cycle through every pixel and every pixel in kernel
    for row in range(radius, height - radius - size - 1):
        for col in range(radius, width - radius - size - 1):
            temp = []
            for i in range(size):
                for j in range(size):
                    # add each pixel to a list
                    temp.append(img[row + i - 1][col + j - 1])

            # sort list
            temp = np.sort(temp)
            length = len(temp)
            # find median of list
            val = int(temp[length // 2])

            # assign median to pixel
            filtered[row, col] = int(val)

    return filtered


# Question 3a
def gaussian_kernel(size, sigma):
    # Gaussian function
    size = int(size) // 2
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    normal = 1 / (2.0 * np.pi * sigma ** 2)
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
    return g


# Question 3
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
            filtered[row, col] = summation

    return filtered


# Question 4x
def gradient_operations(img):
    kernel = np.array([-1, 0, 1])
    # obtain gradients
    fx, fy = apply_filter_1D(img, kernel)
    # magnitude function
    mag = np.hypot(fx, fy)
    mag = (mag - np.min(mag)) / (np.max(mag) - np.min(mag))

    # scale magnitude
    mag *= 255
    mag_display = mag.astype(np.uint8)

    # find pure values of each image gradients
    fx = np.abs(fx)
    fy = np.abs(fy)
    fx = ((fx - np.min(fx)) / (np.max(fx) - np.min(fx) * 255).astype(np.uint8))
    fy = ((fy - np.min(fy)) / (np.max(fy) - np.min(fy) * 255).astype(np.uint8))

    gradient_orientation = np.degrees(np.arctan2(fx, fy))

    return fx, fy, gradient_orientation


# Question 5 (inspired from TA notes)
def sobel_filter(img, kernel):

    # get image gradients
    fx, fy = apply_filter_1D(img, kernel)

    # find magnitude
    mag = np.hypot(fx, fy)
    # scale
    mag = (mag - np.min(mag)) / (np.max(mag) - np.min(mag))

    mag *= 255
    mag_display = mag.astype(np.uint8)

    return mag_display


def fastgaussian(img, size, sigma):
    # Get gaussian kernel
    kernel = gaussian_kernel(size, sigma)

    # take first index of kernel
    kernel = kernel[0]

    # apply 1D Gaussian Filter to image
    img1, img2 = apply_filter_1D(img, kernel)

    # find magnitude
    combined = np.hypot(img1, img2)

    return combined


# Question 7
def histogram(img, bins, s):
    # Get shape of image
    [width, height] = np.shape(img)

    # get scale
    div = int(256 / bins)
    # Make array of size 'bins'
    arr = np.zeros(int(bins))

    # cycle through each pixel
    for row in range(width):
        for col in range(height):
            # update index of corresponding pixel
            i = int(img[row][col] / div)
            arr[i] += 1

    # Show the histogram (plot or bar)
    plt.figure()
    plt.title(s)
    plt.plot(np.arange(bins), arr)
    plt.show()

    return arr


def cannyedge(img):
    # get shape of image
    [width, height] = np.shape(img)
    # Sobel Kernel
    kernel = np.array([-1, 0, 1], np.float64)
    # Gaussian Filtering
    gaus = gaussian_filter(img, 3, 1)
    cv.imshow('Gaussian', gaus)
    # gradient imagesf
    f_x, f_y, orientation = gradient_operations(gaus)

    cv.imshow('fx', f_x)
    cv.imshow('fy', f_y)

    # Sobel FIlter
    sobel = sobel_filter(gaus, kernel)
    cv.imshow('Sobel Filter', sobel)
    # sup = nonmax_suppresion(img, orientation)

    # Run double thresh/ hysteresis
    double_threshold(sobel)


# Figure this out
def nonmax_suppresion(img, input):
    width = img.shape[0]
    height = img.shape[1]

    print(input)

    suppressed = np.zeros((width, height), dtype=np.int32)
    pi = np.pi
    angle = input * 180. / pi

    angle[angle < 0] += 180

    for row in range(1, width - 1):
        for col in range(1, height - 1):
            print(angle[row, col])
            try:
                q = 255
                r = 255
                if (0 <= angle[row, col] < 22.5) or (157.5 <= angle[row, col] <= 180):
                    q = img[row, col + 1]
                    r = img[row, col - 1]
                # angle 45
                elif 22.5 <= angle[row, col] < 67.5:
                    q = img[row + 1, col - 1]
                    r = img[row - 1, col + 1]
                # angle 90
                elif 67.5 <= angle[row, col] < 112.5:
                    q = img[row + 1, col]
                    r = img[row - 1, col]
                # angle 135
                elif 112.5 <= angle[row, col] < 157.5:
                    q = img[row - 1, col - 1]
                    r = img[row + 1, col + 1]

                if (img[row, col] >= q) and (img[row, col] >= r):
                    suppressed[row, col] = img[row, col]
                else:
                    suppressed[row, col] = 0
            except IndexError as e:
                pass

    return suppressed


def double_threshold(img):
    thresh = 20
    width, height = img.shape

    # Run threshold function once
    for row in range(width):
        for col in range(height):
            # Strong
            if img[row][col] > 20:
                img[row][col] = 255
            # Non-relevent
            elif 0 < img[row][col] < 20:
                img[row][col] = 20
            # weak
            else:
                img[row][col] = 0

    # Run threshold function twice
    for row in range(width):
        for col in range(height):

            # Strong
            if img[row][col] > 20:
                img[row][col] = 255
            # non-relevent
            elif 0 < img[row][col] < 20:
                img[row][col] = 20
            # weak
            else:
                img[row][col] = 0


    cv.imshow('Double threshold', img)

    # Call hysteresis on the returned image for last canny step
    hysteresis(img, 20, 255)

    # return (final_thresh)


# I think this is right
def segmentation(img, size, thresh):
    [width, height] = np.shape(img)
    div = int(256 / size)
    hist = np.zeros(int(size))

    # Cycle through every pixel
    for row in range(width):
        for col in range(height):
            # Get value / bin
            i = int(img[row][col] / div)
            # update that particular index in the histogram
            hist[i] += 1

    # Cycle through each pixel again
    for row in range(width):
        for col in range(height):

            # If it is higher than threshold
            if img[row][col] >= thresh:
                # Make it max
                img[row][col] = 255
            # Else, make it min
            else:
                img[row][col] = 0

    s = "Otsu Threshold\nHistogram (Threshold=" + str(thresh) + ')'

    plt.figure()
    plt.title(s)
    plt.plot(np.arange(size), hist * 4)
    plt.show()

    cv.imshow('threshold', img)


# Threshold to take care of the 'non-relevant' pixels
def hysteresis(img, weak, strong):
    width = img.shape[0]
    height = img.shape[1]

    # Cycle through every pixel in image
    for row in range(1, width - 1):
        for col in range(1, height - 1):
            if img[row, col] == weak:
                try:
                    # if any of the pixels around the current weak pixel is strong, then the weak pixel becomes strong
                    if ((img[row + 1, col - 1] == strong) or (img[row + 1, col] == strong) or (
                            img[row + 1, col + 1] == strong)
                            or (img[row, col - 1] == strong) or (img[row, col + 1] == strong)
                            or (img[row - 1, col - 1] == strong) or (img[row - 1, col] == strong) or (
                                    img[row - 1, col + 1] == strong)):
                        img[row, col] = strong

                    # If no neighbors are strong, make the pixel weak
                    else:
                        img[row, col] = 0
                except IndexError as e:
                    pass
    cv.imshow('hysteresis', img)

# Runs Question 1
def question1():
    img1 = cv.imread('images/image1.png', 0)
    img2 = cv.imread('images/image2.png', 0)
    b1 = box_filter(img1, 3)
    b2 = box_filter(img1, 5)
    b3 = box_filter(img2, 3)
    b4 = box_filter(img2, 5)

    cv.imshow('Image1 3x3 Kernel Size', b1)
    cv.imshow('Image1 5x5 Kernel Size', b2)
    cv.imshow('Image2 3x3 Kernel Size', b3)
    cv.imshow('Image3 5x5 Kernel Size', b4)

# Runs Question 2
def question2():
    img1 = cv.imread('images/image1.png', 0)
    img2 = cv.imread('images/image2.png', 0)
    m1 = median_filter(img1, 3)
    m2 = median_filter(img1, 5)
    m3 = median_filter(img1, 7)
    m4 = median_filter(img2, 3)
    m5 = median_filter(img2, 5)
    m6 = median_filter(img2, 7)

    cv.imshow('Image1 3x3 Kernel Size', m1)
    cv.imshow('Image1 5x5 Kernel Size', m2)
    cv.imshow('Image1 7x7 Kernel Size', m3)
    cv.imshow('Image2 3x3 Kernel Size', m4)
    cv.imshow('Image2 5x5 Kernel Size', m5)
    cv.imshow('Image2 7x7 Kernel Size', m6)

# Runs Question 3
def question3():
    img1 = cv.imread('images/image1.png', 0)
    img2 = cv.imread('images/image2.png', 0)

    g1 = gaussian_filter(img1, 3, 3)
    g2 = gaussian_filter(img1, 3, 5)
    g3 = gaussian_filter(img1, 3, 10)
    g4 = gaussian_filter(img2, 3, 3)
    g5 = gaussian_filter(img2, 3, 5)
    g6 = gaussian_filter(img2, 3, 10)

    cv.imshow('Image1 sigma = 3', g1)
    cv.imshow('Image1 sigma = 5', g2)
    cv.imshow('Image1 sigma = 10', g3)
    cv.imshow('Image2 sigma = 3', g4)
    cv.imshow('Image2 sigma = 5', g4)
    cv.imshow('Image2 sigma = 10', g5)

# Runs Question 4
def question4():
    kernel = np.array([-1, 0, 1], np.float64)
    img = cv.imread('images/image3.png', 0)
    f_x, f_y, orientation = gradient_operations(img)

    cv.imshow('fx', f_x)
    cv.imshow('fy', f_y)

# Runs Question 5
def question5():
    kern = np.array([-1, 0, 1], np.float64)

    img1 = cv.imread('images/image1.png', 0)
    img2 = cv.imread('images/image2.png', 0)

    s1 = sobel_filter(img1, kern)
    s2 = sobel_filter(img2, kern)

    cv.imshow('Image1 Sobel Filter', s1)
    cv.imshow('Image2 Sobel Filter', s2)

# Runs Question 6
def question6():
    img1 = cv.imread('images/image1.png', 0)
    img2 = cv.imread('images/image2.png', 0)

    fg1 = fastgaussian(img1, 3, 1)
    fg2 = fastgaussian(img2, 3, 1)

    cv.imshow('Fast Gaussian Image 1', fg1)
    cv.imshow('Fast Gaussian Image 2', fg2)

# Runs Question 7
def question7():
    img = cv.imread('images/image4.png', 0)

    histogram(img, 256, 'Question 7a')
    histogram(img, 128, 'Question 7b')
    histogram(img, 64, 'Question 7c')

# Runs Question 8
def question8():
    kernel_x = np.array([
        [-1.0, 0.0, 1.0],
        [-2.0, 0.0, 2.0],
        [-1.0, 0.0, 1.0]])

    kernel_y = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0]])

    img1 = cv.imread('images/canny1.jpg', 0)
    img2 = cv.imread('images/canny2.jpg', 0)

    cannyedge(img1)

# Runs Question 9
def question9():
    img1 = cv.imread('images/canny1.jpg', 0)
    img2 = cv.imread('images/canny2.jpg', 0)
    img3 = cv.imread('images/image4.png', 0)

    segmentation(img1, 256, 156)
    # segmentation(img2, 256, 156)
    # segmentation(img3, 256, 156)


if __name__ == '__main__':
    # question1()
    # question2()
    # question3()
    # question4()
    # question5()
    # question6()
    # question7()
    # question8()
    # question9()
    cv.waitKey(0)
    cv.destroyAllWindows()
