import math

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# Convolution function
def convolution(img, kernel, average=False, verbose=False):
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

    for row in range(radius, height - radius - size - 1):
        for col in range(radius, width - radius - size - 1):
            temp = []
            for i in range(size):
                for j in range(size):
                    total = total + img[row + i - 1][col + j - 1][0]

                mean = (int)(total / div)
                filtered[row, col] = mean
                total = 0

            # print(temp)
            # temp = np.sort(temp)
            # length = len(temp)
            # val = int(temp[length // 2])
            # print(val)
            # filtered[row, col] = int(val)

    return filtered


# Question 2
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
                    temp.append(img[row + i - 1][col + j - 1][0])

            # print(temp)
            temp = np.sort(temp)
            length = len(temp)
            val = int(temp[length // 2])
            filtered[row, col] = int(val)

    return filtered


# Question 3A
def gaussian_kernel(size, sigma):
    kernel = np.zeros((size, size), np.float32)

    for i in range(size):
        for j in range(size):
            norm = math.pow(i - 1, 2) + pow(j - 1, 2)
            kernel[i, j] = math.exp(-norm / (2 * math.pow(sigma, 2)))  # Gaussian convolution
            summation = np.sum(kernel)  # summation
            final_kernel = kernel / summation  # normalization
    return final_kernel


def gaussian_kernel_x(size, sigma):
    kernel = np.zeros((size, size), np.float32)

    for i in range(size):
        for j in range(size):
            norm = math.pow(i - 1, 2)
            kernel[i, j] = math.exp(norm / (2 * math.pow(sigma, 2)))  # Gaussian convolution
            summation = np.sum(kernel)  # summation
            final_kernel = kernel / summation  # normalization
    return final_kernel


def gaussian_kernel_y(size, sigma):
    kernel = np.zeros((size, size), np.float32)

    for i in range(size):
        for j in range(size):
            norm = math.pow(j - 1, 2)
            kernel[i, j] = math.exp(norm / (2 * math.pow(sigma, 2)))  # Gaussian convolution
            summation = np.sum(kernel)  # summation
            final_kernel = kernel / summation  # normalization
    return final_kernel


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
            filtered[row, col] = summation[0]

    return filtered

def smooth_sobel(img, size, sigma):
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
                    summation = int(summation)
            filtered[row, col] = summation[0]

    return filtered


# Question 4
def gradient_operations(img, kernelx, kernely):
    x = gradient_x(img, kernelx)
    y = gradient_y(img, kernely)

    cv.imshow('Gradient x', x)
    cv.imshow('Gradient y', y)


# Question 4x
def gradient_x(img, kernel, verbose=False):
    img_x = convolution(img, kernel, verbose)
    return img_x

# Question 4y
def gradient_y(img, kernel, verbose=False):
    img_y = convolution(img, kernel, verbose)
    return img_y


# Question 5
def sobel_filter(img, kernelx, kernely, verbose=False):
    img_x = convolution(img, kernelx, verbose)
    img_y = convolution(img, kernely, verbose)


    # sobel function
    x = np.square(img_x)
    y = np.square(img_y)
    sobel = np.sqrt(x + y)
    sobel *= 255 / sobel.max()

    return sobel



def fastgaussian(img, size, sigma):
    [width, height, temp] = np.shape(img)
    kernel_x = gaussian_kernel_x(size, sigma)
    kernel_y = gaussian_kernel_y(size, sigma)
    gauss_x = fast_x(img, kernel_x)
    gauss_y = fast_y(img, kernel_y)

    filtered = np.zeros(shape=(width, height))
    for row in range(width):
        for col in range(height):
            x = gauss_x[row, col]
            y = gauss_x[row, col]
            result = np.sqrt((x ** 2) + (y ** 2))
            filtered[row, col] = result
    #
    cv.imshow('final', filtered)
    return filtered


def fast_x(img, kernel):
    img_height = img.shape[0]
    img_width = img.shape[1]

    filtered = np.zeros((img_height, img_width), np.uint8)

    for row in range(img_height - 1):
        for col in range(img_width - 1):
            summation = 0
            for x in range(-1, 2):
                for y in range(-1, 2):
                    summation += img[row, col + y] * kernel[1][y + 1]
                    # print(summation)
            filtered[row, col] = summation[0]

    cv.imshow('x', filtered)
    return filtered


def fast_y(img, kernel):
    img_height = img.shape[0]
    img_width = img.shape[1]

    filtered = np.zeros((img_height, img_width), np.uint8)

    for row in range(img_height - 1):
        for col in range(img_width - 1):
            summation = 0
            for x in range(-1, 2):
                for y in range(-1, 2):
                    summation += img[row + x, col] * kernel[x + 1][1]
                    # print(summation)
            filtered[row, col] = summation[0]
            # print(int(summation[0]))
        filtered[row] = int(summation[0])

    cv.imshow('y', filtered)
    return filtered


# i think i figured this out
# Question 7
def histogram(img, bins):
    [width, height, trash] = np.shape(img)
    div = int(256 / bins)
    arr = np.zeros(int(bins))

    for row in range(width):
        for col in range(height):
            i = int(img[row][col][0] / div)
            # print(i)
            arr[i] += 1

    plt.figure()
    plt.bar(np.arange(bins), arr)
    plt.show()

    return arr


def cannyedge(img, size, sigma, gradient_kernel_x, gradient_kernel_y):
    # Gaussian Filtering
    gaus = gaussian_filter(img, size, sigma)
    # gradient images
    gradient_img_x = gradient_x(gaus, gradient_kernel_x)
    gradient_img_y = gradient_y(gaus, gradient_kernel_x)
    sobel = sobel_filter(gaus, gradient_kernel_x, gradient_kernel_y)


# I think this is right
def segmentation(img, size, thresh):
    [width, height, temp] = np.shape(img)
    div = int(256 / size)
    hist = np.zeros(int(size))

    for row in range(width):
        for col in range(height):
            i = int(img[row][col][0] / div)
            # print(i)
            hist[i] += 1

    ret, output = cv.threshold(img, thresh, 255, cv.THRESH_BINARY)
    ret = int(ret)
    print(ret)

    for row in range(width):
        for col in range(height):

            if img[row][col][0] >= ret:
                img[row][col] = 255
            else:
                img[row][col] = 0

    s = "Otsu Threshold\nHistogram (Threshold=" + str(thresh) + ')'

    plt.figure()
    plt.title(s)
    plt.plot(np.arange(size), hist * 4)
    plt.show()

    cv.imshow('threshold', img)


def question1():
    img1 = cv.imread('images/image1.png')
    img2 = cv.imread('images/image2.png')
    b1 = box_filter(img1, 3)
    b2 = box_filter(img1, 5)
    b3 = box_filter(img2, 3)
    b4 = box_filter(img2, 5)

    cv.imshow('Image1 3x3 Kernel Size', b1)
    cv.imshow('Image1 5x5 Kernel Size', b2)
    cv.imshow('Image2 3x3 Kernel Size', b3)
    cv.imshow('Image3 5x5 Kernel Size', b4)

def question2():
    img1 = cv.imread('images/image1.png')
    img2 = cv.imread('images/image2.png')
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

def question3():
    img1 = cv.imread('images/image1.png')
    img2 = cv.imread('images/image2.png')

    g1 = gaussian_filter(img1, 3, 3)
    g2 = gaussian_filter(img1, 3, 5)
    g3 = gaussian_filter(img1, 3, 10)
    g4 = gaussian_filter(img2, 3, 3)
    g5 = gaussian_filter(img2, 3, 5)
    g6 = gaussian_filter(img2, 3, 10)

    cv.imshow('Image1 sigma = 3' , g1)
    cv.imshow('Image1 sigma = 5' , g2)
    cv.imshow('Image1 sigma = 10', g3)
    cv.imshow('Image2 sigma = 3' , g4)
    cv.imshow('Image2 sigma = 5' , g4)
    cv.imshow('Image2 sigma = 10', g5)

def question4():
    kernel_x = np.array([
        [-1.0, 0.0, 1.0],
        [-2.0, 0.0, 2.0],
        [-1.0, 0.0, 1.0]])

    kernel_y = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0]])

    img = cv.imread('images/image3.png')
    fx = gradient_x(img, kernel_x)
    fy = gradient_y(img, kernel_y)

    cv.imshow('Gradient_x', fx)
    cv.imshow('Gradient_y', fy)

def question5():
    kernel_x = np.array([
        [-1.0, 0.0, 1.0],
        [-2.0, 0.0, 2.0],
        [-1.0, 0.0, 1.0]])

    kernel_y = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0]])

    img1 = cv.imread('images/image1.png')
    img2 = cv.imread('images/image2.png')

    s1 = sobel_filter(img1, kernel_x, kernel_y)
    s2 = sobel_filter(img2, kernel_x, kernel_y)

    cv.imshow('Image1 Sobel Filter', s1)
    cv.imshow('Image2 Sobel Filter', s2)


def question6():
    img1 = cv.imread('images/image1.png')
    img2 = cv.imread('images/image2.png')

    fg1 = fastgaussian(img1, 3, 1)
    fg2 = fastgaussian(img2, 3, 1)

    cv.imshow('Fast Gaussian Image 1', fg1)
    cv.imshow('Fast Gaussian Image 2', fg2)


def question7():
    img = cv.imread('images/image4.png')

    histogram(img, 256)
    histogram(img, 128)
    histogram(img, 64 )


def question8():
    return ''


def question9():
    img2 = cv.imread('images/image2.png')
    img3 = cv.imread('images/image3.png')
    img4 = cv.imread('images/image4.png')

    segmentation(img2, 256, 156)
    segmentation(img3, 256, 156)
    segmentation(img4, 256, 156)



if __name__ == '__main__':


    # Load all of the images

    image1 = cv.imread('images/image1.png')
    image2 = cv.imread('images/image2.png')
    image3 = cv.imread('images/image3.png')
    image4 = cv.imread('images/image4.png')
    canny1 = cv.imread('images/canny1.jpg')
    canny2 = cv.imread('images/canny2.jpg')

    # question1()
    # question2()
    # question3()
    # question4()
    # question5()
    # question6()
    # question7()
    # question9()

    cv.waitKey(0)
    cv.destroyAllWindows()

