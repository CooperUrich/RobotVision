# All the imports will be here
import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA

#
# def highest_hog(image_cell_hog):
#     height, width, depth = image_cell_hog.shape
#     newMatrix = np.zeros((height, width))
#
#     for row in range(height):
#         for col in range(width):
#             x = np.max(image_cell_hog[row, col, :])
#             newMatrix[row, col] = x
#
#     return newMatrix


def apply_1D_conv(img, kernel):
    # get image shape
    img_height, img_width = img.shape

    # create empty image arrays
    filtered_img_x = np.zeros((img_height, img_width))
    filtered_img_y = np.zeros((img_height, img_width))

    # get kernel radius, if kernel.shape[0] is an even number then
    # the function will throw an error
    kernel_radius = int(np.floor(kernel.shape[0] / 2))

    # Traverse image and get patches of the image, one in the X-direction
    # the other in the Y-direction.
    # multiply each patch by the kernel and sum their values separately.
    # set its respective filtered_img pixel with the value
    for row in range(kernel_radius, img_height - kernel_radius):
        for col in range(kernel_radius, img_width - kernel_radius):
            patch_x = img[row, col - kernel_radius: col + kernel_radius + 1]
            patch_y = img[row - kernel_radius:row + kernel_radius + 1, col]
            value_x = np.sum(patch_x * kernel)
            value_y = np.sum(patch_y * kernel)
            filtered_img_x[row, col] = value_x
            filtered_img_y[row, col] = value_y
    return filtered_img_x, filtered_img_y


def hog_2D(img, block_size, cell_size, orientations=9):
    '''
    This function computes the HoG feature values for a given image
    and returns the normalized feature values and per cell HoG values.
    :param img: Input image
    :param block_size: cells per block
    :param cell_size: pixels per cell
    :param orientations: orientations per 180 degrees
    :return: normalized_blocks: normalized features for each block
             image_cell_hog: HoG magnitude values for each bin of each cell of the image. Shape: [Cell_per_row x Cell_per_column x orientations]
    '''

    # Convert image to float type
    img = img.astype(np.float)
    img_height, img_width = img.shape

    # Containers for x,y derivative
    f_x = np.zeros(img.shape)
    f_y = np.zeros(img.shape)

    kernel = np.array([-1, 0, 1])
    f_x, f_y = apply_1D_conv(img, kernel)

    # Get Magnitude
    mag = np.hypot(f_x, f_y)
    mag = (mag / np.max(mag)) * 255

    # Get orientation of gradients, convert to degrees
    phase = np.degrees(np.arctan2(f_y, f_x))

    # Convert negative angles to equivalent positive angles, so that it has same direction
    # converts the negative angles (0 to -179) to corresponding positive angle [-20 is equivalent to +160]
    phase[phase < 0] += 180
    # phase = phase % 180   # Alternative way to convert
    phase[phase == 180] = 0  # Make 180 as 0 for computation simplicity later

    # Calculate total number of cell rows and columns Notice that it uses integer number of cell row,cols If the
    # image is of irregular size, we only compute till last position with full cell. If there are some pixels left
    # which dont fill a full cell, it is ignored Alternatively, you can also reshape the image to have height,
    # width be divisible by pixels_per_cell.
    cell_rows = img_height // cell_size
    cell_cols = img_width // cell_size

    # Create container for HoG values per orientation for each cell.
    image_cell_hog = np.zeros((cell_rows, cell_cols, orientations))

    # Compute the angle each bin will have. For orientation 9, it should have 180/9 = 20 degrees per bin
    angle_per_bin = 180 / orientations

    # This is the main HoG values computation part
    # Follow algorithm from class
    # Go through each cell
    for row in range(cell_rows):
        for col in range(cell_cols):
            # Each cell has N x N pixels in it. So get the patch of pixels for each cell
            # Get the magnitude and orientation patch
            cell_patch_mag = mag[row * cell_size:(row + 1) * cell_size, col * cell_size:(col + 1) * cell_size]
            # Same way to get patch for phase
            cell_patch_orient = phase[row * cell_size:(row + 1) * cell_size, col * cell_size:(col + 1) * cell_size]

            # Now for each cell patch, go through each pixel
            # Get the orientation and magnitude
            # Find the bin based on orientation
            # Then add magnitude (weighted) to the bin(s)
            for rr in range(cell_size):
                for cc in range(cell_size):
                    # Get the current pixel's orientation and magnitude
                    current_orientation = cell_patch_orient[rr, cc]  # Get from orientation patch
                    current_magnitude = cell_patch_mag[rr, cc]  # Get from magnitude patch

                    # Find current bin based on magnitude
                    # get bin by dividing orientation by angle per bin
                    current_bin = int(current_orientation // angle_per_bin)

                    # Use voting scheme from class
                    # Find what percentage of magnitude goes to bin on left and right of orientation
                    # So if orientation is 25, then it is between 20 and 40 and the current bin is 1
                    # But orientation of 25 means the magnitude should go somewhat to the bin for 40
                    # So a weighted value is assigned to both bins
                    # Find what percentage is to previous bin => 25 - 20 = 5. Then 5/20 = 25%.
                    # This means 25% of the magnitude should go to bin 2 and 75% should go to
                    # bin 1 as 25 is closer to bin 1

                    # Find the bin by multiplying it by 20
                    new_bin = current_bin * angle_per_bin

                    right_val = current_orientation - new_bin
                    right_val = right_val / 20
                    left_val = 1 - right_val

                    bin_left_percent = left_val
                    bin_left_value = left_val * current_magnitude
                    bin_right_value = right_val * current_magnitude


                    if current_bin + 1 == orientations:  # last bin at 160, which will wrap around to 0 again
                        image_cell_hog[row, col, current_bin] += bin_left_value  # Add to current bin
                        image_cell_hog[row, col, 0] += bin_right_value  # Add to bin 0 since it goes around
                    else:
                        image_cell_hog[row, col, current_bin] += bin_left_value  # Add to current bin
                        image_cell_hog[row, col, current_bin + 1] += bin_right_value  # Add to current bin + 1

    # Now normalize values per block
    # Find number of blocks for given cells per block that fits in the image.
    block_rows = cell_rows - block_size + 1
    block_cols = cell_cols - block_size + 1

    # Create container for features per block
    normalized_blocks = np.zeros((block_rows, block_cols, block_size * block_size * orientations))

    # Iterate through each block, get HoG values of cells in that block, normalize using L2 method
    for row in range(block_rows):
        for col in range(block_cols):
            # Get current block patch with given cells_per_block from image_cell_hog
            current_block_patch = image_cell_hog[row: row + block_rows, col: col + block_cols, :]
            # Normalize using L2 method.
            # Square each value, sum all of them, take square root
            normalized_block_patch = LA.norm(current_block_patch)  # Perform L2 normalization
            # Reshape to 1D array, gives [orientation * number of cells X 1]shape for each block
            normalized_block_patch = np.reshape(normalized_block_patch, -1)  # Make 1D
            # Assign the patch output to container
            normalized_blocks[row, col, :] = normalized_block_patch

    return normalized_blocks, image_cell_hog


# Read image as grayscale
img1 = cv2.imread("images/canny1.jpg", 0)
img2 = cv2.imread("images/canny2.jpg", 0)

# Set parameters
block_size = 2  # Cells per block
cell_size = 8  # Pixels per cell
orientations = 9  # Orientations per 180 degrees

'''
Manual function to get HoG features.
Takes image, block size, cell size and orientations.
Returns the normalized blocks (HoG features per block) and the HoG values per cell of the image.
For visualization, use the HoG values per cell (image_cell_hog) of shape [cells_per_row, cells_per_col]    
'''

normalized_blocks1, image_cell_hog1 = hog_2D(img1, block_size, cell_size, orientations)
normalized_blocks2, image_cell_hog2 = hog_2D(img2, block_size, cell_size, orientations)

# For color coding, take the HoG values per cell, find maximum HoG value among all orientations.
# Then normalize the max value and resize it to match image.
hog_max1 = np.amax(image_cell_hog1, axis=2)
hog_max1 = hog_max1 / np.max(hog_max1)
hog_max1 = cv2.resize(hog_max1, (img1.shape[1], img1.shape[0]), cv2.INTER_NEAREST)

hog_max2 = np.amax(image_cell_hog2, axis=2)
hog_max2 = hog_max2 / np.max(hog_max2)
hog_max2 = cv2.resize(hog_max2, (img2.shape[1], img2.shape[0]), cv2.INTER_NEAREST)

# Here you can implement your own method to draw a line along the bin with highest value.
# For this option, you have to check which bin has highest value, then assign the highest value to the line
# Finally, you will normalize the image so that unimportant lines will have lower weight.
# CODE FOR 18 BINS, CELL SIZE IS 16x16, BLOCK SIZE IS 4x4

# Set parameters
block_size = 4  # Cells per block
cell_size = 16  # Pixels per cell
orientations = 18  # Orientations per 180 degrees

'''
Manual function to get HoG features.
Takes image, block size, cell size and orientations.
Returns the normalized blocks (HoG features per block) and the HoG values per cell of the image.
For visualization, use the HoG values per cell (image_cell_hog) of shape [cells_per_row, cells_per_col]    
'''

normalized_blocks3, image_cell_hog3 = hog_2D(img1, block_size, cell_size, orientations)
normalized_blocks4, image_cell_hog4 = hog_2D(img2, block_size, cell_size, orientations)

# For color coding, take the HoG values per cell, find maximum HoG value among all orientations.
# Then normalize the max value and resize it to match image.
hog_max3 = np.amax(image_cell_hog3, axis=2)
hog_max3 = hog_max3 / np.max(hog_max3)
hog_max3 = cv2.resize(hog_max3, (img1.shape[1], img1.shape[0]), cv2.INTER_NEAREST)

hog_max4 = np.amax(image_cell_hog4, axis=2)
hog_max4 = hog_max4 / np.max(hog_max4)
hog_max4 = cv2.resize(hog_max4, (img2.shape[1], img2.shape[0]), cv2.INTER_NEAREST)

# Here you can implement your own method to draw a line along the bin with highest value.
# For this option, you have to check which bin has highest value, then assign the highest value to the line
# Finally, you will normalize the image so that unimportant lines will have lower weight.

# This is only for comparing your output with actual HoG output.
# Do not use this for final submission. Only use this as reference

# Now draw all the images/outputs using matplotlib.
# Notice that the color is mapped to gray using cmap argument.
# For final submission, remove scikit hog output. You can keep it blank.
fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig1.suptitle('HoG')
ax1.set_title('Image')
ax1.imshow(img1, cmap='gray')
ax2.set_title("HoG color coding1 (Block = 2, Cell = 8 Orientations = 9)")
ax2.imshow(hog_max1, cmap='gray')
ax3.set_title('Image')
ax3.imshow(img2, cmap='gray')
ax4.set_title("HoG color coding2 (Block = 2, Cell = 8 Orientations = 9)")
ax4.imshow(hog_max2, cmap='gray')
plt.show()

# plt.imshow(hog_max1, cmap="gray")
# plt.show()
# plt.imshow(hog_max2, cmap="gray")
# plt.show()

fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig1.suptitle('HoG')
ax1.set_title('Image')
ax1.imshow(img1, cmap='gray')
ax2.set_title("HoG color coding1 (Block = 4, Cell = 16 Orientations = 18")
ax2.imshow(hog_max3, cmap='gray')
ax3.set_title('Image')
ax3.imshow(img2, cmap='gray')
ax4.set_title("HoG color coding2(Block = 4, Cell = 16 Orientations = 18")
ax4.imshow(hog_max4, cmap='gray')
plt.show()
#
# plt.imshow(hog_max3, cmap="gray")
# plt.show()
# plt.imshow(hog_max4, cmap="gray")
# plt.show()
