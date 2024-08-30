import cv2
import numpy as np

# Usage
def solution(image_path):
    image= cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply Gaussian blur to the mask
    # image = cv2.GaussianBlur(image, (5, 5), 0)
    kernel_size = (5, 5)
    sigma = 0
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)

    cv2.imshow('GaussianBlur', image)
    cv2.waitKey(0)

    # Define the lower and upper bounds of the lava color in RGB
    # lower_lava = np.array([150, 0, 0], dtype=np.uint8)
    # upper_lava = np.array([255, 200, 100], dtype=np.uint8)
    lava_color_bounds = ((150, 0, 0), (255, 200, 100))
    lower_lava = np.array(lava_color_bounds[0], dtype=np.uint8)
    upper_lava = np.array(lava_color_bounds[1], dtype=np.uint8)

    # Create a mask based on the color bounds
    # mask = cv2.inRange(image, lower_lava, upper_lava)
    mask = np.all(np.logical_and(lower_lava <= image, image <= upper_lava), axis=-1).astype(np.uint8) * 255

    cv2.imshow('mask', mask)
    cv2.waitKey(0)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty black image (all zeros) to draw the contours
    contour_image = np.zeros_like(image)
    cv2.imshow('contour_image', contour_image)
    cv2.waitKey(0)

    # # Draw the contours on the black image
    # cv2.drawContours(contour_image, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    # # Define the kernel for erosion (adjust size as needed)
    # kernel = np.ones((1, 1), np.uint8)

    # # Apply the opening operation
    # result_opening = cv2.morphologyEx(contour_image, cv2.MORPH_OPEN, kernel)

    # Create a black image to draw contours
    contour_image = np.zeros_like(contour_image)

    # Draw the contours on the black image
    cv2.fillPoly(contour_image, contours, color=(255, 255, 255))

    # Define the kernel for erosion (adjust size as needed)
    kernel_size = (1, 1)
    kernel = np.ones(kernel_size, np.uint8)

    # Apply the opening operation
    result_opening = cv2.morphologyEx(contour_image, cv2.MORPH_OPEN, kernel)

    # Apply the erosion operation
    smoothed_image = cv2.erode(result_opening, kernel, iterations=1)


    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    kernel_size = (5, 5)
    sigma = 0.2
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)

    lava_color_bounds = ((180, 0, 0), (255, 200, 100))
    lower_lava = np.array(lava_color_bounds[0], dtype=np.uint8)
    upper_lava = np.array(lava_color_bounds[1], dtype=np.uint8)

    mask = np.all(np.logical_and(lower_lava <= image, image <= upper_lava), axis=-1).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_image = np.zeros_like(image)
    contour_image = np.zeros_like(contour_image)

    if contours:
        outer_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(contour_image, [outer_contour], -1,(255, 255, 255), 2)
    kernel = np.ones((2, 2), np.uint8)

    contour_image=cv2.dilate(contour_image, kernel, iterations=5)
    cv2.drawContours(contour_image, [outer_contour], -1, (255, 255, 255), thickness=cv2.FILLED)


    cv2.imshow('output', smoothed_image)
    cv2.waitKey()
    cv2.destroyAllWindows()






    ######################################################################  
    return smoothed_image

image = solution('test/lava25.jpg')
