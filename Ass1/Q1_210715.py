import cv2
import numpy as np

# Usage
def solution(image_path):
    image= cv2.imread(image_path)
    ######################################################################
    ######################################################################
    #####  WRITE YOUR CODE BELOW THIS LINE ###############################

    def calculate_nearest_point(approximated_polygon,h,w):
        nearest_point = None
        min_distance = float('inf') 
        
        for point in approximated_polygon:
            distance = (point[0,0]- w)**2 + (point[0,1] - h)**2
            if distance < min_distance:
                min_distance = distance
                nearest_point = point
        return nearest_point


    h, w, _ = image.shape

    width, height = 600,600
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = None
    largest_contour_area = 0

    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area > largest_contour_area:
            largest_contour = contour
            largest_contour_area = contour_area
            
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approximated_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)

    top_left = calculate_nearest_point(approximated_polygon,0,0)
    top_right = calculate_nearest_point(approximated_polygon,0,w)
    down_right = calculate_nearest_point(approximated_polygon,h,w)
    down_left = calculate_nearest_point(approximated_polygon,h,0)

    top_left[0,0]=top_left[0,0]+1
    top_left[0,1]=top_left[0,1]+1
    top_right[0,0]=top_right[0,0]-1
    top_right[0,1]=top_right[0,1]+1
    down_right[0,0]=down_right[0,0]-1
    down_right[0,1]=down_right[0,1]-1
    down_left[0,0]=down_left[0,0]+1
    down_left[0,1]=down_left[0,1]-1

    source_points = np.vstack((top_left, top_right, down_right, down_left))

    destination_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    perspective_matrix = cv2.getPerspectiveTransform(source_points.astype(np.float32), destination_points)
    corrected_image = cv2.warpPerspective(image, perspective_matrix, (width, height))


    ######################################################################

    return corrected_image
