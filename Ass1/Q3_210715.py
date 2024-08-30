import cv2
import numpy as np

def solution(image_path):
    ############################
    ############################

    ############################
    ############################
    ## comment the line below before submitting else your code wont be executed##
    # pass
    image = cv2.imread(image_path)

    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.bitwise_not(gray_img)
    gray_threshold_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coordinates = np.column_stack(np.where(gray_threshold_img > 0))
    angle = cv2.minAreaRect(coordinates)[-1]
    if angle < -45:
        angle = -(90 + angle)
    elif angle >= 45:
        angle = 90 - angle
    else:
        angle = -angle
    if(angle<0):
        angle+=180

    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    new_width = int(width * np.abs(np.cos(np.radians(angle))) + height * np.abs(np.sin(np.radians(angle))))
    new_height = int(height * np.abs(np.cos(np.radians(angle))) + width * np.abs(np.sin(np.radians(angle))))
    rotation_matrix[0, 2] += (new_width - width) / 2
    rotation_matrix[1, 2] += (new_height - height) / 2
    rotated_img = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)



    return rotated_img
