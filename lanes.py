import cv2 
import numpy as np
import matplotlib.pyplot as plt

def outline(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges_image = cv2.Canny(blurred_image, 50, 150)

    return edges_image

def isolate(image):
    height = image.shape[0]
    polygons = np.array([
        [(0, 500), (550, 250), (1100, height), (0, height)]
    ])
    
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


image = cv2.imread('image.png')
lane_image = np.copy(image)

grayscale_image = outline(lane_image)
cropped_image = isolate(grayscale_image)
lanes = cv2.HoughLinesP(cropped_image, )

cv2.imshow("Lane image", cropped_image)
cv2.waitKey(0)

