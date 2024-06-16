import cv2 
import numpy as np
import matplotlib.pyplot as plt

def outline(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges_image = cv2.Canny(blurred_image, 75, 200)

    return edges_image

def isolate(image):
    height = image.shape[0]
    polygons = np.array([
        [(500, 280), (900, height), (100, height)]
    ])
    
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def display_lines(image, lines):
    lane_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(lane_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

    return lane_image

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]

        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    
    return np.array([left_line, right_line])

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

image = cv2.imread('image_copy.png')
work_image = np.copy(image)

grayscale_image = outline(work_image)
cropped_image = isolate(grayscale_image)

lanes = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
averaged_lanes = average_slope_intercept(work_image, lanes)
lanes_image = display_lines(work_image, averaged_lanes) 

combined_image = cv2.addWeighted(image, 0.8, lanes_image, 1, 1)

cv2.imshow("Lane image", combined_image)
cv2.waitKey(0)

