import cv2
import numpy as np
import os
def edge_detect(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Gaussian filter
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Grayscale image, Gaussian kernel size, standard deviation
    # Edge detection
    edged = cv2.Canny(blurred, 50, 150)
    return edged

# Edge detection
def find_contours(image):
    # Edge detection
    edged = edge_detect(image)
    # Contour detection
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Crop images
def crop(rotated, contours, h, w):
    # Coordinates of the rectangle
    contours = find_contours(rotated)
    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    x1, y1 = box[0]
    x2, y2 = box[2]
    if y1 < h - y2:
        rotated = cv2.rotate(rotated, cv2.ROTATE_180)
        x1, y1 = w - x1, h - y1
        x2, y2 = w - x2, h - y2

    # Crop the rectangular answer area
    answer_id = rotated[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
    # Crop the ID area
    id_area = rotated[int(0.47*min(y1, y2)): int(0.93*min(y1, y2)),int(0.76*max(x1, x2)):int(0.98*max(x1, x2))]
    return answer_id, id_area

# rotaion image
def deskew(image):
    # Get image size
    (h, w) = image.shape[:2]

    # Detect contours
    contours = find_contours(image)


    # cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

    # Find the largest contour
    c = max(contours, key=cv2.contourArea)

    # Find the largest contour
    rect = cv2.minAreaRect(c) # get the min rectangle of the lagest contour
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    # Rotate rectangle, narrow side horizontal
    angle = rect[2]  # Get the rotation angle
    if rect[1][0] > rect[1][1]:# Whether the width of the rectangle is greater than its height
        angle = 90 + angle

    # Rotate the image
    center = (w // 2, h // 2) # Determining the center of rotation
    M = cv2.getRotationMatrix2D(center, angle, 1.0)  # Get rotation matrix
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)  # Rotate the image

    # Crop the image
    answer_area, id_area = crop(rotated, contours, h, w)

    return answer_area,id_area
