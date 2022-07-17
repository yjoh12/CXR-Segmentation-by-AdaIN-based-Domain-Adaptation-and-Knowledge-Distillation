import copy
import cv2
import numpy as np


def post_processing(image):
    
    image = image.astype('uint8')

    # largist
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(contours)
    if not (contours == []):
        max_contour = np.argmax([cv2.contourArea(x) for x in contours])
        contours.pop(max_contour)

        if not (contours == []):
            max_contour = np.argmax([cv2.contourArea(x) for x in contours])
            contours.pop(max_contour)
        if not (contours == []):
            for c in contours:
                cv2.drawContours(image,[c], 0, (0,0,0), -1)
    
    # Fill holes
    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(contours)
    if not (contours == []):
        max_contour = np.argmax([cv2.contourArea(x) for x in contours])
        contours.pop(max_contour)
        if not (contours == []):
            max_contour = np.argmax([cv2.contourArea(x) for x in contours])
            contours.pop(max_contour)
        if not (contours == []):
            cv2.drawContours(image, contours, -1, (1,1,1), -1) #255,255,255

    return image
