import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from mpl_toolkits.mplot3d import Axes3D

def get_hsv_range(image, title):
    roi = cv2.selectROI(title, image, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    if roi == (0, 0, 0, 0):
        return None
    roi_cropped = image[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
    hsv_cropped = cv2.cvtColor(roi_cropped, cv2.COLOR_BGR2HSV)
    mean_hsv = np.mean(hsv_cropped, axis=(0, 1))
    std_hsv = np.std(hsv_cropped, axis=(0, 1))
    lower_color = np.maximum(mean_hsv - 2 * std_hsv, 0)
    upper_color = np.minimum(mean_hsv + 2 * std_hsv, [180, 255, 255])
    return lower_color.astype(int), upper_color.astype(int)

def select_contour(image, hsv_lower, hsv_upper):
    mask = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), hsv_lower, hsv_upper)
    masked_image = cv2.bitwise_and(image, image, mask=mask)  # Apply mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    window_title = 'Select Contour'
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

    for i, contour in enumerate(contours):
        img_copy = masked_image.copy()
        cv2.drawContours(img_copy, [contour], -1, (0, 255, 0), 3)
        cv2.imshow(window_title, img_copy)
        key = cv2.waitKey(0)
        if key == ord('y'):
            cv2.destroyAllWindows()
            return contour
        elif key == ord('n'):
            continue

    cv2.destroyAllWindows()
    return None

def main():
    img_face = cv2.imread('face_view.jpeg')
    img_top = cv2.imread('top_view.jpeg')

    lower_face, upper_face = get_hsv_range(img_face, "Define HSV Range for Face View")
    lower_top, upper_top = get_hsv_range(img_top, "Define HSV Range for Top View")

    contour_face = select_contour(img_face, lower_face, upper_face)
    if contour_face is None:
        print("No suitable contour selected for the face view.")
        return

    contour_top = select_contour(img_top, lower_top, upper_top)
    if contour_top is None:
        print("No suitable contour selected for the top view.")
        return

    # Continue with point selection and spline fitting...

if __name__ == "__main__":
    main()
