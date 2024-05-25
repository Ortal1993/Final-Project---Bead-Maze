import cv2
import numpy as np


def filter_contours_by_size(contours, min_area, min_length):
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        length = cv2.arcLength(contour, True)
        if area > min_area and length > min_length:
            valid_contours.append(contour)
    return valid_contours


def select_contour_and_calibrate(img, title):
    points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            points.append((x, y))
            if len(points) == 2:
                cv2.destroyAllWindows()

    cv2.imshow(title, img)
    cv2.setMouseCallback(title, click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return points


def process_image(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # You might need to adjust the Canny thresholds and the blur level depending on your image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = filter_contours_by_size(contours, min_area=100, min_length=200)

    # Allow user to select contour
    for contour in filtered_contours:
        img_copy = img.copy()
        cv2.drawContours(img_copy, [contour], -1, (0, 255, 0), 3)
        cv2.imshow("Select Contour", img_copy)
        if cv2.waitKey(0) == ord('y'):
            cv2.destroyAllWindows()
            # Allow user to select calibration points
            points = select_contour_and_calibrate(img_copy, "Select Two Points for Calibration")
            return contour, points
    cv2.destroyAllWindows()
    return None, None


# Usage
# Update the filename as per the correct image name
contour, calibration_points = process_image('face_view.jpeg')
if contour is not None:
    print("Contour selected and calibration points are set.")
else:
    print("No contour was selected.")
