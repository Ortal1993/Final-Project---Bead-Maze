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


def find_closest_contour_point(contour, point):
    distances = np.sqrt((contour[:, 0, 0] - point[0]) ** 2 + (contour[:, 0, 1] - point[1]) ** 2)
    min_index = np.argmin(distances)
    return min_index


def calculate_contour_length(contour, scale_factor):
    length_pixel = cv2.arcLength(contour, True)
    return length_pixel * scale_factor


def display_and_trim_contour(img, contour):
    global points
    points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 2:
                cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
                points.append((x, y))
                cv2.imshow('Select Trim Points', img)
                if len(points) == 2:
                    start_idx = find_closest_contour_point(contour, points[0])
                    end_idx = find_closest_contour_point(contour, points[1])
                    trimmed_contour = contour[min(start_idx, end_idx):max(start_idx, end_idx) + 1]
                    img_copy = img.copy()
                    cv2.drawContours(img_copy, [trimmed_contour], -1, (0, 255, 0), 2)
                    cv2.imshow('Trimmed Contour', img_copy)
                    cv2.waitKey(0)

                    # Calculate scaling factor
                    pixel_distance = np.linalg.norm(np.array(points[0]) - np.array(points[1]))
                    real_distance = float(input("Enter the real world distance between the points (in cm): "))
                    scale_factor = real_distance / pixel_distance
                    contour_length = calculate_contour_length(trimmed_contour, scale_factor)
                    print(f"The scaled contour length is: {contour_length:.2f} cm")

    cv2.imshow('Select Trim Points', img)
    cv2.setMouseCallback('Select Trim Points', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def display_contours(img, contours):
    for i, contour in enumerate(contours):
        img_copy = img.copy()
        cv2.drawContours(img_copy, [contour], -1, (0, 255, 0), 3)
        cv2.imshow('Contour', img_copy)
        response = cv2.waitKey(0)
        if response == ord('y'):  # Press 'y' if this is the correct contour
            cv2.destroyAllWindows()
            return contour
        elif response == ord('n'):  # Press 'n' to check the next contour
            continue
    cv2.destroyAllWindows()
    return None


def main(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("Error loading the image. Check the file path and ensure the file is accessible.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = filter_contours_by_size(contours, min_area=100, min_length=200)
    selected_contour = display_contours(img, filtered_contours)
    if selected_contour is not None:
        print("Correct contour selected.")
        display_and_trim_contour(img, selected_contour)
    else:
        print("No contour selected.")


# Usage
main('bead_maze2.jpeg')
