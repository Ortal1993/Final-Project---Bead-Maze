import cv2
import matplotlib.pyplot as plt
import numpy as np

def click_event(event, x, y, flags, params):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x,y), 3, (255, 0, 0), -1)
        points.append((x, y))
        if len(points) >= 2:
            cv2.line(img, points[-2], points[-1], (255, 0, 0), 2)
        cv2.imshow('image', img)

def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Load the image
img = cv2.imread('bead_maze2.jpeg')
cv2.imshow('image', img)
points = []

cv2.setMouseCallback('image', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Ask user for the real world distance between the first two points
if len(points) >= 2:
    real_distance = float(input("Enter the real world distance between the selected points (in cm): "))
    pixel_distance = calculate_distance(points[0], points[1])
    scale_factor = real_distance / pixel_distance  # cm per pixel

    # Assuming contours[0] is your maze path
    contours, _ = cv2.findContours(cv2.Canny(img, 100, 200), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    path_length_pixels = cv2.arcLength(contours[0], True)
    path_length_real = path_length_pixels * scale_factor

    print(f"Total path length of the bead maze is {path_length_real:.2f} cm")

    # Optionally, draw the contour on the image
    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    cv2.imshow('Path', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
