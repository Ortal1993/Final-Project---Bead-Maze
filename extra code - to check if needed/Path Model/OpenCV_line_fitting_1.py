import argparse
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import cdist

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

def fit_spline(contour, smoothness=0.2):
    x = contour[:, 0, 0]
    y = contour[:, 0, 1]
    y = max(y) - y  # Invert y-coordinates for correct plotting
    tck, u = splprep([x, y], s=smoothness)  # Smaller s means more smoothing
    new_points = splev(np.linspace(0, 1, 100), tck)
    return new_points, tck, u

def plot_difference(contour, spline_points):
    contour_points = np.column_stack((contour[:, 0, 0], max(contour[:, 0, 1]) - contour[:, 0, 1]))
    spline_points_transposed = np.column_stack(spline_points)
    distances = cdist(contour_points, spline_points_transposed, 'euclidean')
    min_distances = np.min(distances, axis=1)
    plt.figure()
    plt.plot(min_distances)
    plt.title('Distance from Contour Points to Spline')
    plt.xlabel('Contour Point Index')
    plt.ylabel('Distance (pixels)')
    plt.show()


def display_and_trim_contour(img, contours):
    array_contour = []
    colors_array = []
    for contour in contours:
        color1, color2, color3 = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        img_copy = img.copy()
        cv2.drawContours(img_copy, [contour], -1, (color1, color2, color3), 3)
        cv2.namedWindow('Contour', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Contour', 800, 600)
        cv2.imshow('Contour', img_copy)
        response = cv2.waitKey(0)
        if response == ord('y'):
            array_contour.append(contour)
            colors_array.append((color1, color2, color3))
        elif response == ord('n'):
            continue
    cv2.destroyAllWindows()

    global points
    global all_points
    points = []    
    all_points = []
    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 2:
                cv2.circle(img_selcted_contour, (x, y), 10, (255, 0, 0), -1)
                points.append((x, img_selcted_contour.shape[0] - y))  # Save inverted y
                all_points.append((x, img_selcted_contour.shape[0] - y))
                cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Image', 800, 600)
                cv2.imshow('Image', img_selcted_contour)
            #for point in all_points:
            #    cv2.circle(img_selcted_contour, (point[0], point[1]), 5, (0, 255, 0), -1)
            if len(points) == 2:
                cv2.destroyAllWindows()                
    
    trimmed_contours = []
    for i, contour in enumerate(array_contour):
        img_selcted_contour = img.copy()
        cv2.drawContours(img_selcted_contour, [contour], -1, colors_array[i], 3)
        cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Image', 800, 600)
        cv2.imshow('Image', img_selcted_contour)
        points = []
        print("Please select two points to trim the contour.")
        cv2.setMouseCallback('Image', click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()   
    
        if len(points) == 2:
            real_distance = float(input("Enter the real world distance between the points (in cm): "))
            pixel_distance = np.linalg.norm(np.array(points[0]) - np.array(points[1]))
            scale_factor = real_distance / pixel_distance  # cm per pixel

            start_idx = find_closest_contour_point(contour, points[0])
            end_idx = find_closest_contour_point(contour, points[1])
            trimmed_contour = contour[min(start_idx, end_idx):max(start_idx, end_idx) + 1]
            trimmed_contours.append(trimmed_contour)

        # Apply scaling to trimmed_contour
        spline_points_array = []
        for contour in trimmed_contours:
            scaled_trimmed_contour = np.array(contour) * scale_factor

            spline_points, tck, u = fit_spline(scaled_trimmed_contour)
            spline_points_array.append(spline_points)
            #plot_difference(scaled_trimmed_contour, spline_points)
            #plt.figure()
            plt.plot(spline_points[0], spline_points[1], 'b-', label='Fitted Spline')
            plt.plot(scaled_trimmed_contour[:, 0, 0],
                        max(scaled_trimmed_contour[:, 0, 1]) - scaled_trimmed_contour[:, 0, 1], 'ro', label='Contour Points')
            plt.legend()
            plt.show()
            
        
    return spline_points_array


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='script for testing planners')
    parser.add_argument('-img_path', '--img_path', type=str, default=None, help='insert image path')

    args = parser.parse_args()

    # prepare the map
    img = cv2.imread(args.img_path)
    if img is None:
        print("Error loading the image")
        
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)
    #ret, thresh = cv2.threshold(img_gray, 100, 200, cv2.THRESH_BINARY)
    #contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)#cv2.RETR_LIST
    contours, hierarchy = cv2.findContours(image=edges, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)#cv2.RETR_EXTERNAL, cv2.RETR_TREE, cv2.RETR_LIST ##cv2.CHAIN_APPROX_SIMPLE
    filtered_contours = filter_contours_by_size(contours, min_area=20, min_length=150)
    spline_points = display_and_trim_contour(img, filtered_contours)
    print(spline_points)
    # Close all OpenCV windows and exit the program
    cv2.destroyAllWindows()    
    

