import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('bead_maze2.jpeg')

if image is None:
    print("Error loading the image. Check the file path and ensure the file is accessible.")
else:
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection using Canny
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)

    # Finding contours based on the edges detected
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

    # Display the results
    plt.figure(figsize=(10, 10))
    plt.subplot(121), plt.imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB)), plt.title('Edge Detection')
    plt.subplot(122), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Contours on Image')
    plt.show()
