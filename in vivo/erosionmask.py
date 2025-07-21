import cv2
import numpy as np

# Load the TIFF image
input_file = r''  #input the thresholded tiff file
output_file = r'' #output path

image = cv2.imread(input_file, cv2.IMREAD_UNCHANGED)

if image is None:
    print(f"Error: Could not read the image file at {input_file}. Check the file path and integrity.")
else:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

    eroded_image = cv2.erode(image, kernel, iterations=1)

    cv2.imwrite(output_file, eroded_image)

    print(f'Eroded image saved as {output_file}')
