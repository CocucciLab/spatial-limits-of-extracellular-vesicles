import numpy as np
import tifffile
from PIL import Image
import os

# List of initial TIFF file paths to process
initial_files = [
    r'',  #input the alx647 channel tiff file with multiple frames


]

# Output directory for the selected frames and masks
output_dir = r''

os.makedirs(output_dir, exist_ok=True)

# Threshold value
threshold = 600

best_frame_indices = []
file_names = []

# Process each initial file to find the best frame and save thresholded masks
for file_path in initial_files:

    tiff = tifffile.imread(file_path)
    max_red_sum = -1
    best_frame_index = -1
    frame_count = tiff.shape[0]

    for i in range(frame_count):
        frame = tiff[i]
        red_channel = frame[:, :, 0]
        red_sum = np.sum(red_channel)

        if red_sum > max_red_sum:
            max_red_sum = red_sum
            best_frame_index = i

    # Record the best frame index
    best_frame_indices.append(best_frame_index)
    file_names.append(os.path.basename(file_path))

    # Get the best frame for thresholding
    best_frame = tiff[best_frame_index]
    red_channel = best_frame[:, :, 0]

    # Apply thresholding
    binary_mask = red_channel > threshold

    height, width = binary_mask.shape
    output_image = Image.new('RGBA', (width, height), (0, 0, 0, 0))

    for y in range(height):
        for x in range(width):
            if binary_mask[y, x]:
                output_image.putpixel((x, y), (255, 0, 0, 255))

    base_name_no_ext = os.path.splitext(os.path.basename(file_path))[0]
    mask_output_file_name = f"{base_name_no_ext}_thresholdmask.tif"
    mask_output_file_path = os.path.join(output_dir, mask_output_file_name)
    output_image.save(mask_output_file_path)

    print(f"Processed file: {file_path}")
    print(f"The frame with the highest sum of red pixels is: Frame {best_frame_index}")
    print(f"Thresholded mask saved as: {mask_output_file_path}")

print("Best frame indices from initial files:", best_frame_indices)



import tifffile
import os

# List of corresponding TIFF file paths to export the selected frames
corresponding_files = [
    r'', #write the corresponding RGB tiff file path of the alx647 file


]


output_dir = r''


os.makedirs(output_dir, exist_ok=True)

# Process each corresponding file to export the selected frames
for file_path, best_frame_index in zip(corresponding_files, best_frame_indices):

    tiff = tifffile.imread(file_path)

    if best_frame_index < tiff.shape[0]:
        selected_frame = tiff[best_frame_index]
    else:
        print(f"Index {best_frame_index} is out of bounds for file: {file_path}")
        continue

    base_name_no_ext = os.path.splitext(os.path.basename(file_path))[0]
    selected_frame_output_file_name = f"{base_name_no_ext}_selected_frame.tif"
    selected_frame_output_file_path = os.path.join(output_dir, selected_frame_output_file_name)

    tifffile.imwrite(selected_frame_output_file_path, selected_frame)

    print(f"Selected frame saved as: {selected_frame_output_file_path}")
