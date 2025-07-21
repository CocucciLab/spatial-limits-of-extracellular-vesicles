import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure, draw
from scipy.ndimage import distance_transform_edt
import os
from tabulate import tabulate
import pickle
import pandas as pd

# Function to process the TIFF file
def process_tiff(file_path):
    try:
        with tiff.TiffFile(file_path) as tif:
            return [tif.pages[0].asarray().astype(np.float64)]
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return []

# Function to enhance image contrast
def enhance_contrast(image):
    p2, p98 = np.percentile(image, (2, 98))
    return exposure.rescale_intensity(image, in_range=(p2, p98))

# Function to create masks using polygon selection
def select_polygons(image):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='Reds')
    plt.axis('off')
    polygons, coords = [], []

    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            coords.append((event.xdata, event.ydata))
            ax.plot(event.xdata, event.ydata, 'bo')
            plt.draw()

    def on_key(event):
        nonlocal coords
        if event.key == 'enter' and len(coords) >= 3:
            polygon_coords = np.array(coords)
            rr, cc = draw.polygon(polygon_coords[:, 1], polygon_coords[:, 0], image.shape)
            polygons.append((rr, cc))
            ax.fill(polygon_coords[:, 0], polygon_coords[:, 1], alpha=0.3)
            coords.clear()
            plt.draw()
        elif event.key == 'escape':
            plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.title('Click to select polygon points, press Enter to finalize each polygon, Escape to finish')
    plt.show()

    return polygons

# Function to compute distances to the nearest polygon boundary
def compute_distances_using_distance_transform(image, polygon_masks):
    combined_mask = np.zeros_like(image, dtype=bool)
    for mask in polygon_masks:
        combined_mask |= mask

    non_zero_mask = (image > 0)
    distance_transform = distance_transform_edt(~combined_mask)
    filtered_pixels = non_zero_mask & ~combined_mask
    distances = distance_transform[filtered_pixels]
    hist, bin_edges = np.histogram(distances, bins=np.arange(0, 800, 25))
    return hist, bin_edges

# Function to save masks to a file
def save_masks(masks, mask_file):
    with open(mask_file, 'wb') as f:
        pickle.dump(masks, f)

# Function to load masks from a file
def load_masks(mask_file):
    with open(mask_file, 'rb') as f:
        return pickle.load(f)

# Function to process each TIFF file
def process_file(file_path, mask_file=None, use_saved_masks=False):
    frames = process_tiff(file_path)
    if not frames:
        return None, None, None, None, None

    print(f"Processing file: {file_path}")

    single_frame = frames[0]
    contrast_image = enhance_contrast(single_frame[..., 0])

    if use_saved_masks and mask_file and os.path.exists(mask_file):
        print("Loading saved masks...")
        polygons = load_masks(mask_file)
    else:
        print("Creating new masks...")
        polygons = select_polygons(contrast_image)
        if mask_file:
            save_masks(polygons, mask_file)

    polygon_masks = [np.zeros_like(single_frame[..., 0], dtype=bool) for _ in polygons]
    for i, (rr, cc) in enumerate(polygons):
        polygon_masks[i][rr, cc] = True

    hist, bins = compute_distances_using_distance_transform(single_frame[..., 0], polygon_masks)

    total_pixels_image = single_frame[..., 0]
    combined_mask = np.zeros_like(total_pixels_image, dtype=bool)
    for mask in polygon_masks:
        combined_mask |= mask

    distance_transform = distance_transform_edt(~combined_mask)
    filtered_pixels = (total_pixels_image > 0)
    total_pixel_distances = distance_transform[filtered_pixels]
    total_pixel_hist, total_pixel_bins = np.histogram(total_pixel_distances, bins=np.arange(0, 800, 25))

    return hist, bins, None, total_pixel_hist, total_pixel_bins


# function to compute distances for all pixels
def plot_total_pixel_distances(files_to_process, folder_path):

    manual_labels = [ #write corresponding labels like 'Red1','Red2'


    ]

    for idx, file in enumerate(files_to_process):
        if idx >= len(manual_labels):
            break
        file_path = os.path.join(folder_path, file)
        mask_file = file_path + '.mask.pkl'
        hist, bins, _, total_pixel_hist, _ = process_file(file_path, mask_file, use_saved_masks=True)

        if total_pixel_hist is not None:
            frames = process_tiff(file_path)
            single_frame = frames[0]
            polygons = load_masks(mask_file)
            polygon_masks = [np.zeros_like(single_frame[..., 0], dtype=bool) for _ in polygons]
            for i, (rr, cc) in enumerate(polygons):
                polygon_masks[i][rr, cc] = True

            combined_mask = np.zeros_like(single_frame[..., 0], dtype=bool)
            for mask in polygon_masks:
                combined_mask |= mask

            distance_transform = distance_transform_edt(~combined_mask)
            all_pixel_distances = distance_transform

            total_pixel_hist, _ = np.histogram(all_pixel_distances, bins=np.arange(0, 800, 25))
            bins_micrometers = (bins[:-1] + 25) * 0.2


            # Print data for Total Pixel Count vs. Distance
            print(f"\nTotal Pixel Count vs. Distance for {manual_labels[idx]}:")
            print(list(zip(bins_micrometers, total_pixel_hist)))



# function to create tables for normalized count
def create_normalized_and_total_pixel_tables(files_to_process, folder_path):
    all_normalized_ratios = []
    common_bins = None

    for file in files_to_process:
        file_path = os.path.join(folder_path, file)
        mask_file = file_path + '.mask.pkl'
        hist, bins, _, total_pixel_hist, _ = process_file(file_path, mask_file, use_saved_masks=True)

        if hist is not None:
            total_frequency = np.sum(hist)
            normalized_hist = hist / total_frequency if total_frequency > 0 else np.zeros_like(hist)

            all_normalized_ratios.append(normalized_hist)


            if common_bins is None:
                common_bins = bins
            else:
                common_bins = np.intersect1d(common_bins, bins)

    # Create normalized ratios table
    distance_values = (common_bins[:-1] + 25) * 0.2
    normalized_table = [['Distance (µm)'] + [f"{label}" for label in files_to_process]]

    for idx in range(len(distance_values)):
        row = [distance_values[idx]]
        for normalized_ratios in all_normalized_ratios:
            if idx < len(normalized_ratios):
                row.append(normalized_ratios[idx])
            else:
                row.append(0)
        normalized_table.append(row)


    return normalized_table

def main(folder_path):
    use_saved_masks = input("Do you want to use saved masks? (yes/no): ").strip().lower() == 'yes'

    files_to_process = [f for f in os.listdir(folder_path) if
                        f.endswith('.tif') and 'red' in f.lower() and 'alx647' in f.lower()]
    if not files_to_process:
        print("No matching TIFF files found.")
        return

    print("Files to process:")
    for file in files_to_process:
        print(file)


    # function to calculate total pixel distances
    plot_total_pixel_distances(files_to_process, folder_path)

    # functions to create  tables
    normalized_table = create_normalized_and_total_pixel_tables(files_to_process, folder_path)

    excel_file_path = os.path.join(folder_path, '.xlsx')  #output excel file name
    with pd.ExcelWriter(excel_file_path) as writer:

        pd.DataFrame(normalized_table[1:], columns=normalized_table[0]).to_excel(writer, sheet_name='Normalized Ratios', index=False)


    print(f"\nResults exported to: {excel_file_path}")

# Example usage
if __name__ == "__main__":
    folder_path = r""   #word directtory path
    main(folder_path)

