import os
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from PIL import Image
from skimage import exposure, draw
from scipy.ndimage import distance_transform_edt
import pandas as pd


def process_file(filepath):
    with tiff.TiffFile(filepath) as tif:
        frame = tif.pages[0].asarray()
    return frame


def enhance_contrast_percentile(image):
    img_array = np.array(image)
    p2, p98 = np.percentile(img_array, (2, 98))
    img_array_rescaled = exposure.rescale_intensity(
        img_array,
        in_range=(p2, p98),
        out_range=(0, 255)
    )
    img_enhanced = Image.fromarray(img_array_rescaled.astype(np.uint8))
    return img_enhanced, (p2, p98)


def select_polygons(image):
    fig, ax = plt.subplots()
    enhanced_image, _ = enhance_contrast_percentile(image)
    ax.imshow(enhanced_image, cmap='Greens')
    plt.axis('off')
    polygons = []
    coords = []

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


def apply_mask_and_find_distances(image, mask, pixel_to_micrometer=0.2):
    mask_array = np.zeros(image.shape, dtype=bool)
    for rr, cc in mask:
        mask_array[rr, cc] = True

    non_zero_pixels = (image > 0)
    mask_array &= non_zero_pixels

    distance_transform = distance_transform_edt(~mask_array)

    distances_all_pixels = distance_transform[non_zero_pixels]
    distances_all_pixels_micrometers = distances_all_pixels * pixel_to_micrometer

    bin_edges = np.arange(0, 155, 5)
    hist_all_pixels, _ = np.histogram(distances_all_pixels_micrometers, bins=bin_edges)

    return hist_all_pixels, bin_edges


def apply_mask_and_find_all_distances(image, mask, pixel_to_micrometer=0.2):

    mask_array = np.zeros(image.shape, dtype=bool)
    for rr, cc in mask:
        mask_array[rr, cc] = True

    distance_transform = distance_transform_edt(~mask_array)

    distances_all_pixels = distance_transform
    distances_all_pixels_micrometers = distances_all_pixels * pixel_to_micrometer

    bin_edges = np.arange(0, 155, 5)
    hist_all_pixels, _ = np.histogram(distances_all_pixels_micrometers, bins=bin_edges)

    return hist_all_pixels, bin_edges




def calculate_selected_pixel_fraction(px_histograms, tp_histograms, bin_edges):
    selected_fractions = []
    total_non_zero_pixels = []
    data = []

    for px_hist, tp_hist in zip(px_histograms, tp_histograms):
        # Total non-zero pixels across all distance ranges
        total_px = np.sum(px_hist)

        if total_px == 0:
            selected_fraction = np.zeros_like(px_hist, dtype=float)
        else:
            selected_fraction = px_hist / total_px


            for i in range(len(px_hist)):
                data.append({
                    'Distance Interval (micrometers)': f"{bin_edges[i]:.1f} - {bin_edges[i + 1]:.1f}",
                    'Selected Pixel Fraction': selected_fraction[i],
                    'Total Pixels (tp_hist)': tp_hist[i]
                })


        # Create a DataFrame and export to Excel
        df = pd.DataFrame(data)
        df.to_excel(r'.xlsx', index=False) #excel file path for output

    return selected_fractions, total_non_zero_pixels




def main(folder_path, gfp_files, alx647_files, labels):

    px_histograms = []
    tp_histograms = []
    bin_edges = np.arange(0, 155, 5)

    for gfp_file, alx647_file in zip(gfp_files, alx647_files):
        gfp_filepath = os.path.join(folder_path, gfp_file)
        gfp_frame = process_file(gfp_filepath)
        polygons = select_polygons(gfp_frame)

        alx647_filepath = os.path.join(folder_path, alx647_file)
        alx647_frame = process_file(alx647_filepath)

        print(f"Processing Alx647 file: {alx647_file}")
        px_hist, _ = apply_mask_and_find_distances(alx647_frame, polygons)
        tp_hist, _ = apply_mask_and_find_all_distances(alx647_frame, polygons)

        px_histograms.append(px_hist)
        tp_histograms.append(tp_hist)



    # Calculate and plot selected pixel fraction
    selected_fractions, total_non_zero_pixels = calculate_selected_pixel_fraction(px_histograms, tp_histograms, bin_edges)



if __name__ == "__main__":
    folder_path = r''  #work directory path
    gfp_files = [
          # tif files of selected gfp frames



    ]
    alx647_files = [
          #corresponding tif files of thresholded and erosion masked alx647 files



    ]
    labels = [
         #corresponding labels like "GFP1","GFP2"




    ]

    main(folder_path, gfp_files, alx647_files, labels)
