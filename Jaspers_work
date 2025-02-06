import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.io import imread
import cv2
from skimage import exposure
from skimage.feature import canny
from skimage.measure import label, regionprops
from math import atan2, degrees
from skimage.color import rgb2gray

def tellme(s):
    print(s)
    plt.title(s, fontsize=16)
    plt.draw()

def plot_rgb_histogram(rgb_array):
    """
    Plots the histograms for the Red, Green, and Blue channels of an image.
    
    This helps in understanding the color distribution in the image.
    """
    # Split the image into R, G, B channels
    r_channel = rgb_array[:, :, 0].flatten()
    g_channel = rgb_array[:, :, 1].flatten()
    b_channel = rgb_array[:, :, 2].flatten()
    
    # Define histogram bins
    bins = 256  # 256 intensity levels (0-255)
    
    # Create the figure and axis
    plt.figure(figsize=(10, 6))
    
    # Plot histograms
    plt.hist(r_channel, bins=bins, color='red', alpha=0.6, label='Red Channel')
    plt.hist(g_channel, bins=bins, color='green', alpha=0.6, label='Green Channel')
    plt.hist(b_channel, bins=bins, color='blue', alpha=0.6, label='Blue Channel')

    # Labels and title
    plt.xlabel("Pixel Intensity (0-255)")
    plt.ylabel("Frequency")
    plt.title("RGB Histogram of the Image")
    plt.legend()
    
    # Show plot
    plt.show()

def threshold_yellow(rgb_array):
    """
    Binarizes an image to detect yellow regions based on RGB thresholds.
    Pixels within the defined range for yellow are set to 1 (white), others to 0 (black).
    
    Returns a binary image highlighting yellow regions.
    """
    # Extract RGB channels
    R, G, B = rgb_array[:, :, 0], rgb_array[:, :, 1], rgb_array[:, :, 2]
    
    # Define threshold ranges - Need UV light to edit this and perfect as it changes
    red_mask = (R >= 125) & (R <= 255)
    green_mask = (G >= 90) & (G <= 125) # This is the important one supposedly
    blue_mask = (B >= 25) & (B <= 200)

    # Create binary mask for yellow detection
    yellow_mask = red_mask & green_mask & blue_mask
    
    # Convert boolean mask to binary image (0 or 1)
    binary_image = yellow_mask.astype(np.uint8)

    return binary_image

def find_centroid(binary_image):
    """
    Finds the centroid (center of mass) of the yellow spot in the binary image
    using first-order moments.
    
    Returns the (x, y) coordinates of the centroid.
    """
    # Get the indices of the non-zero (white) pixels in the binary image
    y_indices, x_indices = np.where(binary_image == 1)

    # Calculate the first-order moments
    M_x = np.sum(x_indices)
    M_y = np.sum(y_indices)

    # Calculate the total number of white pixels
    total_pixels = len(x_indices)

    # Compute the centroid coordinates
    C_x = M_x / total_pixels
    C_y = M_y / total_pixels

    return C_x, C_y

def find_centroids_of_all_blobs(binary_image):
    """
    Finds the centroids of all blobs (connected components) in the binary image.
    
    Returns a list of tuples containing the (x, y) coordinates of the centroids of all blobs.
    """
    # Label the connected components (blobs)
    labeled_image = label(binary_image)

    # Get properties of each labeled region (blob)
    regions = regionprops(labeled_image)

    # List to store centroids of all blobs
    centroids = []

    # Calculate the centroid for each region
    for region in regions:
        # Get the centroid of the current region
        centroid = region.centroid  # (y, x) format
        centroids.append(centroid)

    return centroids

def crop(rgb_array):
    # Display image
    plt.imshow(rgb_array)
    plt.title('Image')
    plt.axis('off')
    plt.draw()

    tellme("Please click on the corners of the TLC plate")
    corners = np.array(plt.ginput(4, 0, True))
    rounded_corners = corners.astype(int)

    x_min_crop = min([val[0] for val in rounded_corners])
    x_max_crop = max([val[0] for val in rounded_corners])
    y_min_crop = min([val[1] for val in rounded_corners])
    y_max_crop = max([val[1] for val in rounded_corners])

    cropped_rgb_array = rgb_array[y_min_crop:y_max_crop, x_min_crop:x_max_crop]
    return cropped_rgb_array

def convert_to_grayscale(rgb_array):
    pil_image = Image.fromarray(rgb_array)  # Convert NumPy array to PIL image
    gray_image = pil_image.convert('L')       # Convert to grayscale (L mode)
    return np.array(gray_image)

def apply_bilateral_filter(gray_array):
    return cv2.bilateralFilter((gray_array * 255).astype(np.uint8), d=2, sigmaColor=100, sigmaSpace=75)

def apply_CLAHE(smoothed_image_bilateral):
    return exposure.equalize_adapthist(smoothed_image_bilateral, clip_limit=0.03)

def detect_edges(enhanced_image):
    return canny(enhanced_image, sigma=10)

def convert_edges_to_uint8(edges):
    return (edges * 255).astype(np.uint8)

def detect_horizontal_lines(edges_uint8):
    return cv2.HoughLinesP(edges_uint8, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)

def draw_horizontal_lines(rgb_array, lines, angle_threshold):
    image_with_lines = ((rgb_array * 255).astype(np.uint8))
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = degrees(atan2(y2 - y1, x2 - x1))
            if abs(angle) < angle_threshold or abs(angle - 180) < angle_threshold:
                cv2.line(image_with_lines, (x1, y1), (x2, y2), (255, 0, 0), 1)  # Draw in blue
    return image_with_lines

def main():
    # Load image and convert to grayscale
    image_path = '/Users/jasper/Library/CloudStorage/OneDrive-Personal/Documents/Bioengineering Year 3/Project/Development/Image.jpeg'
    tlc_image = imread(image_path)
    rgb_array = np.array(tlc_image)

    rgb_array = crop(rgb_array)

    pil_image = Image.fromarray(rgb_array)  # Convert NumPy array to PIL image
    gray_image = pil_image.convert('L')       # Convert to grayscale (L mode)
    gray_array = np.array(gray_image)

    # Generating figure 1
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes.ravel()

    # Display grayscale image
    ax[0].imshow(rgb_array)
    ax[0].set_title('Image')
    ax[0].set_axis_off()

    smoothed_image_bilateral = apply_bilateral_filter(gray_array)
    enhanced_image = apply_CLAHE(smoothed_image_bilateral)
    edges = detect_edges(enhanced_image)
    edges_uint8 = convert_edges_to_uint8(edges)
    lines = detect_horizontal_lines(edges_uint8)

    image_with_lines = draw_horizontal_lines(rgb_array, lines, angle_threshold=10)

    ax[1].imshow(image_with_lines)
    ax[1].set_title('Detected Horizontal Lines')
    ax[1].set_axis_off()

    plt.show()

    plot_rgb_histogram(rgb_array)

    binary_yellow = threshold_yellow(rgb_array)

    # Display the binarized image
    plt.imshow(binary_yellow, cmap='gray')
    plt.title("Detected Yellow Region")
    plt.axis('off')
    plt.show()


    # Find the centroid of the yellow spot
    centroid_x, centroid_y = find_centroid(binary_yellow)

    # Display the result
    print(f"Centroid of the yellow spot: (x, y) = ({centroid_x}, {centroid_y})")

    #  Optionally, you can plot the result on the binary image:
    plt.imshow(binary_yellow, cmap='gray')
    plt.scatter(centroid_x, centroid_y, color='red', marker='x', label="Centroid")
    plt.legend()
    plt.title("Detected Yellow Region with Centroid")
    plt.axis('off')
    plt.show()

    # Find the centroids of all blobs in the binary yellow spot image
    centroids = find_centroids_of_all_blobs(binary_yellow)

    # Display the centroids
    for i, centroid in enumerate(centroids, 1):
        print(f"Centroid of Blob {i}: (x, y) = ({centroid[1]}, {centroid[0]})")

    # Optionally, plot the blobs and their centroids
    plt.imshow(binary_yellow, cmap='gray')
    for centroid in centroids:
        plt.scatter(centroid[1], centroid[0], color='red', marker='x')  # Centroid in red
    plt.title("Detected Blobs with Centroids")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()

