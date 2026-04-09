"""
Task 2: Image Segmentation
Apply global thresholding and Otsu's thresholding to identify regions of interest
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, find_objects

# Image path
IMAGE_PATH = r"C:\Users\gtcam\OneDrive\Pictures\Camera Roll\OIP (2).webp"

def load_and_convert_to_grayscale(image_path):
    """Load image and convert to grayscale"""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image

def apply_global_thresholding(grayscale_image, threshold_value=127):
    """Apply global thresholding using a fixed threshold value"""
    _, binary_image = cv2.threshold(grayscale_image, threshold_value, 255, cv2.THRESH_BINARY)
    return binary_image

def apply_otsu_thresholding(grayscale_image):
    """Apply Otsu's automatic thresholding"""
    _, binary_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image

def identify_regions_of_interest(binary_image):
    """
    Identify connected components (regions of interest)
    Returns labeled image and properties of each region
    """
    # Label connected components
    labeled_array, num_features = label(binary_image // 255)
    
    # Find bounding boxes for each region
    regions = find_objects(labeled_array)
    
    # Filter regions by size (remove very small regions)
    region_properties = []
    for i, region in enumerate(regions):
        if region is not None:
            region_slice = labeled_array[region]
            region_size = np.sum(region_slice == (i + 1))
            
            if region_size > 50:  # Minimum region size threshold
                y_min, y_max = region[0].start, region[0].stop
                x_min, x_max = region[1].start, region[1].stop
                
                region_properties.append({
                    'label': i + 1,
                    'size': region_size,
                    'bbox': (x_min, y_min, x_max - x_min, y_max - y_min),
                    'centroid': ((x_min + x_max) / 2, (y_min + y_max) / 2)
                })
    
    return labeled_array, region_properties

def draw_regions_on_image(original_image, region_properties, color=(0, 255, 0), thickness=2):
    """Draw bounding boxes on the original image"""
    result_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR) if len(original_image.shape) == 2 else original_image.copy()
    
    for region in region_properties:
        x, y, w, h = region['bbox']
        cv2.rectangle(result_image, (x, y), (x + w, y + h), color, thickness)
        
        # Draw centroid
        cx, cy = int(region['centroid'][0]), int(region['centroid'][1])
        cv2.circle(result_image, (cx, cy), 5, (255, 0, 0), -1)
    
    return result_image

def compare_segmentation_results(grayscale_image):
    """Compare global and Otsu's thresholding results"""
    # Apply both methods
    global_threshold = apply_global_thresholding(grayscale_image, 127)
    otsu_threshold = apply_otsu_thresholding(grayscale_image)
    
    # Get Otsu's threshold value
    _, otsu_binary = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_value = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    
    # Identify regions for both
    labeled_global, regions_global = identify_regions_of_interest(global_threshold)
    labeled_otsu, regions_otsu = identify_regions_of_interest(otsu_threshold)
    
    return {
        'global_threshold': global_threshold,
        'otsu_threshold': otsu_threshold,
        'otsu_value': otsu_value,
        'regions_global': regions_global,
        'regions_otsu': regions_otsu,
        'labeled_global': labeled_global,
        'labeled_otsu': labeled_otsu
    }

def main():
    print("=" * 60)
    print("TASK 2: IMAGE SEGMENTATION")
    print("=" * 60)
    
    # Load grayscale image
    print("\n1. Loading and converting image to grayscale...")
    grayscale_image = load_and_convert_to_grayscale(IMAGE_PATH)
    print(f"   ✓ Image loaded: Shape = {grayscale_image.shape}")
    print(f"   ✓ Pixel value range: [{grayscale_image.min()}, {grayscale_image.max()}]")
    
    # Apply global thresholding
    print("\n2. Applying Global Thresholding (threshold = 127)...")
    global_threshold = apply_global_thresholding(grayscale_image, 127)
    white_pixels_global = np.sum(global_threshold == 255)
    print(f"   ✓ White pixels: {white_pixels_global}")
    print(f"   ✓ Foreground percentage: {(white_pixels_global / global_threshold.size) * 100:.2f}%")
    
    # Apply Otsu's thresholding
    print("\n3. Applying Otsu's Thresholding...")
    otsu_threshold = apply_otsu_thresholding(grayscale_image)
    otsu_value = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    white_pixels_otsu = np.sum(otsu_threshold == 255)
    print(f"   ✓ Otsu's threshold value: {otsu_value}")
    print(f"   ✓ White pixels: {white_pixels_otsu}")
    print(f"   ✓ Foreground percentage: {(white_pixels_otsu / otsu_threshold.size) * 100:.2f}%")
    
    # Identify regions of interest
    print("\n4. Identifying Regions of Interest...")
    comparison = compare_segmentation_results(grayscale_image)
    
    regions_global = comparison['regions_global']
    regions_otsu = comparison['regions_otsu']
    
    print(f"   ✓ Global thresholding regions: {len(regions_global)}")
    for i, region in enumerate(regions_global[:5]):
        print(f"      Region {i+1}: Size = {region['size']} pixels, BBox = {region['bbox']}")
    
    print(f"   ✓ Otsu's thresholding regions: {len(regions_otsu)}")
    for i, region in enumerate(regions_otsu[:5]):
        print(f"      Region {i+1}: Size = {region['size']} pixels, BBox = {region['bbox']}")
    
    # Draw regions on images
    print("\n5. Drawing regions on images...")
    image_global_roi = draw_regions_on_image(grayscale_image, regions_global, color=(0, 255, 0))
    image_otsu_roi = draw_regions_on_image(grayscale_image, regions_otsu, color=(255, 0, 0))
    
    # Visualization
    print("\n6. Creating visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original
    axes[0, 0].imshow(grayscale_image, cmap='gray')
    axes[0, 0].set_title('Original Grayscale Image')
    axes[0, 0].axis('off')
    
    # Global thresholding
    axes[0, 1].imshow(comparison['global_threshold'], cmap='gray')
    axes[0, 1].set_title(f'Global Thresholding (T=127)')
    axes[0, 1].axis('off')
    
    # Global with ROIs
    axes[0, 2].imshow(cv2.cvtColor(image_global_roi, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title(f'Global ROIs ({len(regions_global)} regions)')
    axes[0, 2].axis('off')
    
    # Otsu thresholding
    axes[1, 0].imshow(comparison['otsu_threshold'], cmap='gray')
    axes[1, 0].set_title(f"Otsu's Thresholding (T={otsu_value})")
    axes[1, 0].axis('off')
    
    # Otsu with ROIs
    axes[1, 1].imshow(cv2.cvtColor(image_otsu_roi, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f"Otsu's ROIs ({len(regions_otsu)} regions)")
    axes[1, 1].axis('off')
    
    # Labeled regions
    axes[1, 2].imshow(comparison['labeled_otsu'], cmap='nipy_spectral')
    axes[1, 2].set_title("Labeled Connected Components")
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(r'c:\Users\gtcam\OneDrive\Desktop\DIP image processing\task2_segmentation_result.png', dpi=150, bbox_inches='tight')
    print("   ✓ Result saved as 'task2_segmentation_result.png'")
    plt.show()
    
    # Summary
    print("\n" + "=" * 60)
    print("SEGMENTATION SUMMARY")
    print("=" * 60)
    print(f"Global Thresholding: {len(regions_global)} ROIs detected")
    print(f"Otsu's Thresholding: {len(regions_otsu)} ROIs detected")
    print(f"Otsu's threshold value automatically determined as: {otsu_value}")
    print("=" * 60)

if __name__ == "__main__":
    main()
