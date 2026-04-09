"""
Task 3: Morphological Processing
Apply dilation and erosion to refine segmented regions
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Image path
IMAGE_PATH = r"C:\Users\gtcam\OneDrive\Pictures\Camera Roll\OIP (2).webp"

def load_and_convert_to_grayscale(image_path):
    """Load image and convert to grayscale"""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image

def apply_otsu_thresholding(grayscale_image):
    """Apply Otsu's automatic thresholding"""
    _, binary_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image

def create_kernel(kernel_type='ellipse', size=5):
    """Create morphological kernel"""
    if kernel_type == 'rect':
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    elif kernel_type == 'ellipse':
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    elif kernel_type == 'cross':
        return cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))
    else:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

def apply_erosion(binary_image, kernel_size=5, iterations=1):
    """Apply erosion operation"""
    kernel = create_kernel('ellipse', kernel_size)
    eroded = cv2.erode(binary_image, kernel, iterations=iterations)
    return eroded

def apply_dilation(binary_image, kernel_size=5, iterations=1):
    """Apply dilation operation"""
    kernel = create_kernel('ellipse', kernel_size)
    dilated = cv2.dilate(binary_image, kernel, iterations=iterations)
    return dilated

def apply_opening(binary_image, kernel_size=5):
    """Apply opening (erosion followed by dilation) - removes small objects"""
    kernel = create_kernel('ellipse', kernel_size)
    opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    return opened

def apply_closing(binary_image, kernel_size=5):
    """Apply closing (dilation followed by erosion) - fills small holes"""
    kernel = create_kernel('ellipse', kernel_size)
    closed = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    return closed

def apply_gradient(binary_image, kernel_size=5):
    """Apply morphological gradient (dilation - erosion)"""
    kernel = create_kernel('ellipse', kernel_size)
    gradient = cv2.morphologyEx(binary_image, cv2.MORPH_GRADIENT, kernel)
    return gradient

def calculate_region_changes(original, processed):
    """Calculate changes in region properties"""
    original_pixels = np.sum(original == 255)
    processed_pixels = np.sum(processed == 255)
    
    pixel_change = processed_pixels - original_pixels
    percentage_change = (pixel_change / original_pixels) * 100 if original_pixels > 0 else 0
    
    return original_pixels, processed_pixels, pixel_change, percentage_change

def count_connected_components(binary_image):
    """Count number of connected components"""
    num_labels, labels = cv2.connectedComponents(binary_image)
    return num_labels - 1  # Exclude background

def main():
    print("=" * 60)
    print("TASK 3: MORPHOLOGICAL PROCESSING")
    print("=" * 60)
    
    # Load and segment image
    print("\n1. Loading and segmenting image...")
    grayscale_image = load_and_convert_to_grayscale(IMAGE_PATH)
    binary_image = apply_otsu_thresholding(grayscale_image)
    print(f"   ✓ Image loaded: Shape = {grayscale_image.shape}")
    print(f"   ✓ Initial white pixels: {np.sum(binary_image == 255)}")
    print(f"   ✓ Initial connected components: {count_connected_components(binary_image)}")
    
    # Apply erosion
    print("\n2. Applying Erosion...")
    eroded_1iter = apply_erosion(binary_image, kernel_size=5, iterations=1)
    eroded_2iter = apply_erosion(binary_image, kernel_size=5, iterations=2)
    
    orig_px, ero1_px, ero1_change, ero1_pct = calculate_region_changes(binary_image, eroded_1iter)
    _, ero2_px, ero2_change, ero2_pct = calculate_region_changes(binary_image, eroded_2iter)
    
    print(f"   ✓ Erosion (1 iteration):")
    print(f"      Original pixels: {orig_px}")
    print(f"      After erosion: {ero1_px}")
    print(f"      Change: {ero1_change} pixels ({ero1_pct:.2f}%)")
    print(f"      Connected components: {count_connected_components(eroded_1iter)}")
    
    print(f"   ✓ Erosion (2 iterations):")
    print(f"      After erosion: {ero2_px}")
    print(f"      Change: {ero2_change} pixels ({ero2_pct:.2f}%)")
    print(f"      Connected components: {count_connected_components(eroded_2iter)}")
    
    # Apply dilation
    print("\n3. Applying Dilation...")
    dilated_1iter = apply_dilation(binary_image, kernel_size=5, iterations=1)
    dilated_2iter = apply_dilation(binary_image, kernel_size=5, iterations=2)
    
    _, dil1_px, dil1_change, dil1_pct = calculate_region_changes(binary_image, dilated_1iter)
    _, dil2_px, dil2_change, dil2_pct = calculate_region_changes(binary_image, dilated_2iter)
    
    print(f"   ✓ Dilation (1 iteration):")
    print(f"      Original pixels: {orig_px}")
    print(f"      After dilation: {dil1_px}")
    print(f"      Change: {dil1_change} pixels ({dil1_pct:.2f}%)")
    print(f"      Connected components: {count_connected_components(dilated_1iter)}")
    
    print(f"   ✓ Dilation (2 iterations):")
    print(f"      After dilation: {dil2_px}")
    print(f"      Change: {dil2_change} pixels ({dil2_pct:.2f}%)")
    print(f"      Connected components: {count_connected_components(dilated_2iter)}")
    
    # Apply opening and closing
    print("\n4. Applying Opening and Closing...")
    opened = apply_opening(binary_image, kernel_size=5)
    closed = apply_closing(binary_image, kernel_size=5)
    
    _, open_px, open_change, open_pct = calculate_region_changes(binary_image, opened)
    _, close_px, close_change, close_pct = calculate_region_changes(binary_image, closed)
    
    print(f"   ✓ Opening (removes small objects):")
    print(f"      Original pixels: {orig_px}")
    print(f"      After opening: {open_px}")
    print(f"      Change: {open_change} pixels ({open_pct:.2f}%)")
    print(f"      Connected components: {count_connected_components(opened)}")
    
    print(f"   ✓ Closing (fills holes):")
    print(f"      Original pixels: {orig_px}")
    print(f"      After closing: {close_px}")
    print(f"      Change: {close_change} pixels ({close_pct:.2f}%)")
    print(f"      Connected components: {count_connected_components(closed)}")
    
    # Apply gradient
    print("\n5. Applying Morphological Gradient...")
    gradient = apply_gradient(binary_image, kernel_size=5)
    print(f"   ✓ Gradient (edge detection):")
    print(f"      Edge pixels: {np.sum(gradient == 255)}")
    
    # Combined refinement
    print("\n6. Applying Combined Refinement (Opening + Closing)...")
    refined = apply_closing(apply_opening(binary_image, kernel_size=5), kernel_size=5)
    _, refined_px, refined_change, refined_pct = calculate_region_changes(binary_image, refined)
    
    print(f"   ✓ Original pixels: {orig_px}")
    print(f"   ✓ Refined pixels: {refined_px}")
    print(f"   ✓ Total change: {refined_change} pixels ({refined_pct:.2f}%)")
    print(f"   ✓ Connected components: {count_connected_components(refined)}")
    
    # Visualization
    print("\n7. Creating visualization...")
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    # Row 1: Original and Erosion
    axes[0, 0].imshow(binary_image, cmap='gray')
    axes[0, 0].set_title('Original Segmented Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(eroded_1iter, cmap='gray')
    axes[0, 1].set_title('Erosion (1 iter)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(eroded_2iter, cmap='gray')
    axes[0, 2].set_title('Erosion (2 iter)')
    axes[0, 2].axis('off')
    
    # Row 2: Dilation
    axes[1, 0].imshow(binary_image, cmap='gray')
    axes[1, 0].set_title('Original Segmented Image')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(dilated_1iter, cmap='gray')
    axes[1, 1].set_title('Dilation (1 iter)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(dilated_2iter, cmap='gray')
    axes[1, 2].set_title('Dilation (2 iter)')
    axes[1, 2].axis('off')
    
    # Row 3: Opening, Closing, Gradient
    axes[2, 0].imshow(opened, cmap='gray')
    axes[2, 0].set_title('Opening (Remove Noise)')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(closed, cmap='gray')
    axes[2, 1].set_title('Closing (Fill Holes)')
    axes[2, 1].axis('off')
    
    axes[2, 2].imshow(refined, cmap='gray')
    axes[2, 2].set_title('Refined (Opening + Closing)')
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(r'c:\Users\gtcam\OneDrive\Desktop\DIP image processing\task3_morphological_result.png', dpi=150, bbox_inches='tight')
    print("   ✓ Result saved as 'task3_morphological_result.png'")
    plt.show()
    
    # Summary
    print("\n" + "=" * 60)
    print("MORPHOLOGICAL PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Erosion: Removes small objects, reduces region size")
    print(f"Dilation: Fills small holes, expands region size")
    print(f"Opening: Noise removal (erosion → dilation)")
    print(f"Closing: Hole filling (dilation → erosion)")
    print(f"Opening + Closing achieved {abs(refined_pct):.2f}% region refinement")
    print("=" * 60)

if __name__ == "__main__":
    main()
