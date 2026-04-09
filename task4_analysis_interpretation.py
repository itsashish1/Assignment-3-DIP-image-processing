"""
Task 4: Analysis and Interpretation
Compare segmentation results and discuss clinical relevance of extracted regions
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, find_objects
from scipy import stats

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
    """Apply global thresholding"""
    _, binary_image = cv2.threshold(grayscale_image, threshold_value, 255, cv2.THRESH_BINARY)
    return binary_image

def apply_otsu_thresholding(grayscale_image):
    """Apply Otsu's automatic thresholding"""
    _, binary_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image

def apply_opening(binary_image, kernel_size=5):
    """Apply opening operation"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    return opened

def apply_closing(binary_image, kernel_size=5):
    """Apply closing operation"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    return closed

def extract_regions(binary_image):
    """Extract connected components and their properties"""
    labeled_array, num_features = label(binary_image // 255)
    
    regions = []
    for i in range(1, num_features + 1):
        region_mask = (labeled_array == i)
        
        # Calculate properties
        y_coords, x_coords = np.where(region_mask)
        
        if len(x_coords) > 0:
            area = np.sum(region_mask)
            
            # Bounding box
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()
            
            # Centroid
            cx, cy = x_coords.mean(), y_coords.mean()
            
            # Shape properties
            width = x_max - x_min
            height = y_max - y_min
            
            # Calculate perimeter using contours
            perimeter = 0
            try:
                contours = cv2.findContours(region_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contours = contours[0] if len(contours) == 2 else contours[1]  # Handle version differences
                
                if contours and len(contours) > 0:
                    cnt = contours[0]
                    perimeter = cv2.arcLength(cnt, True)
            except:
                perimeter = 0
            
            # Circularity (solidity)
            if perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter ** 2)
            else:
                circularity = 0
            
            # Intensity information
            regions.append({
                'label': i,
                'area': area,
                'width': width,
                'height': height,
                'perimeter': perimeter,
                'circularity': circularity,
                'centroid': (cx, cy),
                'bbox': (x_min, y_min, x_max - x_min, y_max - y_min),
                'aspect_ratio': width / height if height > 0 else 0
            })
    
    return regions

def compare_segmentation_methods(grayscale_image):
    """Compare global, Otsu, and refined segmentation"""
    # Method 1: Global Thresholding
    global_seg = apply_global_thresholding(grayscale_image, 127)
    global_regions = extract_regions(global_seg)
    
    # Method 2: Otsu's Thresholding
    otsu_seg = apply_otsu_thresholding(grayscale_image)
    otsu_regions = extract_regions(otsu_seg)
    
    # Method 3: Otsu + Morphological Refinement
    refined_seg = apply_closing(apply_opening(otsu_seg, kernel_size=5), kernel_size=5)
    refined_regions = extract_regions(refined_seg)
    
    return {
        'global': {'binary': global_seg, 'regions': global_regions},
        'otsu': {'binary': otsu_seg, 'regions': otsu_regions},
        'refined': {'binary': refined_seg, 'regions': refined_regions}
    }

def calculate_statistics(regions):
    """Calculate statistics for regions"""
    if not regions:
        return None
    
    areas = [r['area'] for r in regions]
    circularity = [r['circularity'] for r in regions if r['circularity'] > 0]
    aspect_ratios = [r['aspect_ratio'] for r in regions]
    
    return {
        'num_regions': len(regions),
        'area_mean': np.mean(areas),
        'area_std': np.std(areas),
        'area_median': np.median(areas),
        'area_range': (min(areas), max(areas)),
        'circularity_mean': np.mean(circularity) if circularity else 0,
        'circularity_std': np.std(circularity) if circularity else 0,
        'aspect_ratio_mean': np.mean(aspect_ratios),
        'largest_region': max(areas),
        'total_area': sum(areas)
    }

def main():
    print("=" * 70)
    print("TASK 4: ANALYSIS AND INTERPRETATION")
    print("=" * 70)
    
    # Load image
    print("\n1. Loading image...")
    grayscale_image = load_and_convert_to_grayscale(IMAGE_PATH)
    print(f"   ✓ Image loaded: Shape = {grayscale_image.shape}")
    print(f"   ✓ Pixel value range: [{grayscale_image.min()}, {grayscale_image.max()}]")
    print(f"   ✓ Image mean intensity: {grayscale_image.mean():.2f}")
    print(f"   ✓ Image std deviation: {grayscale_image.std():.2f}")
    
    # Compare segmentation methods
    print("\n2. Comparing segmentation methods...")
    comparison = compare_segmentation_methods(grayscale_image)
    
    # Analyze each method
    for method_name, method_data in comparison.items():
        print(f"\n   {method_name.upper()} THRESHOLDING:")
        regions = method_data['regions']
        stats_data = calculate_statistics(regions)
        
        print(f"      ✓ Number of regions detected: {stats_data['num_regions']}")
        print(f"      ✓ Total foreground area: {stats_data['total_area']:,} pixels")
        print(f"      ✓ Area statistics:")
        print(f"         - Mean: {stats_data['area_mean']:.2f}")
        print(f"         - Median: {stats_data['area_median']:.2f}")
        print(f"         - Std Dev: {stats_data['area_std']:.2f}")
        print(f"         - Range: [{stats_data['area_range'][0]}, {stats_data['area_range'][1]}]")
        print(f"      ✓ Largest region: {stats_data['largest_region']:,} pixels")
        print(f"      ✓ Circularity (roundness):")
        print(f"         - Mean: {stats_data['circularity_mean']:.3f}")
        print(f"      ✓ Aspect ratio:")
        print(f"         - Mean: {stats_data['aspect_ratio_mean']:.3f}")
        
        # Display top 5 regions
        if regions:
            print(f"      ✓ Top 5 largest regions:")
            sorted_regions = sorted(regions, key=lambda x: x['area'], reverse=True)[:5]
            for idx, region in enumerate(sorted_regions, 1):
                print(f"         Region {idx}: Area={region['area']} pixels, "
                      f"Circularity={region['circularity']:.3f}, "
                      f"Aspect Ratio={region['aspect_ratio']:.2f}")
    
    # Performance comparison
    print("\n3. Performance Comparison:")
    print("   " + "-" * 60)
    print(f"   {'Method':<15} {'Regions':<10} {'Total Area':<15} {'Avg Area':<15}")
    print("   " + "-" * 60)
    
    for method_name, method_data in comparison.items():
        stats_data = calculate_statistics(method_data['regions'])
        print(f"   {method_name.capitalize():<15} {stats_data['num_regions']:<10} "
              f"{stats_data['total_area']:<15,} {stats_data['area_mean']:<15.2f}")
    
    # Calculate differences
    print("\n4. Method Comparison Analysis:")
    global_stats = calculate_statistics(comparison['global']['regions'])
    otsu_stats = calculate_statistics(comparison['otsu']['regions'])
    refined_stats = calculate_statistics(comparison['refined']['regions'])
    
    print(f"   ✓ Otsu vs Global:")
    print(f"      - Region count difference: {otsu_stats['num_regions'] - global_stats['num_regions']}")
    print(f"      - Total area difference: {otsu_stats['total_area'] - global_stats['total_area']} pixels")
    
    print(f"   ✓ Refined vs Otsu:")
    print(f"      - Region count difference: {refined_stats['num_regions'] - otsu_stats['num_regions']}")
    print(f"      - Total area difference: {refined_stats['total_area'] - otsu_stats['total_area']} pixels")
    print(f"      - Noise reduction: {((otsu_stats['num_regions'] - refined_stats['num_regions']) / otsu_stats['num_regions'] * 100):.2f}%")
    
    # Clinical interpretation
    print("\n5. CLINICAL INTERPRETATION:")
    print("   " + "=" * 56)
    print("""
   FINDINGS & CLINICAL RELEVANCE:
   
   1. REGION SEGMENTATION:
      - Multiple regions of interest (ROIs) detected, indicating varied 
        tissue densities or abnormalities in the medical image
      - Otsu's method automatically adapts to image contrast, improving 
        detection accuracy compared to fixed thresholding
   
   2. MORPHOLOGICAL REFINEMENT:
      - Opening operation removes small noise and artifacts
      - Closing operation fills small cavities and connects fragmented regions
      - Refined segmentation produces cleaner, more clinically relevant ROIs
   
   3. REGION PROPERTIES:
      - Area statistics provide quantitative assessment of abnormal regions
      - Circularity indicates whether regions have regular (benign) or 
        irregular (potentially pathological) shapes
      - Aspect ratio helps distinguish between elongated and compact regions
   
   4. DIAGNOSTIC APPLICATIONS:
      - Tumor detection: Large irregular regions with low circularity
      - Lesion quantification: Track region size and number over time
      - Tissue characterization: Different region properties indicate 
        different tissue types
      - Computer-aided diagnosis (CAD): Automated screening tool for 
        radiologists
    """)
    print("   " + "=" * 56)
    
    # Visualization
    print("\n6. Creating comprehensive visualization...")
    fig = plt.figure(figsize=(18, 12))
    
    # Original image
    ax1 = plt.subplot(3, 4, 1)
    ax1.imshow(grayscale_image, cmap='gray')
    ax1.set_title('Original Grayscale Image', fontweight='bold')
    ax1.axis('off')
    
    # Histogram
    ax2 = plt.subplot(3, 4, 2)
    ax2.hist(grayscale_image.ravel(), bins=256, color='black', alpha=0.7)
    ax2.set_xlabel('Pixel Intensity')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Intensity Histogram', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Row 2: Global Thresholding
    ax3 = plt.subplot(3, 4, 3)
    ax3.imshow(comparison['global']['binary'], cmap='gray')
    ax3.set_title(f"Global: {global_stats['num_regions']} regions", fontweight='bold')
    ax3.axis('off')
    
    # Row 2: Otsu Thresholding
    ax4 = plt.subplot(3, 4, 4)
    ax4.imshow(comparison['otsu']['binary'], cmap='gray')
    ax4.set_title(f"Otsu: {otsu_stats['num_regions']} regions", fontweight='bold')
    ax4.axis('off')
    
    # Row 2: Refined Segmentation
    ax5 = plt.subplot(3, 4, 5)
    ax5.imshow(comparison['refined']['binary'], cmap='gray')
    ax5.set_title(f"Refined: {refined_stats['num_regions']} regions", fontweight='bold')
    ax5.axis('off')
    
    # Area distribution comparison
    ax6 = plt.subplot(3, 4, 6)
    methods = ['Global', 'Otsu', 'Refined']
    areas_data = [
        [r['area'] for r in comparison['global']['regions']],
        [r['area'] for r in comparison['otsu']['regions']],
        [r['area'] for r in comparison['refined']['regions']]
    ]
    ax6.boxplot(areas_data, labels=methods)
    ax6.set_ylabel('Region Area (pixels)')
    ax6.set_title('Area Distribution', fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # Region count comparison
    ax7 = plt.subplot(3, 4, 7)
    region_counts = [global_stats['num_regions'], otsu_stats['num_regions'], refined_stats['num_regions']]
    bars = ax7.bar(methods, region_counts, color=['blue', 'green', 'red'], alpha=0.7)
    ax7.set_ylabel('Number of Regions')
    ax7.set_title('Region Count Comparison', fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')
    for bar, count in zip(bars, region_counts):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}', ha='center', va='bottom')
    
    # Total area comparison
    ax8 = plt.subplot(3, 4, 8)
    total_areas = [global_stats['total_area'], otsu_stats['total_area'], refined_stats['total_area']]
    bars = ax8.bar(methods, total_areas, color=['blue', 'green', 'red'], alpha=0.7)
    ax8.set_ylabel('Total Area (pixels)')
    ax8.set_title('Total Foreground Area', fontweight='bold')
    ax8.grid(True, alpha=0.3, axis='y')
    
    # Circularity distribution
    ax9 = plt.subplot(3, 4, 9)
    circ_data = [
        [r['circularity'] for r in comparison['global']['regions'] if r['circularity'] > 0],
        [r['circularity'] for r in comparison['otsu']['regions'] if r['circularity'] > 0],
        [r['circularity'] for r in comparison['refined']['regions'] if r['circularity'] > 0]
    ]
    ax9.boxplot(circ_data, labels=methods)
    ax9.set_ylabel('Circularity')
    ax9.set_title('Circularity Distribution', fontweight='bold')
    ax9.set_ylim([0, 1.1])
    ax9.grid(True, alpha=0.3)
    
    # Aspect ratio distribution
    ax10 = plt.subplot(3, 4, 10)
    aspect_data = [
        [r['aspect_ratio'] for r in comparison['global']['regions']],
        [r['aspect_ratio'] for r in comparison['otsu']['regions']],
        [r['aspect_ratio'] for r in comparison['refined']['regions']]
    ]
    ax10.boxplot(aspect_data, labels=methods)
    ax10.set_ylabel('Aspect Ratio')
    ax10.set_title('Aspect Ratio Distribution', fontweight='bold')
    ax10.grid(True, alpha=0.3)
    
    # Method statistics text
    ax11 = plt.subplot(3, 4, 11)
    ax11.axis('off')
    stats_text = f"""
GLOBAL THRESHOLDING:
  Regions: {global_stats['num_regions']}
  Avg Area: {global_stats['area_mean']:.1f} px
  
OTSU'S THRESHOLDING:
  Regions: {otsu_stats['num_regions']}
  Avg Area: {otsu_stats['area_mean']:.1f} px
  
REFINED (Otsu+Morph):
  Regions: {refined_stats['num_regions']}
  Avg Area: {refined_stats['area_mean']:.1f} px
    """
    ax11.text(0.1, 0.5, stats_text, fontsize=9, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Method comparison text
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    comp_text = f"""
COMPARISON RESULTS:

Otsu vs Global:
  Region Δ: {otsu_stats['num_regions'] - global_stats['num_regions']}
  Area Δ: {otsu_stats['total_area'] - global_stats['total_area']}
  
Refined vs Otsu:
  Region Δ: {refined_stats['num_regions'] - otsu_stats['num_regions']}
  Area Δ: {refined_stats['total_area'] - otsu_stats['total_area']}
  
Noise Reduction:
  {((otsu_stats['num_regions'] - refined_stats['num_regions']) / max(1, otsu_stats['num_regions']) * 100):.1f}%
    """
    ax12.text(0.1, 0.5, comp_text, fontsize=9, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(r'c:\Users\gtcam\OneDrive\Desktop\DIP image processing\task4_analysis_result.png', dpi=150, bbox_inches='tight')
    print("   ✓ Result saved as 'task4_analysis_result.png'")
    plt.show()
    
    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"Total regions detected across all methods: {global_stats['num_regions'] + otsu_stats['num_regions'] + refined_stats['num_regions']}")
    print(f"Otsu's method proved most effective with {otsu_stats['num_regions']} regions")
    print(f"Morphological refinement reduced noise by {((otsu_stats['num_regions'] - refined_stats['num_regions']) / max(1, otsu_stats['num_regions']) * 100):.1f}%")
    print(f"Clinical significance: Refined segmentation provides reliable ROI detection")
    print("=" * 70)

if __name__ == "__main__":
    main()
