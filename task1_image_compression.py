"""
Task 1: Image Compression using Run Length Encoding (RLE)
Load a grayscale medical image, implement RLE, and calculate compression ratio
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Image path
IMAGE_PATH = r"C:\Users\gtcam\OneDrive\Pictures\Camera Roll\OIP (2).webp"

def load_and_convert_to_grayscale(image_path):
    """Load image and convert to grayscale"""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image

def run_length_encode(data):
    """
    Implement Run Length Encoding (RLE)
    Input: flattened array of pixel values
    Output: list of (value, count) tuples
    """
    if len(data) == 0:
        return []
    
    encoded = []
    current_value = data[0]
    count = 1
    
    for i in range(1, len(data)):
        if data[i] == current_value and count < 255:  # Limit count to 255
            count += 1
        else:
            encoded.append((current_value, count))
            current_value = data[i]
            count = 1
    
    # Don't forget the last run
    encoded.append((current_value, count))
    
    return encoded

def run_length_decode(encoded_data):
    """Decode RLE data back to original format"""
    decoded = []
    for value, count in encoded_data:
        decoded.extend([value] * count)
    return np.array(decoded, dtype=np.uint8)

def calculate_compression_metrics(original_data, encoded_data):
    """Calculate compression ratio and storage savings"""
    # Original size (in bytes)
    original_size = original_data.nbytes
    
    # Encoded size (each tuple = 2 bytes approximately)
    # Each (value, count) pair: 1 byte for value + 1 byte for count
    encoded_size = len(encoded_data) * 2
    
    compression_ratio = (1 - encoded_size / original_size) * 100
    storage_savings = original_size - encoded_size
    
    return {
        'original_size': original_size,
        'encoded_size': encoded_size,
        'compression_ratio': compression_ratio,
        'storage_savings': storage_savings,
        'compression_factor': original_size / encoded_size if encoded_size > 0 else 0
    }

def main():
    print("=" * 60)
    print("TASK 1: IMAGE COMPRESSION USING RLE")
    print("=" * 60)
    
    # Load grayscale image
    print("\n1. Loading and converting image to grayscale...")
    grayscale_image = load_and_convert_to_grayscale(IMAGE_PATH)
    print(f"   ✓ Image loaded: Shape = {grayscale_image.shape}")
    print(f"   ✓ Data type = {grayscale_image.dtype}")
    
    # Flatten the image
    flat_image = grayscale_image.flatten()
    print(f"   ✓ Flattened image size = {len(flat_image)} pixels")
    
    # Apply RLE
    print("\n2. Applying Run Length Encoding...")
    encoded_data = run_length_encode(flat_image)
    print(f"   ✓ Encoded data points = {len(encoded_data)}")
    print(f"   ✓ First 10 encoded pairs: {encoded_data[:10]}")
    
    # Calculate compression metrics
    print("\n3. Calculating compression metrics...")
    metrics = calculate_compression_metrics(flat_image, encoded_data)
    
    print(f"\n   Original Size:        {metrics['original_size']:,} bytes")
    print(f"   Encoded Size:         {metrics['encoded_size']:,} bytes")
    print(f"   Compression Ratio:    {metrics['compression_ratio']:.2f}%")
    print(f"   Storage Savings:      {metrics['storage_savings']:,} bytes")
    print(f"   Compression Factor:   {metrics['compression_factor']:.2f}x")
    
    # Verify decompression
    print("\n4. Verifying decompression...")
    decoded_data = run_length_decode(encoded_data)
    decoded_image = decoded_data.reshape(grayscale_image.shape)
    
    mse = np.mean((grayscale_image - decoded_image) ** 2)
    print(f"   ✓ Mean Squared Error: {mse}")
    
    if mse == 0:
        print("   ✓ Perfect reconstruction achieved!")
    else:
        print("   ⚠ Reconstruction error detected")
    
    # Visualization
    print("\n5. Creating visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(grayscale_image, cmap='gray')
    axes[0].set_title('Original Grayscale Image')
    axes[0].axis('off')
    
    axes[1].imshow(decoded_image, cmap='gray')
    axes[1].set_title('RLE Decompressed Image')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(r'c:\Users\gtcam\OneDrive\Desktop\DIP image processing\task1_compression_result.png', dpi=150, bbox_inches='tight')
    print("   ✓ Result saved as 'task1_compression_result.png'")
    plt.show()
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("COMPRESSION SUMMARY")
    print("=" * 60)
    print(f"RLE achieved {metrics['compression_ratio']:.2f}% compression")
    print(f"Space saved: {metrics['storage_savings'] / 1024:.2f} KB")
    print("=" * 60)

if __name__ == "__main__":
    main()
