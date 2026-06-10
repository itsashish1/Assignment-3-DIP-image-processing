# Assignment 3: Digital Image Processing (DIP) - Image Processing Pipeline

A comprehensive image processing project implementing key DIP concepts including image compression, segmentation, morphological processing, and analysis.

## Project Overview

This project demonstrates a complete image processing pipeline for medical image analysis using classical Digital Image Processing techniques. It processes grayscale images through multiple stages of analysis and transformation.

## Tasks

### Task 1: Image Compression using RLE
**File:** `task1_image_compression.py`

Implements Run Length Encoding (RLE) for lossless image compression.

**Features:**
- Load grayscale medical images
- Implement Run Length Encoding algorithm
- Calculate compression ratio and efficiency
- Decode compressed data back to original format

**Key Functions:**
- `load_and_convert_to_grayscale()` - Load image and convert to grayscale
- `run_length_encode()` - Encode image data using RLE
- `run_length_decode()` - Decode RLE data back to original
- `calculate_compression_ratio()` - Compute compression efficiency

**Output:** Compression statistics and visualizations

---

### Task 2: Image Segmentation
**File:** `task2_image_segmentation.py`

Apply thresholding techniques to identify regions of interest in images.

**Features:**
- Global thresholding with fixed threshold values
- Otsu's automatic thresholding for optimal threshold calculation
- Connected component labeling to identify separate regions
- Region properties extraction and filtering

**Key Functions:**
- `apply_global_thresholding()` - Fixed threshold segmentation
- `apply_otsu_thresholding()` - Automatic threshold segmentation
- `identify_regions_of_interest()` - Extract separate regions
- Region size and property analysis

**Output:** Binary segmented images and region visualizations

---

### Task 3: Morphological Processing
**File:** `task3_morphological_processing.py`

Apply morphological operations to refine segmented regions.

**Features:**
- Erosion operation to reduce region size
- Dilation operation to expand regions
- Opening operation (erosion followed by dilation)
- Closing operation (dilation followed by erosion)
- Configurable kernel types and sizes

**Key Functions:**
- `apply_erosion()` - Reduce region boundaries
- `apply_dilation()` - Expand region boundaries
- `apply_opening()` - Remove small objects
- `apply_closing()` - Fill small holes
- `create_kernel()` - Create structuring elements (rect, ellipse, cross)

**Output:** Refined binary images with processed regions

---

### Task 4: Analysis and Interpretation
**File:** `task4_analysis_interpretation.py`

Compare segmentation results and analyze clinical relevance.

**Features:**
- Compare multiple thresholding methods
- Extract region properties and statistics
- Perform morphological operations
- Clinical interpretation of results

**Key Functions:**
- `apply_global_thresholding()` - Global threshold method
- `apply_otsu_thresholding()` - Otsu's method
- `apply_opening()` and `apply_closing()` - Refinement operations
- `extract_regions()` - Connected component analysis
- `analyze_regions()` - Extract statistical properties

**Output:** Comparative analysis and clinical insights

---

## Requirements

### Dependencies
- Python 3.7+
- OpenCV (`cv2`) - Computer vision library
- NumPy (`numpy`) - Numerical computing
- Matplotlib (`matplotlib`) - Data visualization
- SciPy (`scipy`) - Scientific computing

### Installation

Create a virtual environment and install dependencies:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install opencv-python numpy matplotlib scipy
```

Or install all at once:

```bash
pip install -r requirements.txt
```

## Usage

### Running Individual Tasks

```bash
# Task 1: Image Compression
python task1_image_compression.py

# Task 2: Image Segmentation
python task2_image_segmentation.py

# Task 3: Morphological Processing
python task3_morphological_processing.py

# Task 4: Analysis and Interpretation
python task4_analysis_interpretation.py
```

### Image Input

Each script uses an image path (currently configured for medical images):
```python
IMAGE_PATH = r"C:\Users\gtcam\OneDrive\Pictures\Camera Roll\OIP (2).webp"
```

Update this path in each file to process your own images.

## Algorithm Details

### Run Length Encoding (Task 1)
- Encodes consecutive identical pixel values
- Format: (value, count) tuples
- Effective for images with large uniform regions

### Thresholding (Tasks 2 & 4)
- **Global Thresholding:** Fixed threshold value (default: 127)
- **Otsu's Method:** Automatic optimal threshold calculation using histogram analysis

### Morphological Operations (Tasks 3 & 4)
- **Erosion:** Reduces white regions, removes small objects
- **Dilation:** Expands white regions, fills small holes
- **Opening:** Erosion → Dilation (removes small objects, preserves structure)
- **Closing:** Dilation → Erosion (fills small holes, preserves structure)

### Connected Component Labeling
- Identifies separate regions in binary images
- Extracts bounding boxes and properties
- Filters regions by size

## Project Structure

```
DIP-image-processing/
├── task1_image_compression.py
├── task2_image_segmentation.py
├── task3_morphological_processing.py
├── task4_analysis_interpretation.py
├── requirements.txt
└── README.md
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Grayscale Conversion** | Convert RGB images to single channel (0-255 intensity) |
| **Binary Image** | Image with only 2 values (0 and 255) after thresholding |
| **Structuring Element** | Kernel used in morphological operations |
| **Connected Components** | Groups of adjacent pixels with same intensity |
| **Region of Interest (ROI)** | Significant areas extracted during segmentation |
| **Compression Ratio** | (Original Size / Compressed Size) × 100 |

## Output Examples

### Task 1: Compression Statistics
- Original file size
- Compressed data size
- Compression ratio percentage
- Visual comparison

### Task 2: Segmentation Results
- Global threshold binary image
- Otsu threshold binary image
- Identified regions visualization
- Region count and properties

### Task 3: Morphological Processing
- Original binary image
- Eroded image
- Dilated image
- Opened and closed images

### Task 4: Analysis Comparison
- Side-by-side comparison of methods
- Region statistics
- Clinical interpretation

## Clinical Relevance

These techniques are essential in medical image analysis for:
- **Detection:** Identifying abnormalities or regions of interest
- **Quantification:** Measuring size, shape, and properties
- **Segmentation:** Separating anatomical structures
- **Classification:** Categorizing regions (e.g., healthy vs. pathological)

## Troubleshooting

### Image Not Found
Ensure the image path exists and is correctly specified in each script.

### Module Import Errors
Verify all dependencies are installed:
```bash
pip install opencv-python numpy matplotlib scipy
```

### Memory Issues
For very large images, consider resizing before processing:
```python
image = cv2.resize(image, (width, height))
```

## Performance Considerations

- **Kernel Size:** Larger kernels → more processing time
- **Image Size:** Higher resolution → more computation
- **Iterations:** More iterations → better refinement but slower

## Future Enhancements

- [ ] Add image preprocessing (filtering, normalization)
- [ ] Implement advanced segmentation (watershed, graph cuts)
- [ ] Add machine learning classification
- [ ] create GUI for interactive processing
- [ ] Support for 3D medical images (CT, MRI)
- [ ] Performance optimization for real-time processing

## References

- OpenCV Documentation: https://docs.opencv.org/
- Digital Image Processing (Gonzalez & Woods)
- NumPy Documentation: https://numpy.org/
- SciPy Documentation: https://scipy.org/

## License

This project is open source and available under the MIT License.

## Author

Created for Digital Image Processing coursework/project.

---

**Last Updated:** April 2026

For questions or issues, please create an issue in the repository.
