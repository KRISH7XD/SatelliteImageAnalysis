# Satellite Image Analysis

A deep learning project for land use classification and change detection using satellite imagery, built with PyTorch and Streamlit. This project leverages CNNs and Vision Transformers (ViT) for classifying EuroSAT images and U-Net for segmenting Sentinel-2 imagery with Dynamic World labels, supporting tasks like deforestation monitoring and urban expansion analysis—aligned with ISRO-like earth observation goals.

## Features
- **Classification**: Classify EuroSAT RGB images into 10 land use types (e.g., Forest, Urban) using a Simple CNN and Vision Transformers (ViT).
- **Segmentation**: Segment Sentinel-2 imagery into 9 land cover classes (e.g., Water, Trees) using U-Net with Dynamic World labels.
- **Change Detection**: Compare two Sentinel-2 images to detect land use changes, visualizing differences and statistics.

## Project Structure
```
SatelliteImageAnalysis/
├── app.py              # Streamlit app for interactive analysis
├── models/            # Model definitions
│   ├── cnn.py         # Simple CNN for classification
│   ├── unet.py        # U-Net for segmentation
│   ├── vit.py         # Vision Transformer setup
├── scripts/           # Training and utility scripts
│   ├── train_cnn.py   # Train CNN on EuroSAT
│   ├── train_vit.py   # Train ViT on EuroSAT
│   ├── train_unet.py  # Train U-Net on Sentinel-2
│   ├── preprocess.py  # Utility for data preprocessing
├── requirements.txt   # Python dependencies
├── README.md          # Project documentation
├── .gitignore         # Excludes large files (e.g., .pth, data)
```

## Setup Instructions
### 1. Clone the Repository
```bash
git clone https://github.com/KRISH7XD/SatelliteImageAnalysis.git
cd SatelliteImageAnalysis
```

### 2. Install Dependencies
Ensure Python 3.11+ and a CUDA-enabled GPU (e.g., NVIDIA RTX 3050) for optimal performance.
```bash
pip install -r requirements.txt
```

### 3. Prepare Data
- **EuroSAT**: Download `EuroSAT_RGB.zip` from EuroSAT GitHub and extract it to `data/eurosat/2750/`. Contains 27,000 RGB images in 10 classes.
- **Sentinel-2**: Export Sentinel-2 RGB images and Dynamic World masks from Google Earth Engine (GEE). Requires a GEE account. Use the script below.

### 4. Train Models (or use pretrained weights)
- **CNN**: `python scripts/train_cnn.py` (requires EuroSAT dataset)
- **ViT**: `python scripts/train_vit.py` (requires EuroSAT dataset)
- **U-Net**: `python scripts/train_unet.py` (requires Sentinel-2 + Dynamic World dataset)
- **Pretrained Models**: - Download from [Google Drive](https://drive.google.com/drive/folders/1zw_rt1pNaZr88nadJNLg4Wvun5ceXjvr?usp=drive_link) and place in `models/`.

### 5. Run the Application
Supports large Sentinel-2 TIFFs (~833MB):
```bash
streamlit run app.py --server.maxUploadSize 1000
```
Open http://localhost:8501 in your browser.

## Google Earth Engine (GEE) Script for Sentinel-2 Data
Export a small region (e.g., 0.1°×0.1° near Delhi) to keep files <200MB:
```javascript
var geometry = ee.Geometry.Rectangle([77.5, 28.5, 77.6, 28.6]); 
var s2Col = ee.ImageCollection('COPERNICUS/S2')
  .filterBounds(geometry)
  .filterDate('2023-01-01', '2023-12-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
  .select(['B4', 'B3', 'B2'])
  .median();

Export.image.toDrive({
  image: s2Col,
  description: 's2_2023_small',
  folder: 'SatelliteImageAnalysis',
  scale: 10,
  region: geometry,
  fileFormat: 'GeoTIFF',
  maxPixels: 1e9
});
// Repeat for 's2_2024_small' (2024-01-01 to 2025-02-21), 'mask_2023_small', 'mask_2024_small' (Dynamic World labels)
```

## Pretrained Models and Sample Data
Download pretrained weights (`.pth`) and sample Sentinel-2 TIFFs from Google Drive Link (replace with your link).
Place `.pth` files in `models/` and `.tif` files in `data/sentinel2/` and `data/masks/`.

## Usage
- **Land Use Analysis**: Upload a EuroSAT `.jpg` for CNN/ViT classification or a Sentinel-2 `.tif` for U-Net segmentation.
- **Change Detection**: Upload two Sentinel-2 `.tif` files (e.g., `s2_2023.tif`, `s2_2024.tif`) to detect changes.

## Requirements
- **Python** 3.11+
- **PyTorch with CUDA** (e.g., `torch==2.2.0+cu121`)
- See `requirements.txt` for full list.

## License
This project is licensed under the [MIT License](LICENSE). © 2025 KRISH7XD

## Acknowledgments
Data sources: EuroSAT, Sentinel-2, Dynamic World.

