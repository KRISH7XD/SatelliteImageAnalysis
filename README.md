# Satellite Image Analysis

A deep learning project for land use classification and change detection using satellite imagery, built with PyTorch and Streamlit.

## Features
- **Land Use Classification**: Classify EuroSAT images using Convolutional Neural Networks (CNN) and Vision Transformers (ViT).
- **Image Segmentation**: Segment Sentinel-2 images with U-Net trained on Dynamic World labels.
- **Change Detection**: Detect land use changes between two Sentinel-2 images.

## Project Structure
```
SatelliteImageAnalysis/
├── app.py              # Streamlit app
├── models/            # Model definitions
│   ├── cnn.py
│   ├── unet.py
│   ├── vit.py
├── scripts/           # Training and preprocessing scripts
│   ├── train_cnn.py
│   ├── train_vit.py
│   ├── train_unet.py
│   ├── preprocess.py
├── data/              # (Excluded) Datasets
├── requirements.txt   # Dependencies
├── README.md          # This file
```

## Setup
### 1. Clone the Repository
```bash
git clone https://github.com/your-username/SatelliteImageAnalysis.git
cd SatelliteImageAnalysis
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Data
- **EuroSAT**: Download `EuroSAT_RGB.zip` from GitHub and extract it to `data/eurosat/2750/`.
- **Sentinel-2**: Export from Google Earth Engine (see GEE script below) to `data/sentinel2/` and `data/masks/`.

### 4. Train Models
- **CNN**: `python scripts/train_cnn.py` (requires EuroSAT dataset)
- **ViT**: `python scripts/train_vit.py` (requires EuroSAT dataset)
- **U-Net**: `python scripts/train_unet.py` (requires Sentinel-2 + Dynamic World dataset)

### 5. Run the Application
```bash
streamlit run app.py --server.maxUploadSize 1000
```

## Google Earth Engine (GEE) Script for Sentinel-2 Data
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
// Repeat for s2_2024, mask_2023, mask_2024 with adjusted dates
```

## Requirements
- **Python** 3.11+
- **CUDA-enabled GPU** (optional, e.g., NVIDIA RTX 3050)

## Pretrained Models and Data
- Download pretrained weights and sample data: [Google Drive Link](https://drive.google.com/your-link)
- Place `.pth` files in `models/` and `.tif` files in `data/sentinel2/` and `data/masks/`.

## License
This project is licensed under the [MIT License](LICENSE).

