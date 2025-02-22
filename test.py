import rasterio
import numpy as np

with rasterio.open("data/sentinel2/s2_2024.tif") as src:
    print(f"Shape: {src.shape}, Bands: {src.count}")  
    rgb = src.read()
    print(f"RGB range: {rgb.min()} to {rgb.max()}")

# Dynamic World Mask
with rasterio.open("data/masks/mask_2024.tif") as src:
    print(f"Shape: {src.shape}, Bands: {src.count}") 
    mask = src.read(1)
    print(f"Mask values: {np.unique(mask)}")  