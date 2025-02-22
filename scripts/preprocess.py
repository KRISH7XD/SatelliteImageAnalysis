import rasterio
from PIL import Image
import numpy as np

def load_sentinel2(path):
    """Load Sentinel-2 GeoTIFF and return RGB as PIL Image."""
    with rasterio.open(path) as src:
        r = src.read(4)  
        g = src.read(3)  
        b = src.read(2) 
        rgb = np.stack([r, g, b], axis=-1)
        rgb = (rgb / rgb.max() * 255).astype(np.uint8) 
    return Image.fromarray(rgb)

def load_mask(path):
    """Load segmentation mask as NumPy array."""
    with rasterio.open(path) as src:
        mask = src.read(1)  
    return mask


if __name__ == "__main__":
    img = load_sentinel2("../data/sentinel2/s2_2023.tif")
    img.save("test_s2_2023.jpg")