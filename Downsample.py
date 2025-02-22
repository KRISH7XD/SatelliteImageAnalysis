import rasterio
from rasterio.enums import Resampling
import numpy as np

with rasterio.open("data/sentinel2/s2_2023.tif") as src:
    scale_factor = 0.33  
    new_height = int(src.height * scale_factor)
    new_width = int(src.width * scale_factor)
    data = src.read(out_shape=(src.count, new_height, new_width), resampling=Resampling.bilinear)
    transform = src.transform * src.transform.scale(
        (src.width / new_width), (src.height / new_height)
    )
    profile = src.profile
    profile.update(width=new_width, height=new_height, transform=transform)
    with rasterio.open("data/sentinel2/s2_2023_small.tif", "w", **profile) as dst:
        dst.write(data)

