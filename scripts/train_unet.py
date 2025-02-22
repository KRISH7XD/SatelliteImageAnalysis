import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import Dataset, DataLoader
from models.unet import UNet
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import rasterio
import torch.nn as nn


class DynamicWorldDataset(Dataset):
    def __init__(self, img_dir=None, mask_dir=None):
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.img_dir = img_dir if img_dir else os.path.join(self.root_dir, "data", "sentinel2")
        self.mask_dir = mask_dir if mask_dir else os.path.join(self.root_dir, "data", "masks")
        
        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        if not os.path.exists(self.mask_dir):
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")
        
        self.img_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.tif')])
        self.mask_files = sorted([f for f in os.listdir(self.mask_dir) if f.endswith('.tif')])
        

        if not self.img_files:
            raise FileNotFoundError(f"No .tif files found in {self.img_dir}")
        if not self.mask_files:
            raise FileNotFoundError(f"No .tif files found in {self.mask_dir}")
        
        self.transform = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        
  
        with rasterio.open(img_path) as src:
            img = src.read([1, 2, 3])  
            img = np.moveaxis(img, 0, -1)
            img = (img / img.max() * 255).astype(np.uint8) 
        

        with rasterio.open(mask_path) as src:
            mask = src.read(1)  
        
        transformed = self.transform(image=img, mask=mask)
        return transformed["image"], transformed["mask"].long()


dataset = DynamicWorldDataset()
loader = DataLoader(dataset, batch_size=4, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, out_channels=9).to(device)  
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


for epoch in range(5):
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), '..', "models", "unet_model.pth"))