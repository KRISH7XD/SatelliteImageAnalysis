import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add project root to path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.cnn import SimpleCNN
import torch.nn as nn


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # Project root
data_dir = os.path.join(root_dir, "data", "eurosat", "2750")


if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Dataset directory not found: {data_dir}. Please download EuroSAT to {data_dir}")

transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
dataset = datasets.ImageFolder(data_dir, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


for epoch in range(5):
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")


torch.save(model.state_dict(), os.path.join(root_dir, "models", "cnn_model.pth"))