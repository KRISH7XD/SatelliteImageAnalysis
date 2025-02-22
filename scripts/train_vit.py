import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import ViTForImageClassification, ViTImageProcessor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import torch.nn as nn

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.current_device()} - {torch.cuda.get_device_name(0)}")
else:
    print("Running on CPU - CUDA not detected!")

# Data
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(root_dir, "data", "eurosat", "2750")

if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Dataset directory not found: {data_dir}. Please download EuroSAT to {data_dir}")

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
dataset = datasets.ImageFolder(data_dir, transform=transform)
loader = DataLoader(dataset, batch_size=16, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224", do_rescale=False)
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
model.classifier = nn.Linear(model.classifier.in_features, 10)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()


for epoch in range(3):
    for i, (images, labels) in enumerate(loader):
        inputs = processor(images=images, return_tensors="pt", do_rescale=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)
        outputs = model(**inputs).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % 100 == 0:  
            print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item()}, Device: {next(model.parameters()).device}")
    print(f"Epoch {epoch+1} completed, Loss: {loss.item()}")

torch.save(model.state_dict(), os.path.join(root_dir, "models", "vit_model.pth"))