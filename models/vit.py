from transformers import ViTForImageClassification
import torch.nn as nn

def get_vit_model(num_classes=10):
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model