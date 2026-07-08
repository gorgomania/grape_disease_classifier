import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Устройство: {device}")
 
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
 
test_dataset = datasets.ImageFolder(
    os.path.join('test', 'test'),
    transform=transform
)
test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    num_workers=0
)
 
class_names = test_dataset.classes
print(f"Классы: {class_names}")
print(f"Тестовых изображений: {len(test_dataset)}")
 
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
model.load_state_dict(torch.load('best_grape_model.pth', map_location=device))
model = model.to(device)
model.eval()
print("Модель загружена, считаем метрики...\n")
 
all_preds  = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
 
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)
cm = confusion_matrix(all_labels, all_preds)
print(f"{'':15}", end="")
for name in class_names:
    print(f"{name:15}", end="")
print()
for i, row in enumerate(cm):
    print(f"{class_names[i]:15}", end="")
    for val in row:
        print(f"{val:<15}", end="")
    print()
 
accuracy = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
print(f"\nОбщая точность: {accuracy:.2f}%")