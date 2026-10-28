import numpy as np
import onnxruntime as ort
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

print("Загружаем ONNX модель...")
session = ort.InferenceSession('grape_model.onnx')
input_name = session.get_inputs()[0].name

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(
    'data',
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
print("Считаем метрики...\n")

all_preds  = []
all_labels = []

for images, labels in test_loader:
    outputs = session.run(None, {input_name: images.numpy()})
    predicted = np.argmax(outputs[0], axis=1)
    all_preds.extend(predicted)
    all_labels.extend(labels.numpy())

print("=" * 60)
print("CLASSIFICATION REPORT (ONNX)")
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