# convert_to_onnx.py
import json

import torch
from torch import nn
from torchvision import models


# 1. Воссоздаем архитектуру модели
def create_model(num_classes):
    model = models.efficientnet_b0(weights=None)  # Используем weights вместо pretrained
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

# 2. Загружаем веса
num_classes = 4  # Изменено с 3 на 4, так как в весах 4 класса
model = create_model(num_classes)

# Загружаем веса с обработкой ошибок
try:
    checkpoint = torch.load('best_grape_model.pth', map_location='cpu')
    model.load_state_dict(checkpoint)
    print("Модель загружена успешно")
except RuntimeError as e:
    print(f"Ошибка загрузки модели: {e}")
    print("Проверьте, что файл best_grape_model.pth существует и количество классов совпадает")
    exit(1)

model.eval()

# 3. Создаем пример входных данных
dummy_input = torch.randn(1, 3, 224, 224)  # (batch, channels, height, width)

# 4. Экспортируем в ONNX
try:
    torch.onnx.export(
        model,
        dummy_input,
        "grape_model.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},  # динамический batch size
            'output': {0: 'batch_size'}
        }
    )
    print("Модель сконвертирована в grape_model.onnx")
except Exception as e:
    print(f"Ошибка конвертации: {e}")
    exit(1)

# 5. Сохраняем названия классов (замените на реальные названия ваших сортов винограда)
class_names = ['class1', 'class2', 'class3', 'class4']  # 4 класса для 4 сортов
with open('class_names.json', 'w') as f:
    json.dump(class_names, f)
    
print("Сохранены названия классов")