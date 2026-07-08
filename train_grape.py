import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import gc  # для очистки памяти

if __name__ == "__main__":

    print("Начинаем обучение модели...")
    print(f"Процессор: Intel Core i5-7400")
    print(f"Видеокарта: GTX 1060 3GB")

    # Проверка CUDA
    if torch.cuda.is_available():
        print(f"CUDA доступна: {torch.cuda.get_device_name(0)}")
        print(f"Видеопамять: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("CUDA не найдена, будет использоваться CPU")
        print("Установите CUDA Toolkit: https://developer.nvidia.com/cuda-downloads")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Путь к данным
    data_dir = 'data'

    # Проверяем наличие данных
    if not os.path.exists(data_dir):
        print(f"Ошибка: папка {data_dir} не найдена!")
        print("Создайте папку 'data' и поместите в нее train и test папки")
        exit(1)

    # Оптимизированные преобразования для 3GB VRAM
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Для теста без аугментации (экономия памяти)
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("Загружаем данные...")
    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'train'),
        transform=transform
    )
    test_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'test'),
        transform=transform_test
    )

    print(f"Найдено классов: {train_dataset.classes}")
    print(f"Обучающих изображений: {len(train_dataset)}")
    print(f"Тестовых изображений: {len(test_dataset)}")

    # Критически важно для GTX 1060 3GB:
    # Уменьшаем batch_size до 16 или даже 8
    BATCH_SIZE = 16  # для 3GB видеопамяти
    print(f"Размер батча: {BATCH_SIZE} (оптимизировано для 3GB VRAM)")

    # Используем num_workers=2 для i5-7400
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=2,  # не перегружаем процессор
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print("Загружаем предобученную EfficientNet-B0...")
    model = models.efficientnet_b0(pretrained=True)

    # Заменяем классификатор
    num_classes = len(train_dataset.classes)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    # Используем mixed precision для экономии памяти
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    model = model.to(device)

    # Оптимизатор с меньшей скоростью обучения
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)  # уменьшили lr

    # Обучение
    num_epochs = 15
    print(f"Начинаем обучение на {num_epochs} эпох...")
    print("Примерное время: 1-2 часа")

    # Для отслеживания лучшей модели
    best_accuracy = 0

    for epoch in range(num_epochs):
        # Обучение
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Эпоха {epoch+1}/{num_epochs}")
        
        # Используем enumerate для получения batch_idx
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision training (экономия памяти)
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Очищаем кэш памяти каждые 100 батчей
            if torch.cuda.is_available() and batch_idx % 100 == 0:
                torch.cuda.empty_cache()
        
        avg_loss = running_loss / len(train_loader)
        
        # Тестирование после каждой эпохи
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Тестирование"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Эпоха {epoch+1}: Потеря = {avg_loss:.4f}, Точность = {accuracy:.2f}%")
        
        # Сохраняем лучшую модель
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_grape_model.pth')
            print(f"Сохранена лучшая модель с точностью {accuracy:.2f}%")
        
        # Очищаем память после каждой эпохи
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    print(f"\nОбучение завершено!")
    print(f"Лучшая точность: {best_accuracy:.2f}%")
    print(f"Модель сохранена как 'best_grape_model.pth'")