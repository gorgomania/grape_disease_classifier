import random
import shutil
import zipfile
from pathlib import Path


def generate_vineyard_dataset(structure, source_folders, output_dir, max_archive_size_gb=1, distribution=None):
    """
    structure: список с количеством кустов по рядам
    source_folders: dict вида {'Black Rot': '/path/to/black_rot', 'ESCA': '/path/to/esca', ...}
    output_dir: папка куда складывать результат
    max_archive_size_gb: максимальный размер одного ZIP архива в ГБ
    distribution: dict с распределением классов, например {'Healthy': 0.85, 'Black Rot': 0.05, 'ESCA': 0.05, 'Leaf Blight': 0.05}
    """
    
    # Распределение по умолчанию (85% здоровых, 5% на остальные)
    if distribution is None:
        distribution = {
            'Healthy': 0.85,
            'Black Rot': 0.05,
            'ESCA': 0.05,
            'Leaf Blight': 0.05
        }
    
    # Проверяем сумму вероятностей
    total_prob = sum(distribution.values())
    if abs(total_prob - 1.0) > 0.01:
        print(f"Сумма вероятностей = {total_prob}, нормализуем...")
        distribution = {k: v/total_prob for k, v in distribution.items()}
    
    # Преобразуем в Path если строка
    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)
    
    # Создаём выходную папку
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Собираем все файлы из папок классов
    class_files = {}
    for class_name, folder_path in source_folders.items():
        folder = Path(folder_path)
        if folder.exists():
            files = [f for f in folder.iterdir() 
                    if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]
            class_files[class_name] = files
            print(f"{class_name}: {len(files)} файлов")
        else:
            print(f"Папка не найдена: {folder_path}")
            class_files[class_name] = []
    
    # Список классов и веса для взвешенного выбора
    classes = list(distribution.keys())
    weights = list(distribution.values())
    
    # Счётчики для статистики
    class_counts = {cls: 0 for cls in classes}
    
    # Генерируем кусты во временную папку
    temp_dir = output_path / "_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    total_bushes = 0
    for row_idx, bushes_count in enumerate(structure, start=1):
        for bush_idx in range(1, bushes_count + 1):
            # Выбираем класс с заданной вероятностью
            class_name = random.choices(classes, weights=weights, k=1)[0]
            files = class_files.get(class_name, [])
            
            if not files:
                print(f"Нет файлов для класса {class_name}")
                # Если нет файлов для выбранного класса, берём здоровый
                class_name = 'Healthy'
                files = class_files.get(class_name, [])
                if not files:
                    continue
            
            source_file = random.choice(files)
            extension = source_file.suffix
            new_filename = f"Ряд {row_idx} Куст {bush_idx}{extension}"
            output_path_file = temp_dir / new_filename
            
            shutil.copy2(source_file, output_path_file)
            total_bushes += 1
            class_counts[class_name] += 1
            
            if total_bushes % 100 == 0:
                print(f"Обработано {total_bushes} кустов...")
    
    print(f"\nСоздано {total_bushes} файлов в временной папке")
    
    # Выводим статистику распределения
    print("\nСтатистика распределения:")
    for cls, count in class_counts.items():
        percent = (count / total_bushes * 100) if total_bushes > 0 else 0
        print(f"   {cls}: {count} ({percent:.1f}%)")
    
    # Разбиваем на ZIP архивы
    max_size_bytes = max_archive_size_gb * 1024 * 1024 * 1024
    archives = []
    current_archive = None
    current_size = 0
    archive_index = 1
    
    for file_path in sorted(temp_dir.iterdir()):
        file_size = file_path.stat().st_size
        
        # Если файл больше максимального размера
        if file_size > max_size_bytes:
            print(f"Файл {file_path.name} слишком большой ({file_size / 1024 / 1024:.1f} МБ), создаём отдельный архив")
            archive_name = output_path / f"vineyard_dataset_part_{archive_index}.zip"
            with zipfile.ZipFile(archive_name, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.write(file_path, file_path.name)
            archives.append(archive_name)
            archive_index += 1
            continue
        
        # Если текущий архив не существует, создаём
        if current_archive is None:
            current_archive = output_path / f"vineyard_dataset_part_{archive_index}.zip"
            current_size = 0
        
        # Если файл не влезает, закрываем архив и создаём новый
        if current_size + file_size > max_size_bytes:
            print(f"Создан архив {current_archive.name} ({current_size / 1024 / 1024:.1f} МБ)")
            archive_index += 1
            current_archive = output_path / f"vineyard_dataset_part_{archive_index}.zip"
            current_size = 0
        
        # Добавляем файл в архив
        with zipfile.ZipFile(current_archive, 'a', zipfile.ZIP_DEFLATED) as zf:
            zf.write(file_path, file_path.name)
        current_size += file_size
    
    # Добавляем последний архив
    if current_archive is not None:
        print(f"Создан архив {current_archive.name} ({current_size / 1024 / 1024:.1f} МБ)")
        archives.append(current_archive)
    
    # Удаляем временную папку
    shutil.rmtree(temp_dir)
    
    print(f"\nГотово! Создано {len(archives)} архивов в {output_path}")
    total_size = 0
    for i, archive in enumerate(archives, 1):
        size_mb = archive.stat().st_size / 1024 / 1024
        total_size += archive.stat().st_size
        print(f"   {i}. {archive.name} - {size_mb:.1f} МБ")
    print(f"Общий размер: {total_size / 1024 / 1024 / 1024:.2f} ГБ")

# Пример использования
if __name__ == "__main__":
    print("Введите структуру виноградника")
    vineyard_structure = [int(i) for i in input()[1:-1].split(',')]
    
    source_folders = {
        'Black Rot': 'data/Black Rot',
        'Leaf Blight': 'data/Leaf Blight',
        'Healthy': 'data/Healthy',
        'ESCA': 'data/ESCA'
    }
    
    # Распределение: 85% здоровых, по 5% на каждую болезнь
    distribution = {
        'Healthy': 0.85,
        'Black Rot': 0.05,
        'ESCA': 0.05,
        'Leaf Blight': 0.05
    }
    
    output_directory = './vineyard_dataset'
    
    # Генерируем с разбивкой на архивы по 1 ГБ и заданным распределением
    generate_vineyard_dataset(vineyard_structure, source_folders, output_directory, max_archive_size_gb=1, distribution=distribution)