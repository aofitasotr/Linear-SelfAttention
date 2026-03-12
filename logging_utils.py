import csv
import os
import time

LOG_FILE = os.environ.get("LOG_FILE", "log.txt")
TRAIN_RESULTS_FILE = os.environ.get("TRAIN_RESULTS_FILE", "train_results.csv")


def set_log_file(path: str):
    """Set the log file path globally."""
    global LOG_FILE
    LOG_FILE = path


def get_log_file() -> str:
    return LOG_FILE


def write_log(message: str, log_file: str = None):
    """Append timestamped message to LOG_FILE or provided file."""
    try:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        file_path = log_file if log_file is not None else LOG_FILE
        with open(file_path, "a", encoding="utf-8") as lf:
            lf.write(f"{ts} - {message}\n")
    except Exception:
        pass


import os
import csv
from typing import Dict, Any

def append_train_results(new_data: Dict[str, Any], csv_path: str):
    """
    Добавляет результаты в CSV файл. Автоматически расширяет заголовок при появлении новых полей.
    
    Args:
        new_data: словарь с данными для записи
        csv_path: путь к CSV файлу
    """
    file_exists = os.path.isfile(csv_path)
    file_empty = True
    
    # Проверяем, не пустой ли файл
    if file_exists:
        with open(csv_path, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            file_empty = not bool(first_char)
    
    # Если файла нет или он пуст, создаём с заголовком из ключей
    if not file_exists or file_empty:
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(new_data.keys()))
            writer.writeheader()
            writer.writerow(new_data)
        return
    
    # Читаем существующий заголовок
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            # Файл существует но пуст - перезаписываем
            with open(csv_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(new_data.keys()))
                writer.writeheader()
                writer.writerow(new_data)
            return
    
    # Проверяем, есть ли новые поля
    existing_fields = set(header)
    new_fields = set(new_data.keys()) - existing_fields
    
    if not new_fields:
        # Если новых полей нет, просто добавляем строку
        with open(csv_path, 'a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writerow(new_data)
        return
    
    # Если есть новые поля, нужно пересоздать файл с расширенным заголовком
    all_fields = header + sorted(new_fields)
    
    # Читаем все существующие строки
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Добавляем новые поля с пустыми значениями
            for field in new_fields:
                row[field] = ''
            rows.append(row)
    
    # Добавляем новую строку
    rows.append(new_data)
    
    # Перезаписываем файл с новым заголовком
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(rows)