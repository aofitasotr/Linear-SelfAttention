from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import Any


LOG_FILE = os.environ.get("LOG_FILE", "log.txt")
TRAIN_RESULTS_FILE = os.environ.get("TRAIN_RESULTS_FILE", "train_results.csv")


def set_log_file(path: str) -> None:
    """Устанавливает глобальный путь к файлу логов для текущего процесса."""
    global LOG_FILE
    LOG_FILE = path


def get_log_file() -> str:
    """Возвращает текущий путь к активному лог-файлу."""
    return LOG_FILE


def write_log(message: str, log_file: str | None = None) -> None:
    """Пишет строку в лог с временной меткой.

    Ошибки логирования намеренно не пробрасываются, чтобы не валить обучение из-за
    побочных проблем с файловой системой.
    """
    try:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        file_path = Path(log_file or LOG_FILE)
        file_path.parent.mkdir(parents=True, exist_ok=True) if file_path.parent != Path(".") else None
        with file_path.open("a", encoding="utf-8") as log_stream:
            log_stream.write(f"{timestamp} - {message}\n")
    except Exception:
        pass


def append_train_results(new_data: dict[str, Any], csv_path: str) -> None:
    """Добавляет строку результатов в CSV с автоматическим расширением схемы.

    Если в новой записи появились дополнительные поля, файл пересобирается с
    новым заголовком, а старые строки дополняются пустыми значениями.
    """
    target_path = Path(csv_path)
    file_exists = target_path.is_file()
    file_empty = True

    if file_exists:
        with target_path.open("r", encoding="utf-8") as file:
            file_empty = not bool(file.read(1))

    if not file_exists or file_empty:
        with target_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(new_data.keys()))
            writer.writeheader()
            writer.writerow(new_data)
        return

    with target_path.open("r", encoding="utf-8") as file:
        reader = csv.reader(file)
        try:
            header = next(reader)
        except StopIteration:
            with target_path.open("w", encoding="utf-8", newline="") as output_file:
                writer = csv.DictWriter(output_file, fieldnames=list(new_data.keys()))
                writer.writeheader()
                writer.writerow(new_data)
            return

    existing_fields = set(header)
    new_fields = set(new_data.keys()) - existing_fields

    if not new_fields:
        with target_path.open("a", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=header)
            writer.writerow(new_data)
        return

    all_fields = header + sorted(new_fields)
    rows: list[dict[str, Any]] = []
    with target_path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            for field in new_fields:
                row[field] = ""
            rows.append(row)

    rows.append(new_data)
    with target_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(rows)
