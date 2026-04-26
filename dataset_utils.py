from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from logging_utils import write_log


CSV_DELIMITER = "|"
DEFAULT_DATASETS_DIR = Path("datasets")


def _normalize_text(value: Any) -> str:
    text = value or ""
    return text if isinstance(text, str) else str(text)


def rating_to_label(rating: int) -> int:
    """Преобразует рейтинг 1..5 в индекс класса 0..4."""
    if not 1 <= rating <= 5:
        raise ValueError(f"Некорректный рейтинг: {rating} (ожидается диапазон 1..5)")
    return rating - 1


def load_reviews(csv_path: str) -> list[tuple[str, int]]:
    """Читает CSV отзывов формата `text|rating`."""
    rows: list[tuple[str, int]] = []
    with open(csv_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter=CSV_DELIMITER)
        for row in reader:
            rows.append((_normalize_text(row["text"]), int(row["rating"])))
    return rows


@dataclass(frozen=True, slots=True)
class ReviewRecord:
    text: str
    rating: int


class RawReviewDataset(Dataset):
    """Неформатированный датасет отзывов до токенизации."""

    def __init__(self, rows: list[tuple[str, int]]):
        self.rows = [ReviewRecord(text=text, rating=rating) for text, rating in rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.rows[idx]
        return {"text": record.text, "rating": record.rating}


class ReviewDataset(Dataset):
    """Датасет, совместимый с Hugging Face `Trainer`."""

    def __init__(self, csv_path: str, tokenizer, max_length: int = 768):
        self.encodings: list[dict[str, list[int]]] = []
        self.labels: list[int] = []

        for text, rating in load_reviews(csv_path):
            encoding = tokenizer(
                text,
                truncation=True,
                padding=False,
                max_length=max_length,
                return_attention_mask=True,
                return_tensors=None,
            )
            self.encodings.append(encoding)
            self.labels.append(rating_to_label(rating))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(value, dtype=torch.long) for key, value in self.encodings[idx].items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def collate_fn(batch, tokenizer, max_length: int = 768) -> dict[str, torch.Tensor]:
    """Токенизирует батч сырых текстов и возвращает тензоры для обучения."""
    texts = [item["text"] for item in batch]
    labels = torch.tensor([rating_to_label(item["rating"]) for item in batch], dtype=torch.long)

    encoding = tokenizer(
        texts,
        truncation=True,
        padding=True,
        pad_to_multiple_of=8,
        max_length=max_length,
        return_tensors="pt",
    )

    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels": labels,
    }


def create_data_loader(transformer_loader, csv_path: str, batch_size: int = 16, max_length: int = 768) -> DataLoader:
    """Создаёт DataLoader для инференса или ручной оценки."""
    dataset = RawReviewDataset(load_reviews(csv_path))
    collate_with_params = partial(
        collate_fn,
        tokenizer=transformer_loader.tokenizer,
        max_length=max_length,
    )

    cpu_count = os.cpu_count() or 0
    num_workers = max(1, min(8, cpu_count - 2)) if cpu_count else 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_with_params,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def prepare_datasets(csv_path: str, random_state: int = 42, output_dir: Path | str = DEFAULT_DATASETS_DIR) -> None:
    """Делит исходный CSV на `train/val/test` со стратификацией по рейтингу."""
    if csv_path is None:
        raise ValueError("`csv_path` обязателен и не должен быть None")

    target_dir = Path(output_dir)
    dataframe = pd.read_csv(csv_path, delimiter=CSV_DELIMITER)
    train_df, temp_df = train_test_split(
        dataframe,
        test_size=0.2,
        random_state=random_state,
        stratify=dataframe["rating"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=random_state,
        stratify=temp_df["rating"],
    )

    target_dir.mkdir(exist_ok=True)
    train_df.to_csv(target_dir / "train_dataset.csv", index=False, sep=CSV_DELIMITER)
    val_df.to_csv(target_dir / "val_dataset.csv", index=False, sep=CSV_DELIMITER)
    test_df.to_csv(target_dir / "test_dataset.csv", index=False, sep=CSV_DELIMITER)
    write_log(f"Разделено: {len(train_df)} train / {len(val_df)} val / {len(test_df)} test")
