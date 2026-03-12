import csv
import os
from functools import partial

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from logging_utils import write_log


class ReviewDataset(Dataset):
    """
    Датасет, совместимый с Hugging Face Trainer.
    Возвращает токенизированные данные: input_ids, attention_mask, labels (0-4 для рейтингов 1-5).
    """

    def __init__(self, csv_path: str, tokenizer, max_length: int = 768):
        self.encodings = []
        self.labels = []  # ← Будем хранить индексы 0-4

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='|')
            for row in reader:
                text = row['text'] or ""
                if not isinstance(text, str):
                    text = str(text)

                # Токенизация при загрузке — один раз
                encoding = tokenizer(
                    text,
                    truncation=True,
                    padding=False,  # паддинг сделает DataCollator
                    max_length=max_length,
                    return_attention_mask=True,
                    return_tensors=None,  # вернём списки
                )

                self.encodings.append(encoding)
                
                rating = int(row['rating'])
                assert 1 <= rating <= 5, f"Некорректный рейтинг: {rating} (ожидается 1-5)"
                self.labels.append(rating - 1)  # 1→0, 2→1, ..., 5→4

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # преобразуем списки в тензоры + метка как тензор long
        item = {key: torch.tensor(val, dtype=torch.long) for key, val in self.encodings[idx].items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)  # ← уже 0-4
        return item


def collate_fn(batch, tokenizer, max_length=768):
    texts = [item['text'] for item in batch]
    ratings = [item['rating'] for item in batch]  # ← рейтинги 1-5 из сырых данных

    encoding = tokenizer(
        texts,
        truncation=True,
        padding=True,
        pad_to_multiple_of=8,
        max_length=max_length,
        return_tensors='pt',
    )

    # Уже правильно: преобразуем 1-5 → 0-4
    rating_labels = torch.tensor([r - 1 for r in ratings], dtype=torch.long)
    return {
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask'],
        'labels': rating_labels,  # ← 0-4
    }


def create_data_loader(transformer_loader, csv_path, batch_size=16, max_length=768):
    # Для DataLoader вне Trainer'а — читаем сырые данные
    texts, ratings = [], []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='|')
        for row in reader:
            texts.append(row['text'])
            ratings.append(int(row['rating']))

    # Создаём временный датасет с сырыми данными (только для DataLoader)
    class RawReviewDataset(Dataset):
        def __init__(self, texts, ratings):
            self.texts = texts
            self.ratings = ratings
        def __len__(self):
            return len(self.texts)
        def __getitem__(self, idx):
            return {'text': self.texts[idx], 'rating': self.ratings[idx]}

    dataset = RawReviewDataset(texts, ratings)
    collate_with_params = partial(
        collate_fn,
        tokenizer=transformer_loader.tokenizer,
        max_length=max_length,
    )

    num_workers = max(1, min(8, os.cpu_count() - 2)) if os.cpu_count() else 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_with_params,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )


def prepare_datasets(csv_path: str, random_state: int = 42):
    if csv_path is None:
        raise ValueError("prepare_datasets: csv_path обязателен и не должен быть None")

    df = pd.read_csv(csv_path, delimiter="|")
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=random_state, stratify=df["rating"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=random_state, stratify=temp_df["rating"])

    os.makedirs("datasets", exist_ok=True)
    train_df.to_csv("datasets/train_dataset.csv", index=False, sep="|")
    val_df.to_csv("datasets/val_dataset.csv", index=False, sep="|")
    test_df.to_csv("datasets/test_dataset.csv", index=False, sep="|")
    write_log(f"Разделено: {len(train_df)} train / {len(val_df)} val / {len(test_df)} test")