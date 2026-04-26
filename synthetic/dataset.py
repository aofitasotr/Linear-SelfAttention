import os
import random
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset


def _build_unique_samples(
    total_samples: int,
    vocab_size: int,
    k: int,
    seed: int,
    marker_value: int,
) -> List[Tuple[List[int], int, int]]:
    rng = random.Random(seed)
    seq_len = vocab_size + 1
    max_marker_pos = seq_len - k - 1
    if max_marker_pos < 0:
        raise ValueError(f"k={k} is too large for vocab_size={vocab_size}")

    max_unique_sequences = vocab_size * max(1, max_marker_pos + 1)
    if total_samples > max_unique_sequences:
        raise ValueError(
            f"Cannot create {total_samples} unique samples for vocab_size={vocab_size}, k={k}. "
            f"Maximum possible is {max_unique_sequences}."
        )

    unique_samples: List[Tuple[List[int], int, int]] = []
    seen_sequences = set()
    while len(unique_samples) < total_samples:
        seq = list(range(1, vocab_size + 1))
        rng.shuffle(seq)
        marker_pos = rng.randint(0, max_marker_pos)
        seq.insert(marker_pos, marker_value)
        seq_key = tuple(seq)
        if seq_key in seen_sequences:
            continue
        seen_sequences.add(seq_key)
        target_pos = marker_pos + k
        target_value = seq[target_pos]
        unique_samples.append((seq, target_value, marker_pos))

    return unique_samples


class PositionalLookupDataset(Dataset):
    def __init__(
        self,
        n_samples: int = 10000,
        vocab_size: int = 50,
        k: int = 1,
        seed: int = 42,
        marker_value: int = 0,
        pad_value: int = -1,
        precomputed_data: List[Tuple[List[int], int, int]] | None = None,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.vocab_size = vocab_size
        self.seq_len = vocab_size + 1
        self.k = k
        self.marker_value = marker_value
        self.pad_value = pad_value

        if precomputed_data is not None:
            self.data = list(precomputed_data)
        else:
            self.data = _build_unique_samples(
                total_samples=n_samples,
                vocab_size=vocab_size,
                k=k,
                seed=seed,
                marker_value=marker_value,
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq, target, _ = self.data[idx]
        return {
            "input_ids": torch.tensor(seq, dtype=torch.long),
            "labels": torch.tensor(target - 1, dtype=torch.long),
        }

    def get_vocab_info(self) -> Dict[str, int]:
        return {
            "vocab_size": self.vocab_size,
            "marker_value": self.marker_value,
            "pad_value": self.pad_value,
            "actual_vocab_size": self.vocab_size + 2,
            "seq_len": self.seq_len,
        }


def synthetic_collate_fn(batch, pad_value: int = -1):
    if not isinstance(batch[0], dict):
        raise TypeError(f"Expected list[dict], got {type(batch[0])}")

    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids_list = []
    labels_list = []
    attention_masks = []

    for item in batch:
        seq = item["input_ids"].tolist() if isinstance(item["input_ids"], torch.Tensor) else list(item["input_ids"])
        actual_len = len(seq)
        pad_len = max_len - actual_len

        input_ids_list.append(seq + [pad_value] * pad_len)
        labels_list.append(int(item["labels"].item() if isinstance(item["labels"], torch.Tensor) else item["labels"]))
        attention_masks.append([1.0] * actual_len + [0.0] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
        "labels": torch.tensor(labels_list, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.float),
    }


def create_positional_lookup_datasets(
    train_samples: int = 10000,
    eval_samples: int = 1000,
    test_samples: int = 1000,
    vocab_size: int = 50,
    k: int = 1,
    base_seed: int = 42,
    save_dir: str = "./synthetic_datasets",
):
    total_samples = train_samples + eval_samples + test_samples
    unique_samples = _build_unique_samples(
        total_samples=total_samples,
        vocab_size=vocab_size,
        k=k,
        seed=base_seed,
        marker_value=0,
    )

    train_data = unique_samples[:train_samples]
    eval_data = unique_samples[train_samples:train_samples + eval_samples]
    test_data = unique_samples[train_samples + eval_samples:]

    train_dataset = PositionalLookupDataset(
        n_samples=train_samples,
        vocab_size=vocab_size,
        k=k,
        seed=base_seed,
        precomputed_data=train_data,
    )
    eval_dataset = PositionalLookupDataset(
        n_samples=eval_samples,
        vocab_size=vocab_size,
        k=k,
        seed=base_seed + 1,
        precomputed_data=eval_data,
    )
    test_dataset = PositionalLookupDataset(
        n_samples=test_samples,
        vocab_size=vocab_size,
        k=k,
        seed=base_seed + 2,
        precomputed_data=test_data,
    )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        info_path = os.path.join(save_dir, "dataset_info.txt")
        with open(info_path, "w", encoding="utf-8") as file:
            file.write("Synthetic Positional Lookup Dataset\n")
            file.write("=" * 50 + "\n")
            file.write(f"vocab_size (V): {vocab_size}\n")
            file.write(f"seq_len: {vocab_size + 1} (V + 1)\n")
            file.write(f"k: {k}\n")
            file.write(f"train_samples: {train_samples}\n")
            file.write(f"eval_samples: {eval_samples}\n")
            file.write(f"test_samples: {test_samples}\n")
            file.write(f"base_seed: {base_seed}\n")
            file.write("unique_across_splits: True\n")

    return train_dataset, eval_dataset, test_dataset
