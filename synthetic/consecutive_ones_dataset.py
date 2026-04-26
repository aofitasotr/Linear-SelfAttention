import os
import random
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset


def _max_consecutive_ones(sequence: List[int]) -> int:
    best = 0
    current = 0
    for token in sequence:
        if token == 1:
            current += 1
            if current > best:
                best = current
        else:
            current = 0
    return best


def _build_consecutive_ones_samples(
    total_samples: int,
    context_len: int,
    seed: int,
) -> List[Tuple[List[int], int]]:
    rng = random.Random(seed)
    if context_len < 1:
        raise ValueError(f"context_len={context_len} must be positive")

    max_unique_sequences = 2 ** context_len
    if total_samples > max_unique_sequences:
        raise ValueError(
            f"Cannot create {total_samples} unique consecutive-ones samples for context_len={context_len}. "
            f"Maximum possible is {max_unique_sequences}."
        )

    unique_samples: List[Tuple[List[int], int]] = []
    seen_sequences = set()

    while len(unique_samples) < total_samples:
        sequence = [rng.randint(0, 1) for _ in range(context_len)]
        sequence_key = tuple(sequence)
        if sequence_key in seen_sequences:
            continue

        seen_sequences.add(sequence_key)
        target = _max_consecutive_ones(sequence)
        unique_samples.append((sequence, target))

    return unique_samples


class ConsecutiveOnesDataset(Dataset):
    def __init__(
        self,
        n_samples: int = 10000,
        context_len: int = 64,
        seed: int = 42,
        pad_value: int = -1,
        precomputed_data: List[Tuple[List[int], int]] | None = None,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.context_len = context_len
        self.seq_len = context_len
        self.pad_value = pad_value

        if precomputed_data is not None:
            self.data = list(precomputed_data)
        else:
            self.data = _build_consecutive_ones_samples(
                total_samples=n_samples,
                context_len=context_len,
                seed=seed,
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence, target = self.data[idx]
        return {
            "input_ids": torch.tensor(sequence, dtype=torch.long),
            "labels": torch.tensor(target, dtype=torch.long),
        }


def create_consecutive_ones_datasets(
    train_samples: int = 10000,
    eval_samples: int = 1000,
    test_samples: int = 1000,
    context_len: int = 64,
    base_seed: int = 42,
    save_dir: str = "./synthetic_consecutive_ones_datasets",
):
    total_samples = train_samples + eval_samples + test_samples
    unique_samples = _build_consecutive_ones_samples(
        total_samples=total_samples,
        context_len=context_len,
        seed=base_seed,
    )

    train_data = unique_samples[:train_samples]
    eval_data = unique_samples[train_samples:train_samples + eval_samples]
    test_data = unique_samples[train_samples + eval_samples:]

    train_dataset = ConsecutiveOnesDataset(
        n_samples=train_samples,
        context_len=context_len,
        seed=base_seed,
        precomputed_data=train_data,
    )
    eval_dataset = ConsecutiveOnesDataset(
        n_samples=eval_samples,
        context_len=context_len,
        seed=base_seed + 1,
        precomputed_data=eval_data,
    )
    test_dataset = ConsecutiveOnesDataset(
        n_samples=test_samples,
        context_len=context_len,
        seed=base_seed + 2,
        precomputed_data=test_data,
    )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        info_path = os.path.join(save_dir, "dataset_info.txt")
        with open(info_path, "w", encoding="utf-8") as file:
            file.write("Synthetic Consecutive Ones Dataset\n")
            file.write("=" * 50 + "\n")
            file.write("alphabet: {0, 1}\n")
            file.write(f"context_len: {context_len}\n")
            file.write(f"seq_len: {context_len}\n")
            file.write("label_rule: target = maximum number of consecutive ones\n")
            file.write(f"max_label: {context_len}\n")
            file.write(f"train_samples: {train_samples}\n")
            file.write(f"eval_samples: {eval_samples}\n")
            file.write(f"test_samples: {test_samples}\n")
            file.write(f"base_seed: {base_seed}\n")
            file.write("unique_across_splits: True\n")

    return train_dataset, eval_dataset, test_dataset
