import os
import time
from typing import Any, Dict, List

import pandas as pd
import torch

from logging_utils import set_log_file, write_log
from synthetic.consecutive_ones_dataset import ConsecutiveOnesDataset, create_consecutive_ones_datasets
from synthetic.dataset import synthetic_collate_fn
from synthetic.model import SYNTHETIC_ATTENTION_TYPES, create_original_model, create_synthetic_model
from .callbacks import (
    CustomTrainer,
    DetailedMemoryCallback,
    EarlyStoppingOnPatience,
    EarlyStopOnMetricsThreshold,
    EpochTrainingTimeCallback,
    EvalMetricsLoggingCallback,
    get_system_info,
    reset_gpu_memory_stats,
    set_seed,
)
from .config import build_training_args
from .synthetic_pipeline import compute_synthetic_metrics, evaluate_synthetic_model, save_synthetic_results_to_csv


def train_consecutive_ones_model(
    attention_type: str = "dilated",
    use_original_model: bool = False,
    context_len: int = 64,
    hidden_size: int = 64,
    num_heads: int = 4,
    num_layers: int = 1,
    max_position_embeddings: int = 128,
    dropout_prob: float = 0.5,
    warmup_ratio: float = 0.25,
    train_samples: int = 100000,
    eval_samples: int = 10000,
    test_samples: int = 10000,
    batch_size: int = 64,
    num_epochs: int = 30,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.15,
    seed: int = 42,
    output_dir: str = "./synthetic_consecutive_ones_output",
    log_dir: str = "./synthetic_consecutive_ones_logs",
    save_results: bool = True,
    results_csv_path: str = "./synthetic_consecutive_ones_results.csv",
    early_stop_metric_threshold: float = 1.0,
    early_stopping_patience: int = 0,
) -> Dict[str, Any]:
    del max_position_embeddings

    set_seed(seed)
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_suffix = "original" if use_original_model else attention_type
    log_file = os.path.join(log_dir, f"train_consecutive_ones_{model_suffix}_{timestamp}.log")
    set_log_file(log_file)

    sequence_length = context_len

    write_log("=" * 70)
    write_log("SYNTHETIC TASK: CONSECUTIVE ONES")
    write_log("=" * 70)
    write_log(f"attention_type: {attention_type}")
    write_log(f"use_original_model: {use_original_model}")
    write_log("alphabet: {0, 1}")
    write_log(f"context_len: {context_len}")
    write_log(f"seq_len: {sequence_length}")
    write_log("label_rule: target = maximum number of consecutive ones")

    train_dataset, eval_dataset, test_dataset = create_consecutive_ones_datasets(
        train_samples=train_samples,
        eval_samples=eval_samples,
        test_samples=test_samples,
        context_len=context_len,
        base_seed=seed,
        save_dir=os.path.join(log_dir, "datasets"),
    )

    model_kwargs = {
        "vocab_size": 2,
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "max_position_embeddings": sequence_length + 10,
        "dropout_prob": dropout_prob,
        "num_labels": context_len + 1,
        "pooling_mode": "mean",
    }
    if use_original_model:
        model = create_original_model(**model_kwargs)
    else:
        model = create_synthetic_model(attention_type=attention_type, **model_kwargs)

    num_params = model.get_num_parameters()
    trainable_params = model.get_trainable_parameters()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        reset_gpu_memory_stats()
    model = model.to(device)

    os.makedirs(output_dir, exist_ok=True)
    training_args = build_training_args(
        training_args_config={
            "optim": "adamw_torch",
            "num_train_epochs": num_epochs,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "learning_rate": learning_rate,
            "warmup_ratio": warmup_ratio,
            "lr_scheduler_type": "linear",
            "weight_decay": weight_decay,
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "load_best_model_at_end": True,
            "metric_for_best_model": "f1_macro",
            "greater_is_better": True,
            "save_total_limit": 2,
            "report_to": "none",
            "dataloader_num_workers": 0,
            "dataloader_pin_memory": torch.cuda.is_available(),
            "disable_tqdm": False,
            "fp16": torch.cuda.is_available(),
            "bf16": False,
        },
        output_dir=os.path.join(output_dir, f"consecutive_ones_{model_suffix}"),
        logging_dir=log_dir,
        train_dataset_size=len(train_dataset),
        seed=seed,
    )

    time_callback = EpochTrainingTimeCallback(system_warmup_epochs=0)
    memory_callback = DetailedMemoryCallback()
    eval_callback = EvalMetricsLoggingCallback()
    early_stop_callback = EarlyStopOnMetricsThreshold(threshold=early_stop_metric_threshold)
    patience_callback = EarlyStoppingOnPatience(
        patience=early_stopping_patience,
        metric_name="eval_f1_macro",
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_synthetic_metrics,
        data_collator=synthetic_collate_fn,
        callbacks=[time_callback, memory_callback, eval_callback, early_stop_callback, patience_callback],
    )

    train_start_time = time.time()
    train_output = trainer.train()
    actual_stop_epoch = trainer.state.epoch
    if actual_stop_epoch is None:
        actual_stop_epoch = train_output.metrics.get("epoch", num_epochs)
    actual_stop_epoch = int(round(float(actual_stop_epoch)))
    total_training_time = time.time() - train_start_time

    test_metrics = evaluate_synthetic_model(model, test_dataset, device, batch_size)
    peak_memory = memory_callback.get_memory_stats()["peak_memory_overall"]
    results = {
        "timestamp": timestamp,
        "task_type": "consecutive-ones",
        "attention_type": attention_type,
        "use_original_model": use_original_model,
        "context_len": context_len,
        "seq_len": sequence_length,
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "stopped_epoch": actual_stop_epoch,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "dropout_prob": dropout_prob,
        "seed": seed,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "test_samples": len(test_dataset),
        "num_parameters": num_params,
        "trainable_parameters": trainable_params,
        "test_accuracy": test_metrics["accuracy"],
        "test_f1_macro": test_metrics["f1_macro"],
        "test_time_per_sample_ms": test_metrics["time_per_sample_ms"],
        "test_total_time_sec": test_metrics["total_time_sec"],
        "total_training_time_sec": total_training_time,
        "peak_memory_gpu_mb": peak_memory,
        "device": str(device),
        "system_info": get_system_info(),
    }

    if save_results:
        save_synthetic_results_to_csv(results, results_csv_path)

    model_save_path = os.path.join(output_dir, f"consecutive_ones_{model_suffix}_final")
    trainer.save_model(model_save_path)
    return results


def run_consecutive_ones_attention_sweep(
    attention_types: List[str] = None,
    use_original_model: bool = False,
    context_len: int = 64,
    hidden_size: int = 256,
    num_heads: int = 8,
    num_layers: int = 2,
    batch_size: int = 32,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.01,
    dropout_prob: float = 0.1,
    base_seed: int = 42,
    output_dir: str = "./synthetic_consecutive_ones_output",
    log_dir: str = "./synthetic_consecutive_ones_logs",
    results_csv_path: str = "./synthetic_consecutive_ones_results.csv",
    early_stop_metric_threshold: float = 1.0,
    early_stopping_patience: int = 0,
) -> pd.DataFrame:
    if attention_types is None:
        attention_types = SYNTHETIC_ATTENTION_TYPES

    set_seed(base_seed)
    all_results = []
    for attention_type in attention_types:
        try:
            all_results.append(
                train_consecutive_ones_model(
                    attention_type=attention_type,
                    use_original_model=use_original_model,
                    context_len=context_len,
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    num_layers=num_layers,
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    dropout_prob=dropout_prob,
                    seed=base_seed,
                    output_dir=output_dir,
                    log_dir=log_dir,
                    save_results=False,
                    results_csv_path=results_csv_path,
                    early_stop_metric_threshold=early_stop_metric_threshold,
                    early_stopping_patience=early_stopping_patience,
                )
            )
        except Exception as error:
            all_results.append(
                {
                    "task_type": "consecutive-ones",
                    "attention_type": attention_type,
                    "use_original_model": use_original_model,
                    "context_len": context_len,
                    "test_accuracy": 0.0,
                    "test_f1_macro": 0.0,
                    "error": str(error),
                }
            )

    df = pd.DataFrame(all_results)
    df.to_csv(results_csv_path, index=False, encoding="utf-8")
    return df
