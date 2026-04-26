import os
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from logging_utils import set_log_file, write_log
from synthetic.dataset import PositionalLookupDataset, create_positional_lookup_datasets, synthetic_collate_fn
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


def compute_synthetic_metrics(prediction_output):
    preds = prediction_output.predictions.argmax(-1)
    labels = prediction_output.label_ids
    accuracy = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
    }


def evaluate_synthetic_model(
    model: torch.nn.Module,
    test_dataset: PositionalLookupDataset,
    device: torch.device,
    batch_size: int = 64,
) -> Dict[str, float]:
    model.eval()
    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=synthetic_collate_fn,
    )

    all_preds = []
    all_labels = []
    start_time = time.time()

    with torch.inference_mode():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"] if isinstance(outputs, dict) else (
                outputs.logits if hasattr(outputs, "logits") else outputs[0]
            )

            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    total_time = time.time() - start_time
    num_samples = len(all_labels)
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "time_per_sample_ms": (total_time / num_samples) * 1000,
        "total_time_sec": total_time,
        "num_samples": num_samples,
    }


def save_synthetic_results_to_csv(results: Dict[str, Any], csv_path: str):
    flat_results = {}
    for key, value in results.items():
        if key == "system_info":
            for sys_key, sys_value in value.items():
                flat_results[f"system_{sys_key}"] = sys_value
        else:
            flat_results[key] = value

    if "k" in results:
        flat_results["k"] = results["k"]
    if "v" in results:
        flat_results["v"] = results["v"]
    elif "vocab_size" in results:
        flat_results["v"] = results["vocab_size"]
    if "vocab_size" not in flat_results and "v" in flat_results:
        flat_results["vocab_size"] = flat_results["v"]

    df = pd.DataFrame([flat_results])
    if os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0:
        try:
            existing_df = pd.read_csv(csv_path)
            df = pd.concat([existing_df, df], ignore_index=True)
        except pd.errors.EmptyDataError:
            pass

    df.to_csv(csv_path, index=False, encoding="utf-8")


def train_synthetic_model(
    attention_type: str = "dilated",
    use_original_model: bool = False,
    vocab_size: int = 100,
    hidden_size: int = 64,
    num_heads: int = 4,
    num_layers: int = 1,
    max_position_embeddings: int = 128,
    dropout_prob: float = 0.5,
    warmup_ratio: float = 0.25,
    k: int = 1,
    train_samples: int = 100000,
    eval_samples: int = 10000,
    test_samples: int = 10000,
    batch_size: int = 64,
    num_epochs: int = 30,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.15,
    seed: int = 42,
    output_dir: str = "./synthetic_output",
    log_dir: str = "./synthetic_logs",
    save_results: bool = True,
    results_csv_path: str = "./synthetic_results.csv",
    early_stop_metric_threshold: float = 1.0,
    early_stopping_patience: int = 0,
) -> Dict[str, Any]:
    del max_position_embeddings

    set_seed(seed)
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_suffix = "original" if use_original_model else attention_type
    log_file = os.path.join(log_dir, f"train_k{k}_{model_suffix}_{timestamp}.log")
    set_log_file(log_file)

    write_log("=" * 70)
    write_log("СИНТЕТИЧЕСКАЯ ЗАДАЧА: POSITIONAL LOOKUP")
    write_log("=" * 70)
    write_log(f"Тип внимания: {attention_type}")
    write_log(f"Оригинальная модель: {use_original_model}")
    write_log(f"k (расстояние): {k}")
    write_log(f"vocab_size (V): {vocab_size}")
    write_log(f"seq_len: {vocab_size + 1} (V + 1)")
    write_log(f"hidden_size: {hidden_size}")
    write_log(f"num_heads: {num_heads}")
    write_log(f"num_layers: {num_layers}")
    write_log(f"batch_size: {batch_size}")
    write_log(f"num_epochs: {num_epochs}")
    write_log(f"learning_rate: {learning_rate}")
    write_log(f"weight_decay: {weight_decay}")
    write_log(f"dropout_prob: {dropout_prob}")
    write_log(f"seed: {seed}")
    write_log(f"early_stop_metric_threshold: {early_stop_metric_threshold}")
    write_log("=" * 70)

    write_log("\nСоздание датасетов...")
    train_dataset, eval_dataset, test_dataset = create_positional_lookup_datasets(
        train_samples=train_samples,
        eval_samples=eval_samples,
        test_samples=test_samples,
        vocab_size=vocab_size,
        k=k,
        base_seed=seed,
        save_dir=os.path.join(log_dir, "datasets"),
    )
    write_log(f"  Train: {len(train_dataset)} примеров")
    write_log(f"  Eval: {len(eval_dataset)} примеров")
    write_log(f"  Test: {len(test_dataset)} примеров")

    write_log("\nСоздание модели...")
    model_kwargs = {
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "max_position_embeddings": vocab_size + k + 10,
        "dropout_prob": dropout_prob,
        "k": k,
    }
    if use_original_model:
        model = create_original_model(**model_kwargs)
        write_log("  Режим: ОРИГИНАЛЬНЫЙ BERT (классическое внимание, без marker_pos)")
    else:
        model = create_synthetic_model(attention_type=attention_type, **model_kwargs)
        write_log(f"  Режим: КАСТОМНОЕ ВНИМАНИЕ ({attention_type})")

    num_params = model.get_num_parameters()
    trainable_params = model.get_trainable_parameters()
    write_log(f"  Всего параметров: {num_params:,}")
    write_log(f"  Обучаемых параметров: {trainable_params:,}")
    write_log(f"  Размер: {num_params / 1e6:.2f}M")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_log(f"\nУстройство: {device}")
    if torch.cuda.is_available():
        write_log(f"  GPU: {torch.cuda.get_device_name(0)}")
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
        output_dir=os.path.join(output_dir, f"k{k}_{model_suffix}"),
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

    write_log("\n" + "=" * 70)
    write_log("НАЧАЛО ОБУЧЕНИЯ")
    write_log("=" * 70)
    reset_gpu_memory_stats()
    train_start_time = time.time()
    write_log("=" * 50)
    train_output = trainer.train()

    actual_stop_epoch = trainer.state.epoch
    if actual_stop_epoch is None:
        actual_stop_epoch = train_output.metrics.get("epoch", num_epochs)
    actual_stop_epoch = int(round(float(actual_stop_epoch)))

    total_training_time = time.time() - train_start_time
    write_log("\nОбучение завершено!")
    write_log(f"Общее время обучения: {total_training_time:.2f} сек")
    write_log(f"Время по callback: {sum(time_callback.epoch_train_times.values()):.2f} сек")

    write_log("\n" + "=" * 70)
    write_log("ОЦЕНКА НА ТЕСТЕ")
    write_log("=" * 70)
    test_metrics = evaluate_synthetic_model(model, test_dataset, device, batch_size)
    write_log(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    write_log(f"  F1-macro: {test_metrics['f1_macro']:.4f}")
    write_log(f"  Time per sample: {test_metrics['time_per_sample_ms']:.2f} мс")
    write_log(f"  Total time: {test_metrics['total_time_sec']:.2f} сек")

    peak_memory = memory_callback.get_memory_stats()["peak_memory_overall"]
    write_log(f"  Пиковая память GPU: {peak_memory:.2f} MB")

    results = {
        "timestamp": timestamp,
        "attention_type": attention_type,
        "use_original_model": use_original_model,
        "k": k,
        "v": vocab_size,
        "vocab_size": vocab_size,
        "seq_len": vocab_size + 1,
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
        write_log(f"\nРезультаты сохранены в {results_csv_path}")

    model_save_path = os.path.join(output_dir, f"k{k}_{model_suffix}_final")
    trainer.save_model(model_save_path)
    write_log(f"Модель сохранена в {model_save_path}")
    write_log("\n" + "=" * 70)
    write_log("ЭКСПЕРИМЕНТ ЗАВЕРШЁН")
    write_log("=" * 70)
    return results


def run_synthetic_k_sweep(
    attention_types: List[str] = None,
    use_original_model: bool = False,
    vocab_size: int = 50,
    hidden_size: int = 256,
    num_heads: int = 8,
    num_layers: int = 2,
    k_values: List[int] = None,
    batch_size: int = 32,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.01,
    dropout_prob: float = 0.1,
    base_seed: int = 42,
    output_dir: str = "./synthetic_output",
    log_dir: str = "./synthetic_logs",
    results_csv_path: str = "./synthetic_k_sweep_results.csv",
    early_stop_metric_threshold: float = 1.0,
    early_stopping_patience: int = 0,
) -> pd.DataFrame:
    if attention_types is None:
        attention_types = SYNTHETIC_ATTENTION_TYPES
    if k_values is None:
        k_values = [1, 2, 3, 4, 5, 6, 7, 8]

    set_seed(base_seed)
    write_log("=" * 70)
    write_log("ЭКСПЕРИМЕНТ")
    write_log("=" * 70)
    write_log(f"Типы внимания: {attention_types}")
    write_log(f"Оригинальная модель: {use_original_model}")
    write_log(f"Значения k: {k_values}")
    write_log(f"Всего экспериментов: {len(attention_types) * len(k_values)}")
    write_log("=" * 70)

    all_results = []
    for attention_type in attention_types:
        for k in k_values:
            write_log("\n" + "=" * 70)
            write_log(f"ЭКСПЕРИМЕНТ: attention={attention_type}, k={k}")
            write_log("=" * 70)
            try:
                result = train_synthetic_model(
                    attention_type=attention_type,
                    use_original_model=use_original_model,
                    k=k,
                    vocab_size=vocab_size,
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    num_layers=num_layers,
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    dropout_prob=dropout_prob,
                    seed=base_seed + k,
                    output_dir=output_dir,
                    log_dir=log_dir,
                    save_results=False,
                    results_csv_path=results_csv_path,
                    early_stop_metric_threshold=early_stop_metric_threshold,
                    early_stopping_patience=early_stopping_patience,
                )
                all_results.append(result)
                write_log(f"\nЗавершено: accuracy={result['test_accuracy']:.4f}")
            except Exception as e:
                write_log(f"\n Ошибка: {e}")
                all_results.append(
                    {
                        "attention_type": attention_type,
                        "use_original_model": use_original_model,
                        "k": k,
                        "v": vocab_size,
                        "vocab_size": vocab_size,
                        "test_accuracy": 0.0,
                        "test_f1_macro": 0.0,
                        "error": str(e),
                    }
                )

    if not all_results:
        return pd.DataFrame()

    df = pd.DataFrame(all_results)
    df.to_csv(results_csv_path, index=False, encoding="utf-8")
    write_log(f"\nВсе результаты сохранены в {results_csv_path}")

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))
        for att_type in attention_types:
            subset = df[df["attention_type"] == att_type]
            plt.plot(subset["k"], subset["test_accuracy"], marker="o", label=att_type, linewidth=2)

        plt.xlabel("k (расстояние от маркера)", fontsize=12)
        plt.ylabel("Accuracy", fontsize=12)
        plt.title("Зависимость точности от k для разных типов внимания", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xticks(k_values)
        plt.ylim(0, 1.05)

        plot_path = os.path.join(log_dir, "k_sweep_accuracy_plot.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        write_log(f"График сохранён в {plot_path}")
    except ImportError:
        write_log("matplotlib не установлен, график не сохранён")

    return df
