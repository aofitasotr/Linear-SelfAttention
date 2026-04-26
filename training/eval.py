import time
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from logging_utils import append_train_results, write_log
from .callbacks import reset_gpu_memory_stats


def evaluate_by_length_bins(model, dataset, tokenizer, class_names, device, max_len, step=128):
    model.eval()
    bins = list(range(0, max_len, step))
    if bins[-1] < max_len:
        bins.append(max_len)

    bin_labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]
    groups = {label: {"preds": [], "labels": [], "count": 0} for label in bin_labels}
    groups["all"] = {"preds": [], "labels": [], "count": 0}

    data_collator = DataCollatorWithPadding(tokenizer, padding=True, pad_to_multiple_of=8)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=data_collator)

    total_start_time = time.time()
    group_times = {label: 0.0 for label in bin_labels}
    group_times["all"] = 0.0

    with torch.inference_mode():
        for batch in dataloader:
            lengths = batch["attention_mask"].sum(dim=1).cpu().numpy()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            batch_start = time.time()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_time = time.time() - batch_start

            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            labels_np = labels.cpu().numpy()

            for i, length in enumerate(lengths):
                groups["all"]["preds"].append(preds[i])
                groups["all"]["labels"].append(labels_np[i])
                groups["all"]["count"] += 1

                bin_idx = min(int(length // step), len(bin_labels) - 1)
                bin_label = bin_labels[bin_idx]
                groups[bin_label]["preds"].append(preds[i])
                groups[bin_label]["labels"].append(labels_np[i])
                groups[bin_label]["count"] += 1
                group_times[bin_label] += batch_time / len(lengths)
                group_times["all"] += batch_time / len(lengths)

    total_time = time.time() - total_start_time
    results = {}
    output_lines = [
        "\n" + "=" * 70,
        "ОЦЕНКА НА ТЕСТЕ ПО ГРУППАМ ДЛИНЫ (ШАГ 128)",
        "=" * 70,
    ]

    for label, data in groups.items():
        if data["count"] == 0:
            output_lines.append(f"\n{label}: нет примеров")
            continue

        preds = np.array(data["preds"])
        labels_arr = np.array(data["labels"])
        accuracy = accuracy_score(labels_arr, preds)
        f1_macro = f1_score(labels_arr, preds, average="macro", zero_division=0)
        f1_weighted = f1_score(labels_arr, preds, average="weighted", zero_division=0)
        avg_time_per_sample_ms = (group_times[label] / data["count"]) * 1000

        output_lines.extend(
            [
                f"\n{label} (n={data['count']}):",
                f"  Accuracy: {accuracy:.4f}",
                f"  F1-macro: {f1_macro:.4f}",
                f"  F1-weighted: {f1_weighted:.4f}",
                f"  Total time: {group_times[label]:.4f} сек",
                f"  Avg time per sample: {avg_time_per_sample_ms:.2f} мс",
            ]
        )

        results[label] = {
            "count": data["count"],
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "time_sec": group_times[label],
            "time_per_sample_ms": avg_time_per_sample_ms,
        }

    output_lines.extend(
        [
            "\n" + "=" * 70,
            f"ВСЕГО (n={groups['all']['count']}):",
            f"  Total inference time: {total_time:.4f} сек",
            f"  Avg time per sample: {(total_time / groups['all']['count']) * 1000:.2f} мс",
            "=" * 70 + "\n",
        ]
    )

    full_output = "\n".join(output_lines)
    print(full_output)
    write_log(full_output)
    return results


def log_classification_report(model, eval_dataset, tokenizer, class_names, device="cuda"):
    model.eval()
    data_collator = DataCollatorWithPadding(tokenizer, padding=True, pad_to_multiple_of=8)
    dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=False, collate_fn=data_collator)

    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            reset_gpu_memory_stats()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    write_log("\nФИНАЛЬНАЯ ОЦЕНКА НА ВАЛИДАЦИИ:")
    write_log(f"  Accuracy:  {accuracy:.4f}")
    write_log(f"  F1-macro:  {f1_macro:.4f}")
    write_log(f"  F1-weighted: {f1_weighted:.4f}")
    write_log("\nДЕТАЛЬНЫЙ ОТЧЁТ ПО КЛАССАМ:")

    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        output_dict=True,
        digits=4,
        zero_division=0,
    )
    report_str = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    for line in report_str.split("\n"):
        if line.strip():
            write_log(f"  {line}")

    class_metrics = {}
    for i, _class_name in enumerate(class_names):
        class_key = str(i)
        if class_key not in report:
            continue
        class_data = report[class_key]
        class_metrics[f"class_{i+1}_precision"] = class_data.get("precision", 0)
        class_metrics[f"class_{i+1}_recall"] = class_data.get("recall", 0)
        class_metrics[f"class_{i+1}_f1"] = class_data.get("f1-score", 0)
        class_metrics[f"class_{i+1}_support"] = class_data.get("support", 0)

    return class_metrics


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def save_all_results_to_csv(
    output_path: str,
    run_params: Dict,
    training_args_config: Dict,
    bert_config_params: Dict,
    final_metrics: Dict,
    epoch_metrics: list,
    memory_stats: Dict,
    system_info: Dict,
    test_length_metrics: Optional[Dict] = None,
    attention_type: str = None,
):
    main_metrics = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "experiment_id": run_params.get("experiment_id", "unknown"),
        "attention_type": attention_type if attention_type else "unknown",
        "train_loss": final_metrics.get("train_loss", ""),
        "eval_loss": final_metrics.get("eval_loss", ""),
        "eval_accuracy": final_metrics.get("eval_accuracy", ""),
        "eval_f1_macro": final_metrics.get("eval_f1_macro", ""),
        "eval_f1_weighted": final_metrics.get("eval_f1_weighted", ""),
        "eval_mae": final_metrics.get("eval_mae", ""),
        "eval_precision_macro": final_metrics.get("eval_precision_macro", ""),
        "eval_recall_macro": final_metrics.get("eval_recall_macro", ""),
        "best_metric": final_metrics.get("best_metric", ""),
        "total_training_time_sec": final_metrics.get("total_training_time_sec", ""),
        "avg_epoch_time_sec": final_metrics.get("avg_epoch_time_sec", ""),
        "std_epoch_time_sec": final_metrics.get("std_epoch_time_sec", ""),
        "epochs_completed": final_metrics.get("epochs_completed", ""),
        "num_layers_replace": run_params.get("num_layers_to_replace", ""),
        "num_layers_add": run_params.get("num_layers_to_add", ""),
        "num_layers_remove": run_params.get("num_layers_to_remove", ""),
        "train_dataset_size": run_params.get("train_dataset_size", ""),
        "eval_dataset_size": run_params.get("eval_dataset_size", ""),
        "peak_memory_gpu_mb": memory_stats.get("peak_memory_overall", 0),
        "avg_epoch_peak_memory_mb": memory_stats.get("avg_epoch_peak", 0),
    }

    for idx in ["1", "2", "3", "4", "5"]:
        class_key = f"class_{idx}"
        for key, value in final_metrics.items():
            if key.startswith(f"{class_key}_"):
                main_metrics[key] = value

    if test_length_metrics:
        for bin_label, metrics in test_length_metrics.items():
            prefix = "test_all" if bin_label == "all" else f"test_{bin_label.replace('-', '_')}"
            main_metrics[f"{prefix}_count"] = metrics.get("count", "")
            main_metrics[f"{prefix}_accuracy"] = metrics.get("accuracy", "")
            main_metrics[f"{prefix}_f1_macro"] = metrics.get("f1_macro", "")
            main_metrics[f"{prefix}_f1_weighted"] = metrics.get("f1_weighted", "")
            main_metrics[f"{prefix}_time_sec"] = metrics.get("time_sec", "")
            main_metrics[f"{prefix}_time_per_sample_ms"] = metrics.get("time_per_sample_ms", "")

    exclude_keys = [
        "num_layers_to_replace",
        "num_layers_to_add",
        "num_layers_to_remove",
        "train_dataset_size",
        "eval_dataset_size",
        "experiment_id",
    ]
    flat_run_params = flatten_dict(
        {k: v for k, v in run_params.items() if k not in exclude_keys},
        "run",
    )
    flat_training_args = flatten_dict(training_args_config, "training_arg")
    flat_bert_config = flatten_dict(bert_config_params, "bert_config")
    flat_system_info = flatten_dict(system_info, "system")

    row = {
        **main_metrics,
        **flat_run_params,
        **flat_training_args,
        **flat_bert_config,
        **flat_system_info,
    }
    append_train_results(row, output_path)

    if epoch_metrics:
        epoch_output_path = output_path.replace(".csv", "_epochs.csv")
        for epoch_metric in epoch_metrics:
            append_train_results(epoch_metric, epoch_output_path)
