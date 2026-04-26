from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

from logging_utils import write_log


def build_compute_metrics(class_names: list[str] | None):
    """Создаёт callback-compatible функцию расчёта метрик для `Trainer`."""

    def compute_metrics_fn(prediction_output):
        preds = prediction_output.predictions.argmax(-1)
        labels = prediction_output.label_ids

        mae = np.mean(np.abs(preds - labels))
        precision = precision_score(labels, preds, average="macro", zero_division=0)
        recall = recall_score(labels, preds, average="macro", zero_division=0)
        f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
        f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)

        target_names = class_names[: len(set(labels))] if class_names else [str(i) for i in sorted(set(labels))]
        print("\n" + "=" * 60)
        print(
            f"MAE: {mae:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | "
            f"F1-macro: {f1_macro:.3f} | F1-weighted: {f1_weighted:.3f}"
        )
        print("-" * 60)
        print("Детальный отчёт по классам:")
        print(
            classification_report(
                labels,
                preds,
                target_names=target_names,
                digits=3,
                zero_division=0,
            )
        )
        print("=" * 60 + "\n")

        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "mae": mae,
            "precision_macro": precision,
            "recall_macro": recall,
        }

    return compute_metrics_fn


def build_epoch_metrics(trainer, time_callback, num_layers_to_replace: int, num_layers_to_add: int, num_layers_to_remove: int):
    """Преобразует `trainer.state.log_history` в табличный вид по эпохам."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_history = getattr(trainer.state, "log_history", [])
    last_train_loss_by_epoch: dict[int, Any] = {}

    for log_item in log_history:
        if "loss" in log_item and "eval_loss" not in log_item:
            epoch_val = log_item.get("epoch")
            if epoch_val is not None:
                last_train_loss_by_epoch[int(round(epoch_val))] = log_item["loss"]

    epoch_metrics_list = []
    for log_item in log_history:
        if "eval_loss" not in log_item:
            continue
        epoch_val = log_item.get("epoch")
        if epoch_val is None:
            continue

        epoch_int = int(round(epoch_val))
        epoch_metrics_list.append(
            {
                "epoch": epoch_int,
                "timestamp": timestamp,
                "train_loss": last_train_loss_by_epoch.get(epoch_int, ""),
                "eval_loss": log_item.get("eval_loss", ""),
                "train_time_sec": round(time_callback.get_epoch_train_time(epoch_int), 2),
                "eval_accuracy": log_item.get("eval_accuracy", ""),
                "eval_f1_macro": log_item.get("eval_f1_macro", ""),
                "eval_mae": log_item.get("eval_mae", ""),
                "num_layers_replace": num_layers_to_replace,
                "num_layers_add": num_layers_to_add,
                "num_layers_remove": num_layers_to_remove,
            }
        )

    return epoch_metrics_list


def collect_final_metrics(
    trainer,
    train_metrics: dict[str, Any],
    eval_metrics: dict[str, Any],
    training_duration: float,
    avg_time: float,
    std_time: float,
    classification_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Собирает итоговый словарь метрик для сохранения в CSV/логи."""
    epochs_completed = getattr(trainer.state, "epoch", None)
    if epochs_completed is not None:
        epochs_completed = round(epochs_completed, 2)

    best_metric = trainer.state.best_metric if hasattr(trainer.state, "best_metric") else None
    final_train_loss = train_metrics.get("train_loss")
    final_eval_loss = eval_metrics.get("eval_loss")
    final_eval_accuracy = eval_metrics.get("eval_accuracy")
    final_eval_f1_macro = eval_metrics.get("eval_f1_macro")
    final_eval_f1_weighted = eval_metrics.get("eval_f1_weighted")
    final_eval_mae = eval_metrics.get("eval_mae")
    final_eval_precision = eval_metrics.get("eval_precision_macro")
    final_eval_recall = eval_metrics.get("eval_recall_macro")

    write_log(
        f"train_loss={final_train_loss}, eval_loss={final_eval_loss}, "
        f"accuracy={final_eval_accuracy}, f1_macro={final_eval_f1_macro}, "
        f"mae={final_eval_mae}, train_time={training_duration:.2f}s"
    )

    return {
        "train_loss": final_train_loss if final_train_loss is not None else "",
        "eval_loss": final_eval_loss if final_eval_loss is not None else "",
        "total_training_time_sec": round(training_duration, 2),
        "avg_epoch_time_sec": round(avg_time, 2) if avg_time else 0,
        "std_epoch_time_sec": round(std_time, 2) if std_time else 0,
        "eval_accuracy": final_eval_accuracy if final_eval_accuracy is not None else "",
        "eval_f1_macro": final_eval_f1_macro if final_eval_f1_macro is not None else "",
        "eval_f1_weighted": final_eval_f1_weighted if final_eval_f1_weighted is not None else "",
        "eval_mae": final_eval_mae if final_eval_mae is not None else "",
        "eval_precision_macro": final_eval_precision if final_eval_precision is not None else "",
        "eval_recall_macro": final_eval_recall if final_eval_recall is not None else "",
        "epochs_completed": epochs_completed if epochs_completed is not None else "",
        "best_metric": best_metric if best_metric is not None else "",
        **classification_metrics,
    }
