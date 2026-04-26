from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from transformers import BertForSequenceClassification, DataCollatorWithPadding

from logging_utils import write_log

from .callbacks import CustomTrainer, DetailedMemoryCallback, EpochTrainingTimeCallback
from .config import build_bert_config, build_training_args
from .model_factory import build_loss, wrap_with_custom_attention
from .schemas import BertModelConfig, ModelArtifacts, TrainingRuntimeConfig


@dataclass(frozen=True, slots=True)
class FoldResult:
    fold: int
    eval_accuracy: float | None
    eval_f1_macro: float | None
    eval_loss: float | None
    train_loss: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "eval_accuracy": self.eval_accuracy,
            "eval_f1_macro": self.eval_f1_macro,
            "eval_loss": self.eval_loss,
            "train_loss": self.train_loss,
        }


def _config_value(config: dict[str, Any] | TrainingRuntimeConfig, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _compute_cv_summary(fold_results: list[FoldResult], n_folds: int) -> dict[str, float]:
    accuracies = [result.eval_accuracy for result in fold_results if result.eval_accuracy is not None]
    f1_scores = [result.eval_f1_macro for result in fold_results if result.eval_f1_macro is not None]
    return {
        "cv_accuracy_mean": float(np.mean(accuracies)) if accuracies else 0.0,
        "cv_accuracy_std": float(np.std(accuracies)) if accuracies else 0.0,
        "cv_f1_macro_mean": float(np.mean(f1_scores)) if f1_scores else 0.0,
        "cv_f1_macro_std": float(np.std(f1_scores)) if f1_scores else 0.0,
        "n_folds": n_folds,
    }


def run_cross_validation(
    train_dataset,
    bert_loader: ModelArtifacts,
    base_model,
    run_log_dir: str,
    training_args_config: dict[str, Any] | TrainingRuntimeConfig,
    bert_config_params: dict[str, Any] | BertModelConfig,
    experiment_id: str,
    num_layers_to_replace: int,
    num_layers_to_add: int,
    num_layers_to_remove: int,
    n_folds: int,
    attention_class,
    random_state: int,
    compute_metrics_fn,
) -> tuple[ModelArtifacts, object, dict[str, Any]]:
    """Выполняет стратифицированную кросс-валидацию поверх train-части."""
    del num_layers_to_remove

    write_log(f"\n{'=' * 60}")
    write_log(f"Запуск {n_folds}-fold кросс-валидации на train датасете")
    write_log(f"Размер train датасета: {len(train_dataset)}")
    write_log("=" * 60)

    labels = [train_dataset[index]["labels"] for index in range(len(train_dataset))]
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_results: list[FoldResult] = []

    for fold_index, (train_idx, val_idx) in enumerate(splitter.split(np.zeros(len(labels)), labels), start=1):
        write_log(f"\n{'=' * 60}")
        write_log(f"FOLD {fold_index}/{n_folds}")
        write_log("=" * 60)

        train_subset = torch.utils.data.Subset(train_dataset, train_idx)
        val_subset = torch.utils.data.Subset(train_dataset, val_idx)

        fold_model = BertForSequenceClassification(build_bert_config(bert_config_params))
        torch.manual_seed(random_state + fold_index - 1)

        fold_labels = [train_subset[index]["labels"] for index in range(len(train_subset))]
        fold_model = wrap_with_custom_attention(
            fold_model,
            num_layers_to_replace=num_layers_to_replace,
            num_layers_to_add=num_layers_to_add,
            attention_class=attention_class,
        )
        fold_model.loss_fct = build_loss(fold_labels, base_model.device)

        fold_log_dir = os.path.join(run_log_dir, f"fold_{fold_index}")
        os.makedirs(fold_log_dir, exist_ok=True)

        training_args = build_training_args(
            training_args_config=training_args_config,
            output_dir=os.path.join(fold_log_dir, "checkpoints"),
            logging_dir=fold_log_dir,
            train_dataset_size=len(train_subset),
            seed=_config_value(training_args_config, "seed", random_state),
        )

        trainer = CustomTrainer(
            model=fold_model,
            args=training_args,
            train_dataset=train_subset,
            eval_dataset=val_subset,
            compute_metrics=compute_metrics_fn,
            data_collator=DataCollatorWithPadding(
                tokenizer=bert_loader.tokenizer,
                padding=True,
                pad_to_multiple_of=8,
            ),
            callbacks=[
                DetailedMemoryCallback(),
                EpochTrainingTimeCallback(
                    system_warmup_epochs=_config_value(training_args_config, "system_warmup_epochs", 2)
                ),
            ],
        )

        write_log(f"Начало обучения фолда {fold_index}...")
        train_output = trainer.train()
        eval_metrics = trainer.evaluate()

        result = FoldResult(
            fold=fold_index,
            eval_accuracy=eval_metrics.get("eval_accuracy"),
            eval_f1_macro=eval_metrics.get("eval_f1_macro"),
            eval_loss=eval_metrics.get("eval_loss"),
            train_loss=train_output.metrics.get("train_loss") if hasattr(train_output, "metrics") else None,
        )
        fold_results.append(result)
        write_log(
            f"Фолд {fold_index} завершён. Accuracy: {result.eval_accuracy:.4f}, "
            f"F1: {result.eval_f1_macro:.4f}"
        )

    final_metrics = _compute_cv_summary(fold_results, n_folds=n_folds)
    write_log(f"\n{'=' * 60}")
    write_log(f"РЕЗУЛЬТАТЫ {n_folds}-FOLD КРОСС-ВАЛИДАЦИИ")
    write_log(f"Accuracy: {final_metrics['cv_accuracy_mean']:.4f} ± {final_metrics['cv_accuracy_std']:.4f}")
    write_log(f"F1-macro: {final_metrics['cv_f1_macro_mean']:.4f} ± {final_metrics['cv_f1_macro_std']:.4f}")
    write_log("=" * 60)

    seed = _config_value(training_args_config, "seed")
    run_params = {
        "experiment_id": experiment_id,
        "num_layers_to_replace": num_layers_to_replace,
        "num_layers_to_add": num_layers_to_add,
        "num_layers_to_remove": num_layers_to_remove,
        "train_dataset_size": len(train_dataset),
        "n_folds": n_folds,
    }
    if seed is not None:
        run_params["seed"] = seed

    return bert_loader, base_model, {
        "run_params": run_params,
        "training_args": training_args_config,
        "bert_config": bert_config_params,
        "final_metrics": final_metrics,
        "fold_results": [result.to_dict() for result in fold_results],
    }
