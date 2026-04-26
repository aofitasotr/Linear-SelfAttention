import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import DataCollatorWithPadding, EarlyStoppingCallback, TrainerCallback

from dataset_utils import ReviewDataset, prepare_datasets
from logging_utils import set_log_file, write_log
from .callbacks import (
    CustomTrainer,
    DetailedMemoryCallback,
    EpochTrainingTimeCallback,
    PositionalParamsCallback,
    get_system_info,
    log_detailed_memory,
    reset_gpu_memory_stats,
    set_seed,
)
from .config import build_training_args, load_config_from_json
from .cv import run_cross_validation
from .eval import evaluate_by_length_bins, log_classification_report, save_all_results_to_csv
from .model_factory import (
    apply_layer_removal,
    build_base_model,
    build_loss,
    resolve_attention_class,
    wrap_with_custom_attention,
)
from .schemas import BertModelConfig, ModelArtifacts, TrainingRuntimeConfig
from .text_metrics import (
    build_compute_metrics as build_text_compute_metrics,
    build_epoch_metrics as build_text_epoch_metrics,
    collect_final_metrics as collect_text_final_metrics,
)

@dataclass
class TrainRunConfig:
    """Параметры одного текстового train/eval запуска."""

    train_dataset_path: str
    val_dataset_path: str
    num_layers_to_replace: int
    num_layers_to_add: int
    num_layers_to_remove: int
    bert_loader: ModelArtifacts
    run_log_dir: str
    training_args_config: Dict[str, Any] | TrainingRuntimeConfig
    bert_config_params: Dict[str, Any] | BertModelConfig
    class_names: Optional[list]
    experiment_id: str
    output_path: str
    n_folds: int = 1
    attention_class: Any = None
    random_state: int = 42


def _build_compute_metrics(class_names):
    """Возвращает функцию расчёта метрик для текстовой классификации."""
    return build_text_compute_metrics(class_names)
    def compute_metrics_fn(p):
        preds = p.predictions.argmax(-1)
        labels = p.label_ids
        from sklearn.metrics import precision_score, recall_score

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


def _build_log_loss_callback():
    """Создаёт callback для записи train loss в лог на шагах обучения."""
    class LogLossCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs and "eval_loss" not in logs:
                loss_val = logs["loss"]
                epoch = logs.get("epoch", 0)
                step = logs.get("step", 0)
                write_log(f"[Epoch {epoch:.2f}, Step {step}] loss={loss_val:.4f}")

    return LogLossCallback()


def _load_train_dataset(run_config: TrainRunConfig):
    """Загружает train-датасет отзывов для текущего запуска."""
    if not os.path.exists(run_config.train_dataset_path):
        write_log(f"Train dataset {run_config.train_dataset_path} не найден.")
        return None

    return ReviewDataset(
        csv_path=run_config.train_dataset_path,
        tokenizer=run_config.bert_loader.tokenizer,
        max_length=run_config.bert_config_params.get("max_position_embeddings", 768),
    )


def _load_eval_dataset(run_config: TrainRunConfig):
    """Загружает validation-датасет либо переиспользует train-файл как fallback."""
    val_dataset_path = run_config.val_dataset_path
    if not os.path.exists(val_dataset_path):
        write_log(f"Val dataset {val_dataset_path} не найден. Используем train_dataset для оценки.")
        val_dataset_path = run_config.train_dataset_path

    return ReviewDataset(
        csv_path=val_dataset_path,
        tokenizer=run_config.bert_loader.tokenizer,
        max_length=run_config.bert_config_params.get("max_position_embeddings", 768),
    )


def _build_train_components(run_config: TrainRunConfig, train_dataset, eval_dataset, base_model, compute_metrics_fn):
    """Собирает trainer, callbacks и модель для обычного train/val сценария."""
    train_labels = np.array([train_dataset[i]["labels"] for i in range(len(train_dataset))]).flatten()
    loss_fct = build_loss(train_labels, base_model.device)
    write_log(f"Веса классов для loss: {loss_fct.weight}")

    if run_config.num_layers_to_replace == 0 and run_config.num_layers_to_add == 0:
        write_log("Все параметры replace/add = 0: используем BERT без изменений")
        model_to_train = base_model
    else:
        model_to_train = wrap_with_custom_attention(
            base_model,
            num_layers_to_replace=run_config.num_layers_to_replace,
            num_layers_to_add=run_config.num_layers_to_add,
            attention_class=run_config.attention_class,
        )
    model_to_train.loss_fct = loss_fct

    checkpoint_dir = os.path.join(run_config.run_log_dir, "checkpoints")
    logging_dir = run_config.run_log_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    training_args = build_training_args(
        run_config.training_args_config,
        output_dir=checkpoint_dir,
        logging_dir=logging_dir,
        train_dataset_size=len(train_dataset),
        include_group_by_length=True,
    )
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=run_config.training_args_config["early_stopping_patience"],
        early_stopping_threshold=run_config.training_args_config["early_stopping_threshold"],
    )
    data_collator = DataCollatorWithPadding(
        tokenizer=run_config.bert_loader.tokenizer,
        padding=True,
        pad_to_multiple_of=8,
    )
    detailed_memory_callback = DetailedMemoryCallback()
    time_callback = EpochTrainingTimeCallback(
        system_warmup_epochs=run_config.training_args_config.get("system_warmup_epochs", 2)
    )
    positional_params_callback = PositionalParamsCallback()

    trainer = CustomTrainer(
        model=model_to_train,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics_fn,
        data_collator=data_collator,
        callbacks=[
            _build_log_loss_callback(),
            positional_params_callback,
            early_stopping,
            detailed_memory_callback,
            time_callback,
        ],
    )

    return {
        "trainer": trainer,
        "model_to_train": model_to_train,
        "training_args": training_args,
        "detailed_memory_callback": detailed_memory_callback,
        "time_callback": time_callback,
        "positional_params_callback": positional_params_callback,
    }


def _run_single_split_training(run_config: TrainRunConfig, train_dataset, eval_dataset, base_model, compute_metrics_fn):
    """Выполняет один train/val прогон без кросс-валидации."""
    components = _build_train_components(
        run_config=run_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        base_model=base_model,
        compute_metrics_fn=compute_metrics_fn,
    )
    trainer = components["trainer"]
    model_to_train = components["model_to_train"]
    training_args = components["training_args"]
    detailed_memory_callback = components["detailed_memory_callback"]
    time_callback = components["time_callback"]
    positional_params_callback = components["positional_params_callback"]

    reset_gpu_memory_stats()
    log_detailed_memory("до обучения")

    write_log("Начало обучения...")
    training_start = time.time()
    reset_gpu_memory_stats()
    train_output = trainer.train()
    training_duration = time.time() - training_start

    write_log("Обучение завершено!")
    write_log(f"Общее время обучения ({training_args.num_train_epochs} эпох): {training_duration:.2f} сек")

    avg_time, std_time, warmup_epochs, _stable_epochs = time_callback.get_stable_epoch_stats()
    if avg_time:
        write_log(
            f"Статистика времени обучения:\n"
            f"  Прогрев: первые {warmup_epochs} эпох (исключены из расчёта)\n"
            f"  Среднее время эпохи: {avg_time:.2f} ± {std_time:.2f} сек\n"
            f"  Относительное отклонение: {(std_time / avg_time * 100):.1f}%"
        )
    else:
        write_log("Статистика времени обучения: недостаточно данных")

    log_detailed_memory("после обучения")

    device = next(model_to_train.parameters()).device
    classification_metrics = log_classification_report(
        model=model_to_train,
        eval_dataset=eval_dataset,
        tokenizer=run_config.bert_loader.tokenizer,
        class_names=run_config.class_names or ["1", "2", "3", "4", "5"],
        device=device,
    )

    reset_gpu_memory_stats()
    eval_metrics = trainer.evaluate()
    log_detailed_memory("во время валидации")

    train_metrics = getattr(train_output, "metrics", {}) if train_output is not None else {}
    final_metrics = collect_text_final_metrics(
        trainer=trainer,
        train_metrics=train_metrics,
        eval_metrics=eval_metrics,
        training_duration=training_duration,
        avg_time=avg_time,
        std_time=std_time,
        classification_metrics=classification_metrics,
    )
    epoch_metrics_list = build_text_epoch_metrics(
        trainer=trainer,
        time_callback=time_callback,
        num_layers_to_replace=run_config.num_layers_to_replace,
        num_layers_to_add=run_config.num_layers_to_add,
        num_layers_to_remove=run_config.num_layers_to_remove,
    )

    write_log(
        f"Параметры обучения: replace={run_config.num_layers_to_replace}, "
        f"add={run_config.num_layers_to_add}, remove={run_config.num_layers_to_remove}, "
        f"learning_rate={training_args.learning_rate}, epochs={training_args.num_train_epochs}"
    )
    write_log(f"Размер train: {len(train_dataset)}, val: {len(eval_dataset)}")
    if trainer.state.best_metric is not None:
        write_log(f"Лучшая метрика: {trainer.state.best_metric}")

    trainer.save_model("./custom_bert_finetuned")

    return {
        "model_to_train": model_to_train,
        "final_metrics": final_metrics,
        "epoch_metrics": epoch_metrics_list,
        "memory_stats": detailed_memory_callback.get_memory_stats(),
        "positional_params": positional_params_callback.positional_params,
    }


def _build_epoch_metrics(trainer, time_callback, run_config: TrainRunConfig):
    return build_text_epoch_metrics(
        trainer=trainer,
        time_callback=time_callback,
        num_layers_to_replace=run_config.num_layers_to_replace,
        num_layers_to_add=run_config.num_layers_to_add,
        num_layers_to_remove=run_config.num_layers_to_remove,
    )
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_history = getattr(trainer.state, "log_history", [])
    last_train_loss_by_epoch = {}

    for log_item in log_history:
        if "loss" in log_item and "eval_loss" not in log_item:
            epoch_val = log_item.get("epoch")
            if epoch_val is None:
                continue
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
                "num_layers_replace": run_config.num_layers_to_replace,
                "num_layers_add": run_config.num_layers_to_add,
                "num_layers_remove": run_config.num_layers_to_remove,
            }
        )

    return epoch_metrics_list


def _collect_final_metrics(
    trainer,
    train_metrics: Dict[str, Any],
    eval_metrics: Dict[str, Any],
    training_duration: float,
    avg_time: float,
    std_time: float,
    classification_metrics: Dict[str, Any],
):
    return collect_text_final_metrics(
        trainer=trainer,
        train_metrics=train_metrics,
        eval_metrics=eval_metrics,
        training_duration=training_duration,
        avg_time=avg_time,
        std_time=std_time,
        classification_metrics=classification_metrics,
    )
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


def train_custom_bert(run_config: TrainRunConfig) -> Tuple[object, torch.nn.Module, Dict[str, Any]]:
    """Главный исполняющий сценарий для обучения текстовой модели."""
    set_seed(run_config.random_state)
    base_model = run_config.bert_loader.model

    write_log(
        f"Конфиг модели: num_labels={base_model.config.num_labels}, "
        f"hidden_size={base_model.config.hidden_size}, "
        f"num_attention_heads={base_model.config.num_attention_heads}"
    )

    base_model, removed_layers, kept_layers = apply_layer_removal(base_model, run_config.num_layers_to_remove)
    if removed_layers > 0:
        write_log(
            f"Удаление слоёв: всего было {removed_layers + kept_layers}, "
            f"удалено {removed_layers}, осталось {kept_layers}"
        )

    train_dataset = _load_train_dataset(run_config)
    if train_dataset is None:
        return run_config.bert_loader, base_model, {}

    compute_metrics_fn = build_text_compute_metrics(run_config.class_names)

    if run_config.n_folds > 1:
        bert_loader, base_model, all_results = run_cross_validation(
            train_dataset=train_dataset,
            bert_loader=run_config.bert_loader,
            base_model=base_model,
            run_log_dir=run_config.run_log_dir,
            training_args_config=run_config.training_args_config,
            bert_config_params=run_config.bert_config_params,
            experiment_id=run_config.experiment_id,
            num_layers_to_replace=run_config.num_layers_to_replace,
            num_layers_to_add=run_config.num_layers_to_add,
            num_layers_to_remove=run_config.num_layers_to_remove,
            n_folds=run_config.n_folds,
            attention_class=run_config.attention_class,
            random_state=run_config.random_state,
            compute_metrics_fn=compute_metrics_fn,
        )
        all_results["system_info"] = get_system_info()
        return bert_loader, base_model, all_results

    write_log("Запуск обычного обучения с train/val split")
    eval_dataset = _load_eval_dataset(run_config)
    single_run = _run_single_split_training(
        run_config=run_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        base_model=base_model,
        compute_metrics_fn=compute_metrics_fn,
    )

    seed = run_config.training_args_config.get("seed")
    run_params = {
        "experiment_id": run_config.experiment_id,
        "num_layers_to_replace": run_config.num_layers_to_replace,
        "num_layers_to_add": run_config.num_layers_to_add,
        "num_layers_to_remove": run_config.num_layers_to_remove,
        "train_dataset_size": len(train_dataset),
        "eval_dataset_size": len(eval_dataset),
    }
    if seed is not None:
        run_params["seed"] = seed

    all_results = {
        "run_params": run_params,
        "training_args": run_config.training_args_config,
        "bert_config": run_config.bert_config_params,
        "final_metrics": single_run["final_metrics"],
        "epoch_metrics": single_run["epoch_metrics"],
        "memory_stats": single_run["memory_stats"],
        "system_info": get_system_info(),
        "positional_params": single_run["positional_params"],
    }

    return run_config.bert_loader, single_run["model_to_train"], all_results


def custom_model_train(
    train_path: str,
    output_path: str,
    log_path: str,
    num_layers_to_replace: int = 0,
    num_layers_to_add: int = 0,
    num_layers_to_remove: int = 0,
    config_json_path: str = None,
    config_name: str = None,
    n_folds: int = 1,
    attention_type: str = "dilated",
    random_state: int = 42,
):
    """Высокоуровневый orchestration-метод обучения и оценки текстовой модели."""
    set_seed(random_state)

    base_log_dir = f"./logs/{config_name}"
    os.makedirs(base_log_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    experiment_id = f"{config_name}_{timestamp}"
    run_name = (
        f"run_replace{num_layers_to_replace}_add{num_layers_to_add}_"
        f"remove{num_layers_to_remove}_{timestamp}"
    )
    run_log_dir = os.path.join(base_log_dir, run_name)
    os.makedirs(run_log_dir, exist_ok=True)

    log_path = os.path.join(run_log_dir, os.path.basename(log_path))
    output_path = os.path.join(run_log_dir, os.path.basename(output_path))
    set_log_file(log_path)

    if not config_json_path or not config_name:
        raise ValueError("Необходимо указать config_json_path и config_name")

    write_log(f"Загрузка конфигурации '{config_name}' из {config_json_path}")
    experiment_config = load_config_from_json(config_json_path, config_name)
    training_args_config = experiment_config.training_args.to_dict()
    bert_config_params = experiment_config.bert_config.to_dict()
    class_names = experiment_config.training_args.class_names

    write_log("\n" + "=" * 60)
    write_log("ЗАГРУЖЕННАЯ КОНФИГУРАЦИЯ:")
    write_log("=" * 60)
    write_log("\n--- Training Arguments ---")
    for key, value in training_args_config.items():
        write_log(f"  {key}: {value}")
    write_log("\n--- Bert Config ---")
    for key, value in bert_config_params.items():
        write_log(f"  {key}: {value}")
    if class_names:
        write_log("\n--- Class Names ---")
        write_log(f"  {class_names}")
    write_log("=" * 60 + "\n")

    write_log(f"Запуск обучения: train={train_path}, output={output_path}, log={log_path}")
    write_log(f"Директория запуска: {run_log_dir}")
    write_log(
        f"Параметры слоёв: replace={num_layers_to_replace}, "
        f"add={num_layers_to_add}, remove={num_layers_to_remove}"
    )
    write_log(f"Используемая конфигурация: {config_name}")
    write_log(
        f"Режим: {'кросс-валидация' if n_folds > 1 else 'обычное обучение'} "
        f"с n_folds={n_folds}"
    )

    attention_class = resolve_attention_class(attention_type)
    write_log(f"Тип внимания: {attention_type} -> {attention_class.__name__}")

    datasets_dir = "datasets"
    expected_files = [
        os.path.join(datasets_dir, f)
        for f in ("train_dataset.csv", "val_dataset.csv", "test_dataset.csv")
    ]
    datasets_present = all(os.path.exists(p) for p in expected_files)
    if not datasets_present:
        write_log("Папка 'datasets' с train/val/test не найдена. Попытка создать из входного файла.")
        if os.path.exists(train_path):
            try:
                prepare_datasets(train_path)
            except Exception as e:
                write_log(f"Не удалось создать датасеты из {train_path}: {e}")
                write_log("Прерывание: невозможно продолжить без датасетов.")
                return
        else:
            write_log(f"Входной файл {train_path} не найден. Прерывание.")
            return
        train_path = os.path.join(datasets_dir, "train_dataset.csv")

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(output_path):
        with open(output_path, "a", encoding="utf-8"):
            pass

    if "train" in os.path.basename(train_path):
        val_path = train_path.replace("train", "val")
        test_path = train_path.replace("train", "test")
    else:
        val_path = "datasets/val_dataset.csv"
        test_path = "datasets/test_dataset.csv"

    model_container, base_model = build_base_model(bert_config_params)
    num_params = sum(p.numel() for p in base_model.parameters())
    write_log(f"Размер модели: {num_params / 1e6:.1f}M параметров")

    embedding_mean = base_model.bert.embeddings.word_embeddings.weight.mean().item()
    write_log(
        f"Проверка инициализации: среднее эмбеддингов = {embedding_mean:.6f} "
        f"(ожидаемо ≈ 0.0)"
    )
    pos_emb_size = base_model.bert.embeddings.position_embeddings.num_embeddings
    write_log(f"Размер позиционных эмбеддингов: {pos_emb_size}")

    run_config = TrainRunConfig(
        train_dataset_path=train_path,
        val_dataset_path=val_path,
        num_layers_to_replace=num_layers_to_replace,
        num_layers_to_add=num_layers_to_add,
        num_layers_to_remove=num_layers_to_remove,
        bert_loader=model_container,
        run_log_dir=run_log_dir,
        training_args_config=training_args_config,
        bert_config_params=bert_config_params,
        class_names=class_names,
        experiment_id=experiment_id,
        output_path=output_path,
        n_folds=n_folds,
        attention_class=attention_class,
        random_state=random_state,
    )
    bert_loader, model_trained, all_results = train_custom_bert(run_config)

    test_length_metrics = None
    if n_folds == 1 and test_path and os.path.exists(test_path):
        test_dataset = ReviewDataset(
            csv_path=test_path,
            tokenizer=model_container.tokenizer,
            max_length=bert_config_params["max_position_embeddings"],
        )
        device = next(model_trained.parameters()).device
        test_length_metrics = evaluate_by_length_bins(
            model=model_trained,
            dataset=test_dataset,
            tokenizer=model_container.tokenizer,
            class_names=class_names,
            device=device,
            max_len=bert_config_params["max_position_embeddings"],
            step=128,
        )
        write_log("Оценка на тесте по группам длины завершена.")

    save_all_results_to_csv(
        output_path=output_path,
        run_params=all_results["run_params"],
        training_args_config=training_args_config,
        bert_config_params=bert_config_params,
        final_metrics=all_results["final_metrics"],
        epoch_metrics=all_results.get("epoch_metrics", []),
        memory_stats=all_results.get("memory_stats", {}),
        system_info=all_results["system_info"],
        test_length_metrics=test_length_metrics,
        attention_type=attention_type,
    )

    return bert_loader, model_trained, all_results
