from __future__ import annotations

import inspect
import json
from typing import Any

import torch
from transformers import BertConfig, TrainingArguments

from .schemas import BertModelConfig, ExperimentConfig, TrainingRuntimeConfig


TRAINING_ARGUMENTS_PARAMS = set(inspect.signature(TrainingArguments.__init__).parameters)


def load_config_from_json(json_path: str, config_name: str) -> ExperimentConfig:
    """Загружает именованный конфиг эксперимента из JSON в dataclass-структуры."""
    with open(json_path, "r", encoding="utf-8") as file:
        all_configs = json.load(file)

    if config_name not in all_configs:
        raise KeyError(
            f"Конфигурация '{config_name}' не найдена в {json_path}. "
            f"Доступные варианты: {list(all_configs.keys())}"
        )

    payload = all_configs[config_name]
    if "training_args" not in payload or "bert_config" not in payload:
        raise ValueError(
            f"Конфигурация '{config_name}' должна содержать разделы "
            "'training_args' и 'bert_config'."
        )

    return ExperimentConfig.from_dict(config_name, payload)


def _coerce_bert_config(config: dict[str, Any] | BertModelConfig) -> BertModelConfig:
    return config if isinstance(config, BertModelConfig) else BertModelConfig(**config)


def _coerce_training_config(
    config: dict[str, Any] | TrainingRuntimeConfig,
) -> TrainingRuntimeConfig:
    return config if isinstance(config, TrainingRuntimeConfig) else TrainingRuntimeConfig(**config)


def build_bert_config(bert_config_params: dict[str, Any] | BertModelConfig) -> BertConfig:
    """Строит `transformers.BertConfig` из типизированного описания эксперимента."""
    config = _coerce_bert_config(bert_config_params)
    return BertConfig.from_pretrained(
        config.pretrained_model_name,
        num_hidden_layers=config.num_hidden_layers,
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        num_labels=config.num_labels,
        max_position_embeddings=config.max_position_embeddings,
        hidden_dropout_prob=config.hidden_dropout_prob,
        attention_probs_dropout_prob=config.attention_probs_dropout_prob,
        problem_type=config.problem_type,
    )


def build_training_args(
    training_args_config: dict[str, Any] | TrainingRuntimeConfig,
    output_dir: str,
    logging_dir: str,
    train_dataset_size: int,
    include_group_by_length: bool = False,
    seed: int | None = None,
) -> TrainingArguments:
    """Собирает `TrainingArguments` и вычисляет `logging_steps` от размера train-датасета."""
    config = _coerce_training_config(training_args_config)
    batch_size = config.per_device_train_batch_size
    steps_per_epoch = max(1, train_dataset_size // batch_size)
    logging_steps = max(1, int(steps_per_epoch / 10))

    training_args_kwargs = {
        "output_dir": output_dir,
        "optim": config.optim,
        "num_train_epochs": config.num_train_epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "max_grad_norm": config.max_grad_norm,
        "learning_rate": config.learning_rate,
        "warmup_ratio": config.warmup_ratio,
        "lr_scheduler_type": config.lr_scheduler_type,
        "weight_decay": config.weight_decay,
        "eval_strategy": config.eval_strategy,
        "save_strategy": config.save_strategy,
        "load_best_model_at_end": config.load_best_model_at_end,
        "metric_for_best_model": config.metric_for_best_model,
        "greater_is_better": config.greater_is_better,
        "fp16": config.fp16 if config.fp16 is not None else torch.cuda.is_available(),
        "bf16": config.bf16,
        "logging_dir": logging_dir,
        "logging_steps": logging_steps,
        "save_total_limit": config.save_total_limit,
        "report_to": config.report_to,
        "dataloader_num_workers": config.dataloader_num_workers,
        "dataloader_pin_memory": config.dataloader_pin_memory,
        "disable_tqdm": config.disable_tqdm,
    }

    if include_group_by_length and "group_by_length" in TRAINING_ARGUMENTS_PARAMS:
        training_args_kwargs["group_by_length"] = config.group_by_length
    if seed is not None:
        training_args_kwargs["seed"] = seed
        training_args_kwargs["data_seed"] = seed

    return TrainingArguments(**training_args_kwargs)


def build_training_arguments(
    training_args_config: dict[str, Any] | TrainingRuntimeConfig,
    output_dir: str,
    logging_dir: str,
    train_dataset_size: int,
    include_group_by_length: bool = False,
    seed: int | None = None,
) -> TrainingArguments:
    """Совместимый алиас для старого названия фабрики training arguments."""
    return build_training_args(
        training_args_config=training_args_config,
        output_dir=output_dir,
        logging_dir=logging_dir,
        train_dataset_size=train_dataset_size,
        include_group_by_length=include_group_by_length,
        seed=seed,
    )
