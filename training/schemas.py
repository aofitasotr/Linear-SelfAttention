from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class BertModelConfig:
    """Структурированное описание BERT-конфига эксперимента."""

    pretrained_model_name: str
    num_hidden_layers: int
    hidden_size: int
    num_attention_heads: int
    intermediate_size: int
    num_labels: int
    max_position_embeddings: int
    hidden_dropout_prob: float
    attention_probs_dropout_prob: float
    problem_type: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrainingRuntimeConfig:
    """Структурированное описание training-аргументов из JSON-конфига."""

    optim: str
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    max_grad_norm: float
    learning_rate: float
    warmup_ratio: float
    lr_scheduler_type: str
    weight_decay: float
    eval_strategy: str
    save_strategy: str
    group_by_length: bool
    load_best_model_at_end: bool
    metric_for_best_model: str
    greater_is_better: bool
    fp16: bool
    bf16: bool
    save_total_limit: int
    report_to: str
    dataloader_num_workers: int
    dataloader_pin_memory: bool
    disable_tqdm: bool
    early_stopping_patience: int
    early_stopping_threshold: float
    class_names: list[str]
    seed: int
    data_seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Полный именованный конфиг одного эксперимента из `config.json`."""

    name: str
    training_args: TrainingRuntimeConfig
    bert_config: BertModelConfig

    @classmethod
    def from_dict(cls, name: str, payload: dict[str, Any]) -> "ExperimentConfig":
        return cls(
            name=name,
            training_args=TrainingRuntimeConfig(**payload["training_args"]),
            bert_config=BertModelConfig(**payload["bert_config"]),
        )


@dataclass(slots=True)
class ModelArtifacts:
    """Связка модели и токенизатора, возвращаемая фабрикой моделей."""

    model: Any
    tokenizer: Any
