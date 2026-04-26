from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, BertForSequenceClassification

from custom_attention import (
    BertWithCustomAttention,
    LinearContextAttention,
    LinearContextAttentionDilated,
    LinearContextAttentionLocalWindow,
    LinearContextAttentionPosEnc,
    LinearContextAttentionWeighted,
)
from .config import build_bert_config
from .schemas import BertModelConfig, ModelArtifacts


ATTENTION_CLASSES: dict[str, type[nn.Module]] = {
    "base": LinearContextAttention,
    "pos-enc": LinearContextAttentionPosEnc,
    "dilated": LinearContextAttentionDilated,
    "local-window": LinearContextAttentionLocalWindow,
    "weighted": LinearContextAttentionWeighted,
}


def resolve_attention_class(attention_type: str) -> type[nn.Module]:
    """Разрешает строковый идентификатор внимания в конкретный класс реализации."""
    attention_class = ATTENTION_CLASSES.get(attention_type)
    if attention_class is None:
        raise ValueError(
            f"Неизвестный тип внимания: {attention_type}. "
            f"Допустимые варианты: {list(ATTENTION_CLASSES.keys())}"
        )
    return attention_class


def build_base_model(
    bert_config_params: dict[str, Any] | BertModelConfig,
) -> tuple[ModelArtifacts, torch.nn.Module]:
    """Создаёт базовую модель BERT и синхронизированный с ней токенизатор."""
    config = build_bert_config(bert_config_params)
    base_model = BertForSequenceClassification(config)

    pretrained_model_name = (
        bert_config_params.pretrained_model_name
        if isinstance(bert_config_params, BertModelConfig)
        else bert_config_params["pretrained_model_name"]
    )
    max_position_embeddings = (
        bert_config_params.max_position_embeddings
        if isinstance(bert_config_params, BertModelConfig)
        else bert_config_params["max_position_embeddings"]
    )

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    tokenizer.model_max_length = max_position_embeddings
    tokenizer.init_kwargs["model_max_length"] = max_position_embeddings

    return ModelArtifacts(model=base_model, tokenizer=tokenizer), base_model


def apply_layer_removal(
    base_model: torch.nn.Module,
    num_layers_to_remove: int,
) -> tuple[torch.nn.Module, int, int]:
    """Удаляет верхние encoder-слои BERT, сохраняя согласованность внутренних конфигов."""
    if num_layers_to_remove <= 0:
        return base_model, 0, len(base_model.bert.encoder.layer)

    total_layers = len(base_model.bert.encoder.layer)
    num_layers_to_remove = max(0, min(int(num_layers_to_remove), total_layers - 1))
    num_layers_to_keep = total_layers - num_layers_to_remove

    if num_layers_to_remove > 0:
        base_model.bert.encoder.layer = base_model.bert.encoder.layer[:num_layers_to_keep]
        base_model.config.num_hidden_layers = num_layers_to_keep
        if hasattr(base_model.bert, "config"):
            base_model.bert.config.num_hidden_layers = num_layers_to_keep
        if hasattr(base_model.bert.encoder, "config"):
            base_model.bert.encoder.config.num_hidden_layers = num_layers_to_keep

    return base_model, num_layers_to_remove, num_layers_to_keep


def wrap_with_custom_attention(
    base_model: torch.nn.Module,
    num_layers_to_replace: int,
    num_layers_to_add: int,
    attention_class: type[nn.Module],
) -> torch.nn.Module:
    """Оборачивает BERT кастомным вниманием только если это действительно требуется."""
    if num_layers_to_replace == 0 and num_layers_to_add == 0:
        return base_model

    return BertWithCustomAttention(
        base_model,
        num_layers_to_replace=num_layers_to_replace,
        num_layers_to_add=num_layers_to_add,
        attention_class=attention_class,
    )


def build_loss(labels: np.ndarray | list[int], device: torch.device | str) -> nn.CrossEntropyLoss:
    """Создаёт взвешенную cross-entropy по распределению классов в train-части."""
    labels_np = np.asarray(labels)
    class_weights_np = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels_np),
        y=labels_np,
    )
    class_weights = torch.tensor(class_weights_np, dtype=torch.float, device=device)
    return nn.CrossEntropyLoss(weight=class_weights)
