from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import BertConfig, BertForSequenceClassification, BertModel

from custom_attention import (
    BertWithCustomAttention,
    LinearContextAttention,
    LinearContextAttentionDilated,
    LinearContextAttentionLocalWindow,
    LinearContextAttentionPosEnc,
    LinearContextAttentionWeighted,
)


SYNTHETIC_ATTENTION_CLASSES = {
    "base": LinearContextAttention,
    "pos-enc": LinearContextAttentionPosEnc,
    "dilated": LinearContextAttentionDilated,
    "local-window": LinearContextAttentionLocalWindow,
    "weighted": LinearContextAttentionWeighted,
}
SYNTHETIC_ATTENTION_TYPES = list(SYNTHETIC_ATTENTION_CLASSES.keys())


@dataclass(frozen=True, slots=True)
class SyntheticModelConfig:
    """Единая типизированная конфигурация синтетических моделей."""

    vocab_size: int = 50
    hidden_size: int = 128
    num_heads: int = 4
    num_layers: int = 2
    max_position_embeddings: int = 128
    dropout_prob: float = 0.1
    pooling_mode: str = "mean"
    num_labels: int | None = None

    @property
    def resolved_num_labels(self) -> int:
        return self.num_labels or self.vocab_size

    def build_bert_config(self) -> BertConfig:
        return BertConfig(
            vocab_size=self.vocab_size + 2,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_heads,
            num_hidden_layers=self.num_layers,
            max_position_embeddings=self.max_position_embeddings,
            hidden_dropout_prob=self.dropout_prob,
            attention_probs_dropout_prob=self.dropout_prob,
            intermediate_size=self.hidden_size * 4,
            num_labels=self.resolved_num_labels,
        )


def _pool_hidden_states(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    pooling_mode: str,
) -> torch.Tensor:
    """Агрегирует последовательность в один вектор для финальной классификации."""
    if pooling_mode == "last-token":
        return hidden_states[:, -1, :]

    if attention_mask is None:
        return hidden_states.mean(dim=1)

    mask_expanded = attention_mask.unsqueeze(-1).float()
    return (hidden_states * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)


class BERTLookup(nn.Module):
    """Базовая синтетическая модель на стандартном BERT без кастомного внимания."""

    def __init__(self, config: SyntheticModelConfig, k: int = 1):
        super().__init__()
        self.config = config
        self.k = k
        self.bert = BertModel(config.build_bert_config())
        self.classifier = nn.Linear(config.hidden_size, config.resolved_num_labels)
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        del labels, kwargs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        pooled = _pool_hidden_states(
            hidden_states=outputs.last_hidden_state,
            attention_mask=attention_mask,
            pooling_mode=self.config.pooling_mode,
        )
        logits = self.classifier(self.dropout(pooled))
        return {"logits": logits, "loss": None}

    def get_num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def get_trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


class SyntheticLookupModel(nn.Module):
    """Синтетическая модель с внедрённым кастомным линейным вниманием."""

    def __init__(
        self,
        config: SyntheticModelConfig,
        attention_class=LinearContextAttentionDilated,
        num_layers_to_replace: int = 2,
        use_original_model: bool = False,
    ):
        super().__init__()
        self.config = config
        self.use_original_model = use_original_model

        base_model = BertForSequenceClassification(config.build_bert_config())
        self.model = BertWithCustomAttention(
            model=base_model,
            num_layers_to_replace=num_layers_to_replace,
            num_layers_to_add=0,
            attention_class=attention_class,
        )

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        del kwargs
        outputs = self.model.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        pooled = _pool_hidden_states(
            hidden_states=outputs.last_hidden_state,
            attention_mask=attention_mask,
            pooling_mode=self.config.pooling_mode,
        )
        logits = self.model.classifier(pooled)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"logits": logits, "loss": loss}

    def get_num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def get_trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def create_synthetic_model(
    attention_type: str = "dilated",
    vocab_size: int = 50,
    hidden_size: int = 128,
    num_heads: int = 4,
    num_layers: int = 2,
    max_position_embeddings: int = 128,
    dropout_prob: float = 0.1,
    num_layers_to_replace: int = 2,
    pooling_mode: str = "mean",
    num_labels: int | None = None,
    **kwargs,
) -> SyntheticLookupModel:
    """Фабрика синтетической модели с кастомным вниманием."""
    del kwargs
    attention_class = SYNTHETIC_ATTENTION_CLASSES.get(attention_type)
    if attention_class is None:
        raise ValueError(f"Неизвестный тип внимания: {attention_type}")

    config = SyntheticModelConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        max_position_embeddings=max_position_embeddings,
        dropout_prob=dropout_prob,
        pooling_mode=pooling_mode,
        num_labels=num_labels,
    )
    return SyntheticLookupModel(
        config=config,
        attention_class=attention_class,
        num_layers_to_replace=num_layers_to_replace,
        use_original_model=False,
    )


def create_original_model(
    vocab_size: int = 50,
    hidden_size: int = 128,
    num_heads: int = 4,
    num_layers: int = 2,
    max_position_embeddings: int = 128,
    dropout_prob: float = 0.1,
    k: int = 1,
    pooling_mode: str = "mean",
    num_labels: int | None = None,
    **kwargs,
) -> BERTLookup:
    """Фабрика базовой synthetic-модели на стандартном BERT."""
    del kwargs
    config = SyntheticModelConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        max_position_embeddings=max_position_embeddings,
        dropout_prob=dropout_prob,
        pooling_mode=pooling_mode,
        num_labels=num_labels,
    )
    return BERTLookup(config=config, k=k)
