"""Кастомные реализации линейного attention-блока для интеграции в BERT."""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutput

from logging_utils import write_log


class LinearContextAttention(nn.Module):
    """Базовый линейный механизм контекстного внимания.

    Модуль не строит классическую матрицу попарных сходств ``QK^T``.
    Для каждого токена контекст формируется как среднее значение векторов ``V``
    по остальным непаддинговым токенам последовательности. Такой вариант служит
    базовой линейной схемой для дальнейших модификаций attention-блока.
    """

    def __init__(self, hidden_size: int, num_attention_heads: int, dropout_prob: float = 0.1, max_position_embeddings: int = 768):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.attention_head_size = hidden_size // num_attention_heads

        assert hidden_size % num_attention_heads == 0

        self.value = nn.Linear(hidden_size, hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self._init_weights()

    def _init_weights(self):
        """Инициализирует линейные слои в стиле BERT."""
        nn.init.normal_(self.value.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.dense.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.value.bias)
        nn.init.zeros_(self.dense.bias)

    def _build_mask(self, attention_mask: torch.Tensor, batch_size: int, seq_len: int, device, dtype):
        """Приводит attention mask к форме ``(B, 1, L, 1)`` для работы с ``V``."""
        if attention_mask is None:
            mask = torch.ones(batch_size, seq_len, device=device, dtype=dtype)
        # Маска Hugging Face BERT обычно приходит в расширенной форме (B, 1, 1, L).
        elif attention_mask.dim() == 4:
            mask = (attention_mask[:, 0, 0, :] > -10000).to(dtype)
        # Возможен и обычный формат (B, L), где 1 — токен, 0 — паддинг.
        elif attention_mask.dim() == 2:
            mask = attention_mask.to(dtype)
        else:
            raise ValueError(f"Unsupported attention_mask dim: {attention_mask.dim()}")
        # Добавляем измерения головы и признаков: (B, 1, L, 1).
        return mask.unsqueeze(1).unsqueeze(3)

    def _project_values(self, hidden_states: torch.Tensor):
        """Проецирует hidden states в ``V`` и раскладывает их по головам."""
        batch_size, seq_len, _ = hidden_states.shape
        # (B, L, hidden_size) -> (B, H, L, head_dim)
        return self.value(hidden_states).view(
            batch_size, seq_len, self.num_attention_heads, self.attention_head_size
        ).permute(0, 2, 1, 3).contiguous()

    def _reshape_context(self, context: torch.Tensor):
        """Собирает выходы голов обратно в размерность ``hidden_size``."""
        batch_size, _, seq_len, _ = context.shape
        return context.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.hidden_size)

    def _finalize_context(self, context: torch.Tensor, context_norm=None):
        """Приводит контекст к выходному формату BERT attention-блока."""
        context = self._reshape_context(context)
        if context_norm is not None:
            context = context_norm(context)
        context = self.dense(context)
        return self.dropout(context)

    def _empty_attentions(self, context: torch.Tensor, seq_len: int):
        """Возвращает нулевую матрицу attention для совместимости с Hugging Face API."""
        return torch.zeros(
            context.size(0),
            self.num_attention_heads,
            seq_len,
            seq_len,
            device=context.device,
            dtype=context.dtype,
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None,
                output_attentions: bool = False, **kwargs):
        """Вычисляет линейный контекст без построения матрицы ``QK^T``."""
        B, seq_len, _ = hidden_states.shape
        V = self._project_values(hidden_states)  # (B, H, L, D)

        # Убираем вклад паддинговых токенов.
        mask = self._build_mask(attention_mask, B, seq_len, V.device, V.dtype)  # (B, 1, L, 1)
        V_masked = V * mask

        # Для каждого токена считаем среднее по всем остальным валидным токенам.
        total_sum = V_masked.sum(dim=2, keepdim=True)      # (B, H, 1, D)
        total_count = mask.sum(dim=2, keepdim=True)        # (B, 1, 1, 1)

        denom = torch.clamp(total_count - 1.0, min=1.0)    # (B, 1, 1, 1)
        context = (total_sum - V_masked) / denom           # (B, H, L, D)
        context = context * mask                           # Обнуляем позиции паддинга.

        context = self._finalize_context(context)

        if output_attentions:
            return context, self._empty_attentions(context, seq_len)
        return context, None


class LinearContextAttentionPosEnc(LinearContextAttention):
    """Линейное контекстное внимание с позиционной модуляцией ``Value``."""

    def __init__(self, hidden_size: int, num_attention_heads: int, dropout_prob: float = 0.1,
                 max_position_embeddings: int = 768):
        super().__init__(hidden_size, num_attention_heads, dropout_prob)

        self.max_position_embeddings = max_position_embeddings

        # Нормализация после объединения голов стабилизирует выход attention-блока.
        self.context_norm = nn.LayerNorm(hidden_size)

        # Синусоидальное позиционное кодирование в размерности одной головы.
        pe = torch.zeros(max_position_embeddings, self.attention_head_size)
        position = torch.arange(0, max_position_embeddings, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.attention_head_size, 2).float() *
            (-math.log(10000.0) / self.attention_head_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

        # Обучаемые множители задают силу позиционной модуляции для каждой головы.
        head_scales = torch.linspace(0.5, 2, num_attention_heads)
        self.head_scales = nn.Parameter(head_scales, requires_grad=True)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None,
                output_attentions: bool = False, **kwargs):
        """Вычисляет контекст после мультипликативной позиционной модуляции ``V``."""
        _ = kwargs
        B, seq_len, _ = hidden_states.shape
        H = self.num_attention_heads
        D = self.attention_head_size

        V = self._project_values(hidden_states)  # (B, H, L, D)

        # Добавляем позиционный сигнал в Value: V' = V * (1 + scale_h * PE).
        pos_emb = self.pe[:seq_len].unsqueeze(0).unsqueeze(0)          # (1, 1, L, D)
        pos_emb = pos_emb * self.head_scales.view(1, H, 1, 1)          # (1, H, L, D)
        V = V * (1.0 + pos_emb)                                        # (B, H, L, D)

        mask = self._build_mask(attention_mask, B, seq_len, V.device, V.dtype)  # (B, 1, L, 1)

        V_masked = V * mask
        total_sum = V_masked.sum(dim=2, keepdim=True)                   # (B, H, 1, D)
        total_count = mask.sum(dim=2, keepdim=True)                     # (B, 1, 1, 1)

        numerator = total_sum - V_masked
        denominator = torch.clamp(total_count - 1.0, min=1.0)
        context = numerator / denominator                               # (B, H, L, D)
        context = context * mask

        context = self._finalize_context(context, context_norm=self.context_norm)

        if output_attentions:
            return context, self._empty_attentions(context, seq_len)
        return context, None


class LinearContextAttentionDilated(LinearContextAttentionPosEnc):
    """Линейное внимание с разреженным отбором токенов по головам."""

    def __init__(self, hidden_size: int, num_attention_heads: int, dropout_prob: float = 0.1,
                 max_position_embeddings: int = 768):
        super().__init__(hidden_size, num_attention_heads, dropout_prob, max_position_embeddings)
        self.register_buffer('dilations', self._compute_dilations(num_attention_heads))
        self.register_buffer('offsets', self._compute_offsets(num_attention_heads))

        write_log(
            f"Разреженное внимание: dilations={self.dilations.tolist()}, offsets={self.offsets.tolist()}"
        )

    def _compute_dilations(self, num_heads):
        """Назначает головам шаги отбора 1, 2, 4, ... группами возрастающего размера."""
        head_indices = torch.arange(num_heads)
        # Группируем головы так, чтобы часть голов работала с плотным контекстом,
        # а часть — с более разреженными позициями.
        max_power = (num_heads + 1).bit_length()
        powers = 2 ** torch.arange(max_power)
        cumsum = torch.cumsum(powers, dim=0)
        group_indices = (head_indices.unsqueeze(1) < cumsum.unsqueeze(0)).float()
        group_ids = torch.argmax(group_indices, dim=1)
        dilations = 2 ** group_ids
        return dilations

    def _compute_offsets(self, num_heads):
        """Подбирает смещения для голов с одинаковым шагом отбора."""
        dilations = self._compute_dilations(num_heads)
        unique_dilations = dilations.unique()
        offsets = torch.zeros(num_heads, dtype=torch.long)
        for dilation in unique_dilations:
            mask = (dilations == dilation)
            if mask.any():
                # Для одного и того же шага разные головы покрывают разные остатки по модулю dilation.
                group_indices = torch.arange(mask.sum().item())
                offsets[mask] = group_indices % dilation.item()
        return offsets

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None,
                output_attentions: bool = False, **kwargs):
        """Вычисляет контекст по head-specific разреженной сетке позиций."""
        _ = kwargs
        B, seq_len, _ = hidden_states.shape
        H = self.num_attention_heads

        V = self._project_values(hidden_states)  # (B, H, L, D)

        # Позиционно модулируем Value перед разреженным агрегированием.
        pos_emb = self.pe[:seq_len].unsqueeze(0).unsqueeze(0)          # (1, 1, L, D)
        pos_emb = pos_emb * self.head_scales.view(1, H, 1, 1)          # (1, H, L, D)
        V = V * (1.0 + pos_emb)                       # (B, H, L, D)

        mask = self._build_mask(attention_mask, B, seq_len, V.device, V.dtype)  # (B, 1, L, 1)

        # d=1 выбирает все позиции, d=2 — позиции через одну, d=4 — каждую четвертую.
        positions = torch.arange(seq_len, device=V.device, dtype=torch.long).view(1, seq_len)
        dilations = self.dilations.to(V.device).view(H, 1)
        offsets = self.offsets.to(V.device).view(H, 1)
        selection_mask = ((positions - offsets) % dilations == 0).to(V.dtype)      # (H, L)
        selection_mask = selection_mask.unsqueeze(0).unsqueeze(-1)                  # (1, H, L, 1)

        # Совмещаем padding mask и разреженную маску конкретной головы.
        mask_expanded = mask.expand(-1, H, -1, -1)                                  # (B, H, L, 1)
        dilated_mask = mask_expanded * selection_mask                                # (B, H, L, 1)

        # Контекст строится только по выбранным валидным позициям.
        V_masked = V * dilated_mask
        total_sum = V_masked.sum(dim=2, keepdim=True)                                # (B, H, 1, D)
        total_count = dilated_mask.sum(dim=2, keepdim=True)                          # (B, H, 1, 1)

        # Исключаем текущий токен, если он попал в выбранную сетку.
        numerator = total_sum - V_masked
        denominator = torch.clamp(total_count - dilated_mask, min=1.0)
        context = numerator / denominator                                            # (B, H, L, D)
        context = context * dilated_mask

        context = self._finalize_context(context, context_norm=self.context_norm)

        if output_attentions:
            return context, self._empty_attentions(context, seq_len)
        return context, None


class LinearContextAttentionLocalWindow(LinearContextAttentionPosEnc):
    """Линейное внимание с локальными окнами разной ширины по головам."""

    def __init__(self, hidden_size: int, num_attention_heads: int, dropout_prob: float = 0.1,
                 max_position_embeddings: int = 768):
        super().__init__(hidden_size, num_attention_heads, dropout_prob, max_position_embeddings)
        self.register_buffer(
            "window_sizes",
            self._build_window_sizes(num_attention_heads, max_position_embeddings),
        )
        write_log(f"Локальные окна: windows={self.window_sizes.tolist()}")

    def _build_window_sizes(self, num_attention_heads: int, max_position_embeddings: int) -> torch.Tensor:
        """Строит набор размеров окон: от глобального охвата к узким локальным окнам."""
        window_sizes = []
        for head_idx in range(num_attention_heads):
            if head_idx == 0:
                # Первая голова сохраняет полный охват последовательности.
                window_sizes.append(max_position_embeddings)
                continue

            window = max_position_embeddings // (2 ** head_idx)
            if window < 2:
                # Если окно вырождается, возвращаем голову к глобальному охвату.
                window = max_position_embeddings
            window_sizes.append(window)
        return torch.tensor(window_sizes, dtype=torch.long)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None,
                output_attentions: bool = False, **kwargs):
        """Вычисляет контекст в локальном окне вокруг каждого токена."""
        _ = kwargs
        batch_size, seq_len, _ = hidden_states.shape
        num_heads = self.num_attention_heads
        head_dim = self.attention_head_size
        device = hidden_states.device
        dtype = hidden_states.dtype

        values = self._project_values(hidden_states)  # (B, H, L, D)

        pos_embeds = self.pe[:seq_len].unsqueeze(0).unsqueeze(0)
        pos_embeds = pos_embeds * self.head_scales.view(1, num_heads, 1, 1)
        values = values * (1.0 + pos_embeds.to(dtype))

        mask = self._build_mask(attention_mask, batch_size, seq_len, device, dtype)
        mask_expanded = mask.expand(-1, num_heads, -1, -1)
        masked_values = values * mask_expanded

        # Префиксные суммы позволяют быстро получать сумму Value внутри любого окна.
        value_prefix = torch.cat(
            [torch.zeros(batch_size, num_heads, 1, head_dim, device=device, dtype=dtype), masked_values],
            dim=2,
        ).cumsum(dim=2)
        mask_prefix = torch.cat(
            [torch.zeros(batch_size, num_heads, 1, 1, device=device, dtype=dtype), mask_expanded],
            dim=2,
        ).cumsum(dim=2)

        # Для каждой головы и позиции вычисляем границы локального окна.
        positions = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(num_heads, -1)
        window_sizes = self.window_sizes.to(device)
        global_heads = window_sizes < 2

        token_radius = torch.clamp(window_sizes // 2, min=1)
        desired_span = 2 * token_radius.unsqueeze(1) + 1

        # Shift-to-fit: если окно выходит за правую границу, сдвигаем его влево.
        window_start = torch.clamp(positions - token_radius.unsqueeze(1), min=0)
        window_end = window_start + desired_span
        overflow = torch.clamp(window_end - seq_len, min=0)
        window_start = torch.clamp(window_start - overflow, min=0)
        window_end = torch.clamp(window_start + desired_span, max=seq_len)

        full_sequence_heads = global_heads | (window_sizes >= seq_len)
        if full_sequence_heads.any():
            # Для глобальных голов окно покрывает всю последовательность.
            window_start[full_sequence_heads] = 0
            window_end[full_sequence_heads] = seq_len

        start_idx = window_start.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, head_dim)
        end_idx = window_end.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, head_dim)
        window_value_sum = torch.gather(value_prefix, dim=2, index=end_idx) - torch.gather(
            value_prefix, dim=2, index=start_idx
        )

        # Отдельно считаем количество валидных токенов внутри каждого окна.
        start_mask_idx = window_start.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, 1)
        end_mask_idx = window_end.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, 1)
        window_token_count = torch.gather(mask_prefix, dim=2, index=end_mask_idx) - torch.gather(
            mask_prefix, dim=2, index=start_mask_idx
        )

        # Исключаем текущий токен из среднего, чтобы не копировать его собственное Value.
        numerator = window_value_sum - masked_values
        denominator = torch.clamp(window_token_count - mask_expanded, min=1.0)
        context = numerator / denominator
        context = context * mask_expanded

        context = self._finalize_context(context, context_norm=self.context_norm)

        if output_attentions:
            return context, self._empty_attentions(context, seq_len)
        return context, None


class LinearContextAttentionWeighted(LinearContextAttentionPosEnc):
    """Линейное внимание с обучаемым позиционным затуханием."""

    def __init__(self, hidden_size: int, num_attention_heads: int, dropout_prob: float = 0.1,
                 max_position_embeddings: int = 768):
        super().__init__(hidden_size, num_attention_heads, dropout_prob, max_position_embeddings)

        # Параметры формы позиционного затухания.
        self.alpha = nn.Parameter(torch.tensor(1.0))  # коэффициент масштаба
        self.beta = nn.Parameter(torch.tensor(1.5))   # степень нелинейности

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None,
                output_attentions: bool = False, **kwargs):
        """Строит контекст как взвешенную сумму Value с позиционным затуханием."""
        batch_size, seq_len, _ = hidden_states.shape
        num_heads = self.num_attention_heads
        head_dim = self.attention_head_size
        dtype, device = hidden_states.dtype, hidden_states.device

        values = self._project_values(hidden_states)

        pos_embeds = self.pe[:seq_len].unsqueeze(0).unsqueeze(0) * self.head_scales.view(1, num_heads, 1, 1)
        values = values * (1.0 + pos_embeds.to(dtype))

        mask = self._build_mask(attention_mask, batch_size, seq_len, device, dtype)
        masked_values = values * mask

        # Ограничиваем параметры затухания безопасным диапазоном.
        alpha = torch.clamp(self.alpha, 0.05, 2.0).to(dtype)
        beta = torch.clamp(self.beta, 0.5, 3.0).to(dtype)

        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        position_powers = torch.pow(positions, beta)
        decay_weights = torch.exp(-alpha * position_powers)
        decay_weights = decay_weights / decay_weights.sum().clamp(min=1e-8)

        # Общий взвешенный контекст по всей последовательности.
        weighted_context = torch.einsum('bhld,l->bhd', masked_values, decay_weights)
        weighted_context = weighted_context.unsqueeze(2)

        # Исключаем текущую позицию из собственного контекста.
        position_weights = decay_weights.view(1, 1, seq_len, 1)
        denominator = torch.clamp(1.0 - position_weights, min=1e-8)
        context = (weighted_context - masked_values * position_weights) / denominator
        context = context * mask

        context = self._finalize_context(context, context_norm=self.context_norm)

        if output_attentions:
            return context, self._empty_attentions(context, seq_len)

        return context, None


class LinearSelfAttention(nn.Module):
    """Адаптер, заменяющий стандартный BERT self-attention на выбранный линейный блок."""

    def __init__(self, config, attention_class=LinearContextAttentionDilated):
        super().__init__()
        self.inner_attention = attention_class(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            dropout_prob=config.attention_probs_dropout_prob,
            max_position_embeddings=config.max_position_embeddings,
        )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = config.hidden_size

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        """Проксирует вызов во внутренний кастомный attention-блок."""
        return self.inner_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )


def inject_linear_attention(layer, config, attention_class=LinearContextAttentionDilated):
    """Заменяет self-attention в одном encoder-слое BERT на кастомный блок."""
    layer.attention.self = LinearSelfAttention(config, attention_class)
    return layer


class BertWithCustomAttention(nn.Module):
    """Обертка над BERT-классификатором с частичной заменой attention-слоев.

    Класс позволяет замораживать часть исходных encoder-слоев, заменять последние
    слои на кастомный attention-блок и при необходимости добавлять новые encoder-слои
    поверх базовой архитектуры.
    """

    def __init__(self, model, num_layers_to_replace: int = 0, num_layers_to_add: int = 0,
                 attention_class=LinearContextAttentionDilated):
        super().__init__()
        self.bert = model.bert
        self.classifier = model.classifier
        self.config = model.config
        self.attention_class = attention_class

        total_layers = len(self.bert.encoder.layer)
        self._layer_template = copy.deepcopy(self.bert.encoder.layer[-1])

        num_layers_to_replace = max(0, min(int(num_layers_to_replace), total_layers))
        write_log(
            f"BERT layers: total={total_layers}, replace={num_layers_to_replace}, add={num_layers_to_add}"
        )

        if num_layers_to_replace > 0:
            # Замораживаем исходные нижние слои, которые остаются без замены.
            for i in range(total_layers - num_layers_to_replace):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = False
            write_log(f"Frozen original layers: {total_layers - num_layers_to_replace}")

            for i in range(num_layers_to_replace):
                idx = total_layers - num_layers_to_replace + i
                # Заменяем только self-attention; output, FFN и residual-связи слоя сохраняются.
                inject_linear_attention(self.bert.encoder.layer[idx], self.config, self.attention_class)
                for param in self.bert.encoder.layer[idx].parameters():
                    param.requires_grad = True
            write_log(f"Replaced trainable layers: {num_layers_to_replace}")
        else:
            # Если замены нет, замораживаем все encoder-слои и обучаем только классификатор.
            for i in range(total_layers):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = False
            write_log(f"Frozen original layers: {total_layers}")

        if num_layers_to_add > 0:
            for _ in range(num_layers_to_add):
                new_layer = copy.deepcopy(self._layer_template)
                inject_linear_attention(new_layer, self.config, self.attention_class)
                self.bert.encoder.layer.append(new_layer)
                for param in new_layer.parameters():
                    param.requires_grad = True
            write_log(f"Added custom layers: {num_layers_to_add}")
            write_log(f"Total layers after extension: {len(self.bert.encoder.layer)} (was {total_layers})")

        total_layers_after = len(self.bert.encoder.layer)
        self.config.num_hidden_layers = total_layers_after
        if hasattr(self.bert, "config"):
            self.bert.config.num_hidden_layers = total_layers_after
        if hasattr(self.bert.encoder, "config"):
            self.bert.encoder.config.num_hidden_layers = total_layers_after

        write_log(f"Updated config: num_hidden_layers={self.config.num_hidden_layers}")

        for param in self.classifier.parameters():
            param.requires_grad = True

        model_self = self

        def custom_encoder_forward(
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            output_attentions=False,
            output_hidden_states=False,
            **kwargs,
        ):
            """Forward encoder-а, совместимый с Hugging Face BaseModelOutput."""
            _ = kwargs
            all_hidden_states = () if output_hidden_states else None
            all_self_attentions = () if output_attentions else None
            encoder_layers = model_self.bert.encoder.layer

            for i, layer_module in enumerate(encoder_layers):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                # Каждый encoder-слой возвращает hidden states и, при запросе, attention tensors.
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    past_key_values=past_key_values[i] if past_key_values is not None else None,
                    head_mask=head_mask[i] if head_mask is not None else None,
                    output_attentions=output_attentions,
                )

                if isinstance(layer_outputs, tuple):
                    hidden_states = layer_outputs[0]
                    if output_attentions and len(layer_outputs) > 1:
                        # Кастомные attention-блоки возвращают совместимый placeholder attention.
                        all_self_attentions = all_self_attentions + (layer_outputs[1],)
                else:
                    hidden_states = layer_outputs

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # Возвращаем BaseModelOutput, чтобы внешний BERT-классификатор работал без изменений.
            result = BaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
            )
            result.past_key_values = None
            result.cross_attentions = None
            return result

        self.bert.encoder.forward = custom_encoder_forward
        write_log(f"Custom encoder forward installed. Total layers: {len(self.bert.encoder.layer)}")

    def forward(self, input_ids, attention_mask=None, labels=None):
        """Выполняет forward классификатора и возвращает ``SequenceClassifierOutput``."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = getattr(outputs, "pooler_output", outputs.last_hidden_state[:, 0])
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
