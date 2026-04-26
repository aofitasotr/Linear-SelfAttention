"""????????? ?????????? ????????? ???????? ? ???????? ??? ?????????? ???? ?????? ? BERT."""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutput

from logging_utils import write_log


# простое линейное внимание без позиций
class LinearContextAttention(nn.Module):
    """??????? ?????????? ????????? ???????? ????? ?????????? ?????????? ????????.

    ??? ?????? ?????? ??????????? ?????? ???????? ???????? `V`. ???????? ??????
    ???????? ??? ??????? ?? ???? ?????????? ????????, ????? ???????. ???????? ??
    ?????????? ??????? ???????? ?????????????? `QK^T`, ??????? ????? ????????
    ??????? ?????????? ?? ????? ??????????????????.
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
        nn.init.normal_(self.value.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.dense.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.value.bias)
        nn.init.zeros_(self.dense.bias)

    def _build_mask(self, attention_mask: torch.Tensor, batch_size: int, seq_len: int, device, dtype):
        if attention_mask is None:
            mask = torch.ones(batch_size, seq_len, device=device, dtype=dtype)
        elif attention_mask.dim() == 4:
            mask = (attention_mask[:, 0, 0, :] > -10000).to(dtype)
        elif attention_mask.dim() == 2:
            mask = attention_mask.to(dtype)
        else:
            raise ValueError(f"Unsupported attention_mask dim: {attention_mask.dim()}")
        return mask.unsqueeze(1).unsqueeze(3)

    def _project_values(self, hidden_states: torch.Tensor):
        batch_size, seq_len, _ = hidden_states.shape
        return self.value(hidden_states).view(
            batch_size, seq_len, self.num_attention_heads, self.attention_head_size
        ).permute(0, 2, 1, 3).contiguous()

    def _reshape_context(self, context: torch.Tensor):
        batch_size, _, seq_len, _ = context.shape
        return context.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.hidden_size)

    def _finalize_context(self, context: torch.Tensor, context_norm=None):
        context = self._reshape_context(context)
        if context_norm is not None:
            context = context_norm(context)
        context = self.dense(context)
        return self.dropout(context)

    def _empty_attentions(self, context: torch.Tensor, seq_len: int):
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
        """
        Прямой проход.
        """
        B, seq_len, _ = hidden_states.shape

        # Проекция + разделение на головы
        V = self._project_values(hidden_states)  # (B, H, L, D)

        # Обработка маски
        mask = self._build_mask(attention_mask, B, seq_len, V.device, V.dtype)  # (B, 1, L, 1)
        V_masked = V * mask

        # Агрегация
        total_sum = V_masked.sum(dim=2, keepdim=True)      # (B, H, 1, D)
        total_count = mask.sum(dim=2, keepdim=True)        # (B, 1, 1, 1)

        denom = torch.clamp(total_count - 1.0, min=1.0)    # (B, 1, 1, 1)
        context = (total_sum - V_masked) / denom           # (B, H, L, D)
        context = context * mask                           # обнуление паддинга

        # Сборка
        context = self._finalize_context(context)

        if output_attentions:
            return context, self._empty_attentions(context, seq_len)
        return context, None



# Вариант с синусоидальным позиционным кодированием
class LinearContextAttentionPosEnc(LinearContextAttention):
    """???????? ???????? ? ?????????????? ??????????? ?????????? ????????."""

    def __init__(self, hidden_size: int, num_attention_heads: int, dropout_prob: float = 0.1,
                 max_position_embeddings: int = 768):
        super().__init__(hidden_size, num_attention_heads, dropout_prob)

        self.max_position_embeddings = max_position_embeddings

        # Пост-нормализация
        self.context_norm = nn.LayerNorm(hidden_size)

        # Синусоидальные позиционные эмбеддинги
        pe = torch.zeros(max_position_embeddings, self.attention_head_size)
        position = torch.arange(0, max_position_embeddings, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.attention_head_size, 2).float() *
            (-math.log(10000.0) / self.attention_head_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

        # Обучаемые масштабы голов
        head_scales = torch.linspace(0.5, 2, num_attention_heads)
        self.head_scales = nn.Parameter(head_scales, requires_grad=True)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None,
                output_attentions: bool = False, **kwargs):
        _ = kwargs
        B, seq_len, _ = hidden_states.shape
        H = self.num_attention_heads
        D = self.attention_head_size

        # Проекция значений
        V = self._project_values(hidden_states)  # (B, H, L, D)

        # Позиционная модуляция
        pos_emb = self.pe[:seq_len].unsqueeze(0).unsqueeze(0)          # (1, 1, L, D)
        pos_emb = pos_emb * self.head_scales.view(1, H, 1, 1)          # (1, H, L, D)
        V = V * (1.0 + pos_emb)                                        # (B, H, L, D)

        # Маска
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



# Разреженная версия
class LinearContextAttentionDilated(LinearContextAttentionPosEnc):
    """??????????? ???????? ???????? ?? ?????????? ??????????? ???????."""

    def __init__(self, hidden_size: int, num_attention_heads: int, dropout_prob: float = 0.1,
                 max_position_embeddings: int = 768):
        super().__init__(hidden_size, num_attention_heads, dropout_prob, max_position_embeddings)

        # Предвычисление dilation и offset для каждой головы
        self.register_buffer('dilations', self._compute_dilations(num_attention_heads))
        self.register_buffer('offsets', self._compute_offsets(num_attention_heads))
        self.max_dilation = int(self.dilations.max().item())

        write_log(f"Дилатированное внимание: dilations={self.dilations.tolist()}, "
                  f"offsets={self.offsets.tolist()}, max_dilation={self.max_dilation}")

    def _compute_dilations(self, num_heads):
        head_indices = torch.arange(num_heads)
        max_power = (num_heads + 1).bit_length()
        powers = 2 ** torch.arange(max_power)
        cumsum = torch.cumsum(powers, dim=0)
        group_indices = (head_indices.unsqueeze(1) < cumsum.unsqueeze(0)).float()
        group_ids = torch.argmax(group_indices, dim=1)
        dilations = 2 ** group_ids
        return dilations

    def _compute_offsets(self, num_heads):
        dilations = self._compute_dilations(num_heads)
        unique_dilations = dilations.unique()
        offsets = torch.zeros(num_heads, dtype=torch.long)
        for dilation in unique_dilations:
            mask = (dilations == dilation)
            if mask.any():
                group_indices = torch.arange(mask.sum().item())
                offsets[mask] = group_indices % dilation.item()
        return offsets

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None,
                output_attentions: bool = False, **kwargs):
        _ = kwargs
        B, seq_len, _ = hidden_states.shape
        H = self.num_attention_heads
        D = self.attention_head_size

        # Проекция значений
        V = self._project_values(hidden_states)  # (B, H, L, D)

        # Позиционная модуляция (с dilation)
        pos_emb = self.pe[:seq_len].unsqueeze(0).unsqueeze(0)          # (1, 1, L, D)
        pos_emb = pos_emb * self.head_scales.view(1, H, 1, 1)          # (1, H, L, D)
        V = V * (1.0 + pos_emb)                       # (B, H, L, D)

        # Маска
        mask = self._build_mask(attention_mask, B, seq_len, V.device, V.dtype)  # (B, 1, L, 1)

        # Разреженное индексирование
        target_len = max(1, seq_len // self.max_dilation)
        pos_indices = torch.arange(target_len, device=V.device)
        dilations = self.dilations.to(V.device)                         # (H)
        offsets = self.offsets.to(V.device)                             # (H)

        indices = offsets.unsqueeze(1) + pos_indices.unsqueeze(0) * dilations.unsqueeze(1)
        indices = indices % seq_len
        indices = indices.long()                                         # (H, target_len)

        indices_expanded = indices.unsqueeze(0).expand(B, -1, -1)       # (B, H, target_len)

        # Сбор значений
        indices_for_gather = indices_expanded.unsqueeze(-1).expand(-1, -1, -1, D)
        V_gathered = torch.gather(V, dim=2, index=indices_for_gather)   # (B, H, target_len, D)

        # Сбор маски
        mask_expanded = mask.expand(-1, H, -1, -1)                      # (B, H, L, 1)
        indices_for_mask = indices_expanded.unsqueeze(-1)               # (B, H, target_len, 1)
        mask_gathered = torch.gather(mask_expanded, dim=2, index=indices_for_mask)  # (B, H, target_len, 1)

        V_masked = V_gathered * mask_gathered
        total_sum = V_masked.sum(dim=2, keepdim=True)                   # (B, H, 1, D)
        total_count = mask_gathered.sum(dim=2, keepdim=True)            # (B, H, 1, 1)

        numerator = total_sum - V_masked
        denominator = torch.clamp(total_count - 1.0, min=1.0)
        context_gathered = numerator / denominator                      # (B, H, target_len, D)
        context_gathered = context_gathered * mask_gathered

        # Разворачивание
        context_full = torch.zeros_like(V)                              # (B, H, L, D)
        scatter_indices = indices_expanded.unsqueeze(-1)                # (B, H, target_len, 1)
        context_full.scatter_(dim=2, index=scatter_indices.expand(-1, -1, -1, D),
                              src=context_gathered)
        context_full = context_full * mask

        context = self._finalize_context(context_full, context_norm=self.context_norm)

        if output_attentions:
            return context, self._empty_attentions(context, seq_len)
        return context, None


class LinearContextAttentionLocalWindow(LinearContextAttentionPosEnc):
    """????????? ???????? ???????? ? ?????? ??????? ???????? ?? ???????.

    ?????? ?????? ????? ?????? ??? ??????????????????, ????????? ???????? ?
    ?????????? ??????, ?????? ??????? ??????????? ?? ???? ????? ??????? ??????.
    ? ?????? ???????????? ????? shift-to-fit, ????? ???? ?? ??????????? ?????????
    ???????? ??????.
    """

    def __init__(self, hidden_size: int, num_attention_heads: int, dropout_prob: float = 0.1,
                 max_position_embeddings: int = 768):
        super().__init__(hidden_size, num_attention_heads, dropout_prob, max_position_embeddings)
        self.register_buffer(
            "window_sizes",
            self._build_window_sizes(num_attention_heads, max_position_embeddings),
        )
        write_log(f"Local-window внимание: windows={self.window_sizes.tolist()}")

    def _build_window_sizes(self, num_attention_heads: int, max_position_embeddings: int) -> torch.Tensor:
        window_sizes = []
        for head_idx in range(num_attention_heads):
            if head_idx == 0:
                window_sizes.append(max_position_embeddings)
                continue

            window = max_position_embeddings // (2 ** head_idx)
            if window < 2:
                window = max_position_embeddings
            window_sizes.append(window)
        return torch.tensor(window_sizes, dtype=torch.long)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None,
                output_attentions: bool = False, **kwargs):
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

        # Один cumsum по последовательности, дальше окна берутся вычитанием префиксных сумм.
        value_prefix = torch.cat(
            [torch.zeros(batch_size, num_heads, 1, head_dim, device=device, dtype=dtype), masked_values],
            dim=2,
        ).cumsum(dim=2)
        mask_prefix = torch.cat(
            [torch.zeros(batch_size, num_heads, 1, 1, device=device, dtype=dtype), mask_expanded],
            dim=2,
        ).cumsum(dim=2)

        positions = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(num_heads, -1)
        window_sizes = self.window_sizes.to(device)
        global_heads = window_sizes < 2

        token_radius = torch.clamp(window_sizes // 2, min=1)
        desired_span = 2 * token_radius.unsqueeze(1) + 1

        # Shift-to-fit: у края окно сдвигается внутрь, чтобы по возможности сохранять тот же размер.
        window_start = torch.clamp(positions - token_radius.unsqueeze(1), min=0)
        window_end = window_start + desired_span
        overflow = torch.clamp(window_end - seq_len, min=0)
        window_start = torch.clamp(window_start - overflow, min=0)
        window_end = torch.clamp(window_start + desired_span, max=seq_len)

        full_sequence_heads = global_heads | (window_sizes >= seq_len)
        if full_sequence_heads.any():
            window_start[full_sequence_heads] = 0
            window_end[full_sequence_heads] = seq_len

        start_idx = window_start.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, head_dim)
        end_idx = window_end.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, head_dim)
        window_value_sum = torch.gather(value_prefix, dim=2, index=end_idx) - torch.gather(
            value_prefix, dim=2, index=start_idx
        )

        start_mask_idx = window_start.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, 1)
        end_mask_idx = window_end.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, 1)
        window_token_count = torch.gather(mask_prefix, dim=2, index=end_mask_idx) - torch.gather(
            mask_prefix, dim=2, index=start_mask_idx
        )

        numerator = window_value_sum - masked_values
        denominator = torch.clamp(window_token_count - mask_expanded, min=1.0)
        context = numerator / denominator
        context = context * mask_expanded

        context = self._finalize_context(context, context_norm=self.context_norm)

        if output_attentions:
            return context, self._empty_attentions(context, seq_len)
        return context, None


class LinearContextAttentionWeighted(LinearContextAttentionPosEnc):
    """???????? ???????? ? ????????????? ?????????? ?????? ?? ???????.

    ???? ??????? ???????? ???????? ???? `exp(-alpha * t^beta)`, ??? `alpha` ?
    `beta` ???????? ?????????? ???????????.
    """
    
    def __init__(self, hidden_size: int, num_attention_heads: int, dropout_prob: float = 0.1,
                 max_position_embeddings: int = 768):
        super().__init__(hidden_size, num_attention_heads, dropout_prob, max_position_embeddings)
        
        # Обучаемые параметры формы весов
        self.alpha = nn.Parameter(torch.tensor(1.0))  # коэффициент масштаба
        self.beta = nn.Parameter(torch.tensor(1.5))   # степень нелинейности
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None,
                output_attentions: bool = False, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape
        num_heads = self.num_attention_heads
        head_dim = self.attention_head_size
        dtype, device = hidden_states.dtype, hidden_states.device
        
        # Проекция значений и позиционная модуляция
        values = self._project_values(hidden_states)
        
        pos_embeds = self.pe[:seq_len].unsqueeze(0).unsqueeze(0) * self.head_scales.view(1, num_heads, 1, 1)
        values = values * (1.0 + pos_embeds.to(dtype))
        
        # Подготовка маски
        mask = self._build_mask(attention_mask, batch_size, seq_len, device, dtype)
        masked_values = values * mask
        
        # Вычисление взвешенных коэффициентов
        alpha = torch.clamp(self.alpha, 0.05, 2.0).to(dtype)
        beta = torch.clamp(self.beta, 0.5, 3.0).to(dtype)
        
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        position_powers = torch.pow(positions, beta)
        decay_weights = torch.exp(-alpha * position_powers)
        decay_weights = decay_weights / decay_weights.sum().clamp(min=1e-8)
        
        # Взвешенная агрегация контекста
        weighted_context = torch.einsum('bhld,l->bhd', masked_values, decay_weights)
        weighted_context = weighted_context.unsqueeze(2)
        
        position_weights = decay_weights.view(1, 1, seq_len, 1)
        denominator = torch.clamp(1.0 - position_weights, min=1e-8)
        context = (weighted_context - masked_values * position_weights) / denominator
        context = context * mask
        
        # Финальная проекция
        context = self._finalize_context(context, context_norm=self.context_norm)
        
        if output_attentions:
            return context, self._empty_attentions(context, seq_len)
        
        return context, None


# Обёртка для совместимости с BERT
class LinearSelfAttention(nn.Module):
    """???????, ??????????? ? ??????????? BERT self-attention.

    ??????????? ????????? ?????????? ????????? ???????? ???, ????? ?? ????? ????
    ?????????? ?????? ???????????? `layer.attention.self`.
    """

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
        return self.inner_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )


def inject_linear_attention(layer, config, attention_class=LinearContextAttentionDilated):
    """???????? ??????????? self-attention ???? BERT ?? ???????? ???????? ????."""
    layer.attention.self = LinearSelfAttention(config, attention_class)
    return layer



# Основная модель BERT с поддержкой замены слоёв

class BertWithCustomAttention(nn.Module):
    """??????? ??? BERT ? ??????? ?/??? ??????????? attention-?????.

    ????????? ???????? ???????? ????????? encoder-???? ?? ????????? ??????????
    ????????? ???????? ? ??? ????????????? ???????? ????? ???? ???? ?? ????.
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
            f"Конфигурация слоёв: всего={total_layers}, заменить={num_layers_to_replace}, добавить={num_layers_to_add}"
        )

        if num_layers_to_replace > 0:
            for i in range(total_layers - num_layers_to_replace):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = False
            write_log(f"Заморожены первые {total_layers - num_layers_to_replace} слоёв")

            for i in range(num_layers_to_replace):
                idx = total_layers - num_layers_to_replace + i
                inject_linear_attention(self.bert.encoder.layer[idx], self.config, self.attention_class)
                for param in self.bert.encoder.layer[idx].parameters():
                    param.requires_grad = True
            write_log(f"Заменены последние {num_layers_to_replace} слоёв")
        else:
            for i in range(total_layers):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = False
            write_log(f"Заморожены все {total_layers} слоёв")

        if num_layers_to_add > 0:
            for _ in range(num_layers_to_add):
                new_layer = copy.deepcopy(self._layer_template)
                inject_linear_attention(new_layer, self.config, self.attention_class)
                self.bert.encoder.layer.append(new_layer)
                for param in new_layer.parameters():
                    param.requires_grad = True
            write_log(f"Добавлено {num_layers_to_add} новых слоёв")
            write_log(f"Итого слоёв после добавления: {len(self.bert.encoder.layer)} (было {total_layers})")

        total_layers_after = len(self.bert.encoder.layer)
        self.config.num_hidden_layers = total_layers_after
        if hasattr(self.bert, "config"):
            self.bert.config.num_hidden_layers = total_layers_after
        if hasattr(self.bert.encoder, "config"):
            self.bert.encoder.config.num_hidden_layers = total_layers_after

        write_log(f"Обновление конфига: num_hidden_layers={self.config.num_hidden_layers}")

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
            _ = kwargs
            all_hidden_states = () if output_hidden_states else None
            all_self_attentions = () if output_attentions else None
            encoder_layers = model_self.bert.encoder.layer

            for i, layer_module in enumerate(encoder_layers):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

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
                        all_self_attentions = all_self_attentions + (layer_outputs[1],)
                else:
                    hidden_states = layer_outputs

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            result = BaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
            )
            result.past_key_values = None
            result.cross_attentions = None
            return result

        self.bert.encoder.forward = custom_encoder_forward
        write_log(f"Переопределён forward энкодера: теперь используется {len(self.bert.encoder.layer)} слоёв")

    def forward(self, input_ids, attention_mask=None, labels=None):
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
