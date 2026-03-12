import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutput

from logging_utils import write_log


# ----------------------------------------------------------------------
# Базовый класс: простое линейное внимание без позиций
# ----------------------------------------------------------------------
class LinearContextAttention(nn.Module):
    """
    Линейное внимание с максимальной оптимизацией для GPU.
    Убрано ветвление через дорогостоящий torch.all() — всегда обрабатываем с маской.
    Этот класс служит основой для более сложных вариантов.
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

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None,
                output_attentions: bool = False, **kwargs):
        """
        Прямой проход (без позиционной модуляции).
        """
        B, seq_len, _ = hidden_states.shape

        # Проекция + разделение на головы
        V = self.value(hidden_states).view(
            B, seq_len, self.num_attention_heads, self.attention_head_size
        ).permute(0, 2, 1, 3).contiguous()  # (B, H, L, D)

        # Обработка маски (без ветвлений)
        if attention_mask is None:
            mask = torch.ones(B, seq_len, device=V.device, dtype=V.dtype)
        elif attention_mask.dim() == 4:
            mask = (attention_mask[:, 0, 0, :] > -10000).float()
        elif attention_mask.dim() == 2:
            mask = attention_mask.float()
        else:
            raise ValueError(f"Unsupported attention_mask dim: {attention_mask.dim()}")

        mask = mask.unsqueeze(1).unsqueeze(3)  # (B, 1, L, 1)
        V_masked = V * mask

        # Агрегация
        total_sum = V_masked.sum(dim=2, keepdim=True)      # (B, H, 1, D)
        total_count = mask.sum(dim=2, keepdim=True)        # (B, 1, 1, 1)

        denom = torch.clamp(total_count - 1.0, min=1.0)    # (B, 1, 1, 1)
        context = (total_sum - V_masked) / denom           # (B, H, L, D)
        context = context * mask                           # обнуляем паддинг

        # Сборка
        context = context.permute(0, 2, 1, 3).reshape(B, seq_len, self.hidden_size)
        context = self.dense(context)
        context = self.dropout(context)

        if output_attentions:
            attention_probs = torch.zeros(
                B, self.num_attention_heads, seq_len, seq_len,
                device=context.device, dtype=context.dtype
            )
            return context, attention_probs
        return context, None


# ----------------------------------------------------------------------
# Вариант с синусоидальным позиционным кодированием и обучаемыми head_scales
# ----------------------------------------------------------------------
class LinearContextAttentionPosEnc(LinearContextAttention):
    """
    Линейное внимание с синусоидальными позициями и обучаемыми head_scales.
    Добавлены: позиционные эмбеддинги, пост-нормализация, head_scales.
    """

    def __init__(self, hidden_size: int, num_attention_heads: int, dropout_prob: float = 0.1,
                 max_position_embeddings: int = 768):
        super().__init__(hidden_size, num_attention_heads, dropout_prob)

        self.max_position_embeddings = max_position_embeddings

        # Пост-нормализация
        self.context_norm = nn.LayerNorm(hidden_size)

        # Синусоидальные позиционные эмбеддинги (кэш)
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
        V = self.value(hidden_states)
        V = V.view(B, seq_len, H, D).permute(0, 2, 1, 3)  # (B, H, L, D)

        # Позиционная модуляция
        pos_emb = self.pe[:seq_len].unsqueeze(0).unsqueeze(0)          # (1, 1, L, D)
        pos_emb = pos_emb * self.head_scales.view(1, H, 1, 1)          # (1, H, L, D)
        V = V * (1.0 + pos_emb)                                        # (B, H, L, D)

        # Маска
        if attention_mask is None:
            mask = torch.ones(B, seq_len, device=V.device, dtype=V.dtype)
        elif attention_mask.dim() == 4:
            mask = (attention_mask[:, 0, 0, :] > -10000).float()
        else:
            mask = attention_mask.float()
        mask = mask.unsqueeze(1).unsqueeze(3)                           # (B, 1, L, 1)

        V_masked = V * mask
        total_sum = V_masked.sum(dim=2, keepdim=True)                   # (B, H, 1, D)
        total_count = mask.sum(dim=2, keepdim=True)                     # (B, 1, 1, 1)

        numerator = total_sum - V_masked
        denominator = torch.clamp(total_count - 1.0, min=1.0)
        context = numerator / denominator                               # (B, H, L, D)
        context = context * mask

        context = context.permute(0, 2, 1, 3).reshape(B, seq_len, self.hidden_size)
        context = self.context_norm(context)
        context = self.dense(context)
        context = self.dropout(context)

        if output_attentions:
            attn = torch.zeros(B, H, seq_len, seq_len, device=context.device)
            return context, attn
        return context, None


# ----------------------------------------------------------------------
# Дилатированная версия (наследует от PosEnc)
# ----------------------------------------------------------------------
class LinearContextAttentionDilated(LinearContextAttentionPosEnc):
    """
    Дилатированное линейное внимание.
    Добавлены: предвычисленные dilation и offset.
    """

    def __init__(self, hidden_size: int, num_attention_heads: int, dropout_prob: float = 0.1,
                 max_position_embeddings: int = 768):
        super().__init__(hidden_size, num_attention_heads, dropout_prob, max_position_embeddings)

        # Обучаемый коэффициент модуляции (дополнительный к head_scales)
        # self.pos_scale = nn.Parameter(torch.tensor(0.3), requires_grad=True)

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
        V = self.value(hidden_states)
        V = V.view(B, seq_len, H, D).permute(0, 2, 1, 3)  # (B, H, L, D)

        # Позиционная модуляция (с dilation)
        pos_emb = self.pe[:seq_len].unsqueeze(0).unsqueeze(0)          # (1, 1, L, D)
        pos_emb = pos_emb * self.head_scales.view(1, H, 1, 1)          # (1, H, L, D)
        V = V * (1.0 + pos_emb)                       # (B, H, L, D)

        # Маска
        if attention_mask is None:
            mask = torch.ones(B, seq_len, device=V.device, dtype=V.dtype)
        elif attention_mask.dim() == 4:
            mask = (attention_mask[:, 0, 0, :] > -10000).float()
        else:
            mask = attention_mask.float()
        mask = mask.unsqueeze(1).unsqueeze(3)                           # (B, 1, L, 1)

        # Дилатированное индексирование
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

        context = context_full.permute(0, 2, 1, 3).reshape(B, seq_len, self.hidden_size)
        context = self.context_norm(context)
        context = self.dense(context)
        context = self.dropout(context)

        if output_attentions:
            attn = torch.zeros(B, H, seq_len, seq_len, device=context.device)
            return context, attn
        return context, None
    
class LinearContextAttentionWeighted(LinearContextAttentionPosEnc):
    """
    Линейное внимание с весовой функцией затухания по расстоянию.
    Веса: w(d) = exp(-alpha * d^beta)
    Параметры alpha и beta оптимизируются генетическим алгоритмом.
    """
    
    def __init__(self, hidden_size: int, num_attention_heads: int, dropout_prob: float = 0.1,
                 max_position_embeddings: int = 768):
        super().__init__(hidden_size, num_attention_heads, dropout_prob, max_position_embeddings)
        
        # параметры затухания для оптимизации
        self.alpha = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.beta = nn.Parameter(torch.tensor(1.5), requires_grad=True)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None,
            output_attentions: bool = False, **kwargs):
        B, seq_len, _ = hidden_states.shape
        H = self.num_attention_heads
        D = self.attention_head_size
        
        # Приводим параметры к тому же типу, что и hidden_states
        target_dtype = hidden_states.dtype
        
        # проекция значений
        V = self.value(hidden_states)
        V = V.view(B, seq_len, H, D).permute(0, 2, 1, 3)
        
        # позиционная модуляция
        pos_emb = self.pe[:seq_len].unsqueeze(0).unsqueeze(0)
        pos_emb = pos_emb * self.head_scales.view(1, H, 1, 1).to(target_dtype)
        V = V * (1.0 + pos_emb)
        
        # маска
        if attention_mask is None:
            mask = torch.ones(B, seq_len, device=V.device, dtype=V.dtype)
        elif attention_mask.dim() == 4:
            mask = (attention_mask[:, 0, 0, :] > -10000).to(V.dtype)
        else:
            mask = attention_mask.to(V.dtype)
        mask = mask.unsqueeze(1).unsqueeze(3)
        
        # приводим alpha и beta к нужному типу
        alpha = torch.clamp(self.alpha, 0.05, 2.0).to(target_dtype)
        beta = torch.clamp(self.beta, 0.5, 3.0).to(target_dtype)
        
        # матрица расстояний (приводим к нужному типу)
        positions = torch.arange(seq_len, device=V.device).to(target_dtype)
        dist = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))
        
        # веса
        weights = torch.exp(-alpha * torch.pow(dist, beta))
        weights = weights.to(V.dtype)  # финальное приведение к типу V
        weights = weights * (1 - torch.eye(seq_len, device=V.device, dtype=V.dtype))
        
        # взвешенная сумма
        context = torch.einsum('ij,bhjd->bhid', weights, V)
        
        # нормализация
        weights_sum = weights.sum(dim=1, keepdim=True)
        weights_sum = torch.clamp(weights_sum, min=1e-8)
        context = context / weights_sum.unsqueeze(0).unsqueeze(0)
        
        context = context * mask
        context = context.permute(0, 2, 1, 3).reshape(B, seq_len, self.hidden_size)
        context = self.context_norm(context)
        context = self.dense(context)
        context = self.dropout(context)
        
        if output_attentions:
            return context, weights
        return context, None

# ----------------------------------------------------------------------
# Обёртка для совместимости с BERT
# ----------------------------------------------------------------------
class LinearSelfAttention(nn.Module):
    """
    Обёртка для совместимости с BERT.
    Позволяет подставить любую реализацию внимания через параметр attention_class.
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
    """
    Замена стандартного внимания на линейное с возможностью выбора класса.
    """
    layer.attention.self = LinearSelfAttention(config, attention_class)
    return layer


# ----------------------------------------------------------------------
# Основная модель BERT с поддержкой замены слоёв
# ----------------------------------------------------------------------
class BertWithCustomAttention(nn.Module):
    """
    BERT-модель с поддержкой замены и добавления слоёв.
    Использует inject_linear_attention с переданным классом.
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
                    attention_mask,
                    head_mask[i] if head_mask is not None else None,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_values[i] if past_key_values is not None else None,
                    output_attentions,
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