import os
import platform
import random
import time
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from transformers import Trainer, TrainerCallback

from logging_utils import write_log

try:
    import psutil
except ImportError:
    psutil = None


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        if isinstance(outputs, dict):
            logits = outputs["logits"]
        else:
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        num_labels = getattr(getattr(self.model, "config", None), "num_labels", logits.shape[-1])

        if hasattr(model, "loss_fct") and model.loss_fct is not None:
            loss = model.loss_fct(logits.view(-1, num_labels), labels.view(-1))
        else:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def get_gpu_memory_usage(device="cuda", detail: bool = False) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "allocated_mb": 0,
            "reserved_mb": 0,
            "max_allocated_mb": 0,
            "free_mb": 0,
            "total_mb": 0,
        }

    allocated = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
    total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**2
    free_memory = total_memory - allocated

    result = {
        "allocated_mb": allocated,
        "reserved_mb": reserved,
        "max_allocated_mb": max_allocated,
        "free_mb": free_memory,
        "total_mb": total_memory,
    }

    if detail:
        result["cached_mb"] = reserved
        device_props = torch.cuda.get_device_properties(device)
        result["gpu_name"] = device_props.name
        result["gpu_compute_capability"] = f"{device_props.major}.{device_props.minor}"
        result["gpu_multiprocessor_count"] = device_props.multi_processor_count

    return result


def get_system_info() -> Dict[str, Any]:
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if psutil is not None:
        info["cpu_count"] = psutil.cpu_count(logical=True)
        info["cpu_physical_count"] = psutil.cpu_count(logical=False)
        info["ram_total_gb"] = psutil.virtual_memory().total / (1024**3)
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
        info["current_device"] = torch.cuda.current_device()
    return info


def reset_gpu_memory_stats(device="cuda"):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def log_detailed_memory(phase: str, device="cuda"):
    mem = get_gpu_memory_usage(device, detail=True)
    write_log(
        f"[MEMORY] {phase}: allocated={mem['allocated_mb']:.2f} MB, "
        f"reserved={mem['reserved_mb']:.2f} MB, "
        f"peak={mem['max_allocated_mb']:.2f} MB"
    )


class DetailedMemoryCallback(TrainerCallback):
    def __init__(self):
        self.epoch_memory = {}
        self.current_epoch = 0

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.current_epoch = int(state.epoch) if state.epoch is not None else 0
        reset_gpu_memory_stats()

    def on_epoch_end(self, args, state, control, **kwargs):
        mem = get_gpu_memory_usage()
        epoch = int(state.epoch) if state.epoch is not None else 0
        self.epoch_memory[epoch] = {
            "peak": mem["max_allocated_mb"],
            "final": mem["allocated_mb"],
            "reserved": mem["reserved_mb"],
        }
        write_log(f"  Пиковая память за эпоху {epoch}: {mem['max_allocated_mb']:.2f} MB")

    def get_memory_stats(self):
        if not self.epoch_memory:
            return {"peak_memory_overall": 0, "avg_epoch_peak": 0}
        peak_values = [m["peak"] for m in self.epoch_memory.values()]
        return {
            "peak_memory_overall": max(peak_values),
            "avg_epoch_peak": np.mean(peak_values),
        }


class EpochTrainingTimeCallback(TrainerCallback):
    def __init__(self, system_warmup_epochs=2):
        self.epoch_start_time = None
        self.epoch_train_times = {}
        self.system_warmup_epochs = system_warmup_epochs
        self.epoch_metrics = []

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.epoch_start_time is None:
            return

        epoch_duration = time.time() - self.epoch_start_time
        epoch_int = int(round(state.epoch)) if state.epoch is not None else 0
        self.epoch_train_times[epoch_int] = epoch_duration
        write_log(f"[Эпоха {epoch_int}] Время обучения: {epoch_duration:.2f} сек (без валидации)")
        self.epoch_start_time = None

    def get_epoch_train_time(self, epoch_int):
        return self.epoch_train_times.get(epoch_int, 0.0)

    def get_stable_epoch_stats(self):
        all_times = [
            t for epoch, t in sorted(self.epoch_train_times.items())
            if epoch > self.system_warmup_epochs
        ]
        if not all_times:
            all_times = list(self.epoch_train_times.values())

        if not all_times:
            return 0.0, 0.0, self.system_warmup_epochs, 0

        avg_time = float(np.mean(all_times))
        std_time = float(np.std(all_times))
        return avg_time, std_time, self.system_warmup_epochs, len(all_times)


class PositionalParamsCallback(TrainerCallback):
    def __init__(self):
        self.positional_params = {}

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None or not hasattr(model, "bert"):
            return

        epoch = int(round(state.epoch)) if state.epoch is not None else 0
        epoch_params = {}

        for layer_num, layer in enumerate(model.bert.encoder.layer):
            attention = getattr(getattr(layer, "attention", None), "self", None)
            head_scales = getattr(attention, "head_scales", None)
            if head_scales is None:
                continue

            head_scales = head_scales.detach().cpu().tolist()
            write_log(f"Слой {layer_num}:")
            write_log(f"  head_scales = {head_scales}")
            for i, scale in enumerate(head_scales):
                epoch_params[f"layer_{layer_num}_head_{i}_scale"] = float(scale)

        write_log(f"--- Конец эпохи {epoch} ---")
        self.positional_params[epoch] = epoch_params


class EvalMetricsLoggingCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return control

        eval_loss = metrics.get("eval_loss")
        eval_accuracy = metrics.get("eval_accuracy")
        eval_f1 = metrics.get("eval_f1_macro")

        write_log(f"\n[Эпоха {state.epoch:.1f}] EVAL МЕТРИКИ:")
        if eval_loss is not None:
            write_log(f"  eval_loss: {eval_loss:.4f}")
        if eval_accuracy is not None:
            write_log(f"  eval_accuracy: {eval_accuracy:.4f}")
        if eval_f1 is not None:
            write_log(f"  eval_f1_macro: {eval_f1:.4f}")
        return control


class EarlyStopOnMetricThreshold(TrainerCallback):
    def __init__(self, metric_name: str, threshold: float):
        self.metric_name = metric_name
        self.threshold = threshold

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return control

        metric_value = metrics.get(self.metric_name)
        if metric_value is not None and metric_value >= self.threshold:
            write_log(
                f"Достигнуто значение {self.metric_name}={metric_value:.4f} >= {self.threshold:.4f}, "
                "останавливаем обучение"
            )
            control.should_training_stop = True
        return control


class EarlyStopOnMetricsThreshold(TrainerCallback):
    def __init__(self, threshold: float = 1.0, metric_names=None):
        self.threshold = threshold
        self.metric_names = tuple(metric_names or ("eval_accuracy", "eval_f1_macro"))

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return control

        metric_values = {name: metrics.get(name) for name in self.metric_names}
        if any(value is None for value in metric_values.values()):
            return control

        if all(value >= self.threshold for value in metric_values.values()):
            formatted = ", ".join(f"{name}={value:.4f}" for name, value in metric_values.items())
            write_log(
                f"Достигнуты пороги метрик: {formatted} >= {self.threshold:.4f}. "
                "Останавливаем обучение"
            )
            control.should_training_stop = True
        return control


class EarlyStoppingOnPatience(TrainerCallback):
    def __init__(self, patience: int, metric_name: str = "eval_accuracy", min_delta: float = 0.0):
        self.patience = patience
        self.metric_name = metric_name
        self.min_delta = min_delta
        self.best_metric = None
        self.bad_epochs = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or self.patience <= 0:
            return control

        metric_value = metrics.get(self.metric_name)
        if metric_value is None:
            return control

        if self.best_metric is None or metric_value > self.best_metric + self.min_delta:
            self.best_metric = metric_value
            self.bad_epochs = 0
            write_log(
                f"Лучшая метрика {self.metric_name} обновлена: {metric_value:.4f}"
            )
            return control

        self.bad_epochs += 1
        write_log(
            f"Нет улучшения {self.metric_name}: {metric_value:.4f}, "
            f"ожидание {self.bad_epochs}/{self.patience} эпох"
        )
        if self.bad_epochs >= self.patience:
            write_log(
                f"Останавливаем обучение: {self.patience} эпох без улучшения {self.metric_name}"
            )
            control.should_training_stop = True
        return control
