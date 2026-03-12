import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Отключение предупреждения о параллелизме токенизаторов
import time
import json
from typing import Tuple, Dict, Any, Optional
import numpy as np
import random
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
from transformers import (
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    TrainingArguments,
    TrainerCallback,
    Trainer,
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification
)

from custom_attention import (
    BertWithCustomAttention,
    LinearContextAttention,
    LinearContextAttentionPosEnc,
    LinearContextAttentionDilated,
    LinearContextAttentionWeighted
)
from dataset_utils import prepare_datasets, ReviewDataset
from loader import TransformerLoader
from logging_utils import append_train_results, set_log_file, write_log

# Словарь для выбора типа внимания
ATTENTION_CLASSES = {
    'base': LinearContextAttention,
    'pos-enc': LinearContextAttentionPosEnc,
    'dilated': LinearContextAttentionDilated,
    'weighted': LinearContextAttentionWeighted
}

def set_seed(seed: int = 42):
    """Фиксация всех random seed для воспроизводимости"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def evaluate_by_length_bins(model, dataset, tokenizer, class_names, device, 
                            max_len, step=128):
    """
    Оценка модели на группах отзывов, разбитых по длине с шагом step.
    Группы: [0, step), [step, 2*step), ..., [max_len-step, max_len], и последняя > max_len-step.
    
    Args:
        model: обученная модель
        dataset: датасет для оценки (например, тестовый)
        tokenizer: токенизатор
        class_names: названия классов
        device: устройство
        max_len: максимальная длина (max_position_embeddings)
        step: шаг разбиения (по умолчанию 128)
    
    Returns:
        словарь с метриками по группам
    """
    model.eval()
    
    # Создаём границы бинов
    bins = list(range(0, max_len, step))
    if bins[-1] < max_len:
        bins.append(max_len)
    bin_labels = []
    for i in range(len(bins)-1):
        bin_labels.append(f"{bins[i]}-{bins[i+1]}")
    
    # Инициализируем группы
    groups = {label: {'preds': [], 'labels': [], 'count': 0} for label in bin_labels}
    groups['all'] = {'preds': [], 'labels': [], 'count': 0}
    
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(tokenizer, padding=True, pad_to_multiple_of=8)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=data_collator)
    
    # Замеряем время для всех групп
    total_start_time = time.time()
    
    # Для замера времени по группам создадим отдельные словари
    group_times = {label: 0.0 for label in bin_labels}
    group_times['all'] = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            lengths = batch['attention_mask'].sum(dim=1).cpu().numpy()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Замеряем время forward pass для этого батча
            batch_start = time.time()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_time = time.time() - batch_start
            
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            for i, length in enumerate(lengths):
                groups['all']['preds'].append(preds[i])
                groups['all']['labels'].append(labels_np[i])
                groups['all']['count'] += 1
                
                bin_idx = min(int(length // step), len(bin_labels)-1)
                bin_label = bin_labels[bin_idx]
                groups[bin_label]['preds'].append(preds[i])
                groups[bin_label]['labels'].append(labels_np[i])
                groups[bin_label]['count'] += 1
                
                group_times[bin_label] += batch_time / len(lengths)
                group_times['all'] += batch_time / len(lengths)
    
    total_time = time.time() - total_start_time
    
    # Вычисляем метрики для каждой группы
    results = {}
    output_lines = []
    output_lines.append("\n" + "="*70)
    output_lines.append("ОЦЕНКА НА ТЕСТЕ ПО ГРУППАМ ДЛИНЫ (ШАГ 128)")
    output_lines.append("="*70)
    
    for label, data in groups.items():
        if data['count'] == 0:
            output_lines.append(f"\n{label}: нет примеров")
            continue
            
        preds = np.array(data['preds'])
        labels_arr = np.array(data['labels'])
        accuracy = accuracy_score(labels_arr, preds)
        f1_macro = f1_score(labels_arr, preds, average='macro', zero_division=0)
        f1_weighted = f1_score(labels_arr, preds, average='weighted', zero_division=0)
        
        avg_time_per_sample_ms = (group_times[label] / data['count']) * 1000 if data['count'] > 0 else 0
        
        output_lines.append(f"\n{label} (n={data['count']}):")
        output_lines.append(f"  Accuracy: {accuracy:.4f}")
        output_lines.append(f"  F1-macro: {f1_macro:.4f}")
        output_lines.append(f"  F1-weighted: {f1_weighted:.4f}")
        output_lines.append(f"  Total time: {group_times[label]:.4f} сек")  # ← 4 знака
        output_lines.append(f"  Avg time per sample: {avg_time_per_sample_ms:.2f} мс")
        
        results[label] = {
            'count': data['count'],
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'time_sec': group_times[label],
            'time_per_sample_ms': avg_time_per_sample_ms
        }
    
    output_lines.append("\n" + "="*70)
    output_lines.append(f"ВСЕГО (n={groups['all']['count']}):")
    output_lines.append(f"  Total inference time: {total_time:.4f} сек")  # ← 4 знака
    output_lines.append(f"  Avg time per sample: {(total_time/groups['all']['count'])*1000:.2f} мс")
    output_lines.append("="*70 + "\n")
    
    # Один вывод на экран и в лог
    full_output = "\n".join(output_lines)
    print(full_output)
    write_log(full_output)
    
    return results


def load_config_from_json(json_path: str, config_name: str) -> Dict[str, Any]:
    """
    Загрузка конфигурации из JSON файла.
    
    Args:
        json_path: путь к JSON файлу с конфигурациями
        config_name: имя конфигурации (например, "5class_amazon")
    
    Returns:
        словарь с конфигурацией
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        all_configs = json.load(f)
    
    if config_name not in all_configs:
        raise KeyError(f"Конфигурация '{config_name}' не найдена в {json_path}. Доступные: {list(all_configs.keys())}")
    
    config_data = all_configs[config_name]
    
    # Проверка структуры
    if "training_args" not in config_data or "bert_config" not in config_data:
        raise ValueError(f"Конфигурация '{config_name}' должна содержать поля 'training_args' и 'bert_config'")
    
    return config_data


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """
    Рекурсивно разворачивает вложенный словарь в плоский.
    
    Args:
        d: словарь для разворачивания
        parent_key: родительский ключ
        sep: разделитель
    
    Returns:
        плоский словарь
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def save_all_results_to_csv(
    output_path: str,
    run_params: Dict[str, Any],
    training_args_config: Dict[str, Any],
    bert_config_params: Dict[str, Any],
    final_metrics: Dict[str, Any],
    epoch_metrics: list,
    memory_stats: Dict[str, Any],
    system_info: Dict[str, Any],
    test_length_metrics: Optional[Dict[str, Any]] = None,
    attention_type: str = None,
):
    """
    Сохраняет все результаты в CSV файл (одна строка на эксперимент).
    
    Args:
        output_path: путь к выходному CSV файлу
        run_params: параметры запуска (replace, add, remove и т.д.)
        training_args_config: конфигурация TrainingArguments
        bert_config_params: конфигурация BertConfig
        final_metrics: финальные метрики
        epoch_metrics: метрики по эпохам (будут сохранены отдельно)
        memory_stats: статистика по памяти (только итоговая)
        system_info: системная информация
        test_length_metrics: метрики на тесте по длинам
    """
    # Основные метрики (без дублирования)
    main_metrics = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        'experiment_id': run_params.get('experiment_id', 'unknown'),
        'attention_type': attention_type if attention_type else 'unknown',
        
        # Основные метрики
        'train_loss': final_metrics.get('train_loss', ''),
        'eval_loss': final_metrics.get('eval_loss', ''),
        'eval_accuracy': final_metrics.get('eval_accuracy', ''),
        'eval_f1_macro': final_metrics.get('eval_f1_macro', ''),
        'eval_mae': final_metrics.get('eval_mae', ''),
        'eval_precision_macro': final_metrics.get('eval_precision_macro', ''),
        'eval_recall_macro': final_metrics.get('eval_recall_macro', ''),
        'best_metric': final_metrics.get('best_metric', ''),
        
        # Временные метрики
        'total_training_time_sec': final_metrics.get('total_training_time_sec', ''),
        'avg_epoch_time_sec': final_metrics.get('avg_epoch_time_sec', ''),
        'std_epoch_time_sec': final_metrics.get('std_epoch_time_sec', ''),
        'epochs_completed': final_metrics.get('epochs_completed', ''),
        
        # Параметры замены слоёв (НЕ из run_params, чтобы избежать дублей)
        'num_layers_replace': run_params.get('num_layers_to_replace', ''),
        'num_layers_add': run_params.get('num_layers_to_add', ''),
        'num_layers_remove': run_params.get('num_layers_to_remove', ''),
        
        # Размеры датасета
        'train_dataset_size': run_params.get('train_dataset_size', ''),
        'eval_dataset_size': run_params.get('eval_dataset_size', ''),
        
        # Память (только итоговая)
        'peak_memory_gpu_mb': memory_stats.get('peak_memory_overall', 0),
        'avg_epoch_peak_memory_mb': memory_stats.get('avg_epoch_peak', 0),
    }
    
    # Добавляем метрики по классам (включая precision и recall)
    class_indices = ['1', '2', '3', '4', '5']
    for idx in class_indices:
        class_key = f'class_{idx}'
        # Ищем в final_metrics ключи с этим классом
        for key, value in final_metrics.items():
            if key.startswith(f'{class_key}_'):
                main_metrics[key] = value
    
    # Добавляем метрики по длинам, если есть
    if test_length_metrics:
        for bin_label, metrics in test_length_metrics.items():
            if bin_label == 'all':
                prefix = 'test_all'
            else:
                # Преобразуем "0-128" в "test_0_128"
                prefix = f"test_{bin_label.replace('-', '_')}"
            
            main_metrics[f'{prefix}_count'] = metrics.get('count', '')
            main_metrics[f'{prefix}_accuracy'] = metrics.get('accuracy', '')
            main_metrics[f'{prefix}_f1_macro'] = metrics.get('f1_macro', '')
            main_metrics[f'{prefix}_f1_weighted'] = metrics.get('f1_weighted', '')
            main_metrics[f'{prefix}_time_sec'] = metrics.get('time_sec', '')
            main_metrics[f'{prefix}_time_per_sample_ms'] = metrics.get('time_per_sample_ms', '')
    
    # Создаем плоские словари для гиперпараметров (НО исключаем дублирующие поля)
    # Убираем из flat_run_params те поля, которые уже есть в main_metrics
    exclude_keys = ['num_layers_to_replace', 'num_layers_to_add', 'num_layers_to_remove', 
                    'train_dataset_size', 'eval_dataset_size', 'experiment_id']
    
    flat_run_params = flatten_dict({k: v for k, v in run_params.items() if k not in exclude_keys}, 'run')
    flat_training_args = flatten_dict(training_args_config, 'training_arg')
    flat_bert_config = flatten_dict(bert_config_params, 'bert_config')
    
    # Объединяем всё в один словарь
    all_params = {}
    all_params.update(main_metrics)
    all_params.update(flat_run_params)
    all_params.update(flat_training_args)
    all_params.update(flat_bert_config)
    all_params.update(system_info)
    
    # Создаем DataFrame
    df = pd.DataFrame([all_params])
    
    # Проверяем, существует ли файл и не пуст ли он
    file_exists = os.path.isfile(output_path)
    
    if file_exists and os.path.getsize(output_path) > 0:
        try:
            existing_df = pd.read_csv(output_path)
            df = pd.concat([existing_df, df], ignore_index=True)
        except pd.errors.EmptyDataError:
            # Файл пуст - просто сохраняем новый
            pass
    
    df.to_csv(output_path, index=False, encoding='utf-8')
    write_log(f"Финальные результаты сохранены в {output_path}")
    
    # Сохраняем отдельный файл с метриками по эпохам
    if epoch_metrics:
        epoch_df_path = output_path.replace('.csv', '_epochs.csv')
        epoch_df = pd.DataFrame(epoch_metrics)
        if os.path.exists(epoch_df_path) and os.path.getsize(epoch_df_path) > 0:
            try:
                existing_epoch_df = pd.read_csv(epoch_df_path)
                epoch_df = pd.concat([existing_epoch_df, epoch_df], ignore_index=True)
            except pd.errors.EmptyDataError:
                pass
        epoch_df.to_csv(epoch_df_path, index=False, encoding='utf-8')
        write_log(f"Метрики по эпохам сохранены в {epoch_df_path}")


class SimpleModelContainer:
    """Простой контейнер для модели и токенизатора, совместимый с интерфейсом TransformerLoader."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer


# === Мониторинг памяти GPU ===
def get_gpu_memory_usage(device="cuda", detail: bool = False):
    """
    Текущее использование памяти GPU в МБ.
    
    Args:
        device: устройство CUDA
        detail: если True, возвращает более детальную информацию
    
    Returns:
        словарь с информацией о памяти
    """
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0, "free_mb": 0.0}
    
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
    
    # Получаем общую память GPU
    total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**2
    free_memory = total_memory - allocated
    
    result = {
        "allocated_mb": allocated,
        "reserved_mb": reserved,
        "max_allocated_mb": max_allocated,
        "free_mb": free_memory,
        "total_mb": total_memory,
        "utilization_percent": (allocated / total_memory) * 100 if total_memory > 0 else 0
    }
    
    if detail:
        # Дополнительная информация о распределении памяти
        if hasattr(torch.cuda, 'memory_reserved'):
            result["cached_mb"] = torch.cuda.memory_reserved(device) / 1024**2
        
        # Информация о GPU
        device_props = torch.cuda.get_device_properties(device)
        result["gpu_name"] = device_props.name
        result["gpu_compute_capability"] = f"{device_props.major}.{device_props.minor}"
        result["gpu_multiprocessor_count"] = device_props.multi_processor_count
        
    return result


def get_system_info() -> Dict[str, Any]:
    """Получение системной информации."""
    import platform
    import psutil
    
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(logical=True),
        "cpu_physical_count": psutil.cpu_count(logical=False),
        "ram_total_gb": psutil.virtual_memory().total / (1024**3),
        "ram_available_gb": psutil.virtual_memory().available / (1024**3),
    }
    
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        info["device_count"] = torch.cuda.device_count()
        info["current_device"] = torch.cuda.current_device()
    
    return info


def reset_gpu_memory_stats(device="cuda"):
    """Сброс пиковых счётчиков памяти перед измерением."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()


def log_detailed_memory(phase: str, device="cuda"):
    """
    Логирование детальной информации о памяти GPU.
    
    Args:
        phase: фаза выполнения (forward, backward, и т.д.)
        device: устройство CUDA
    """
    if not torch.cuda.is_available():
        return
    
    mem = get_gpu_memory_usage(device, detail=True)
    write_log(f"\n{'='*60}")
    write_log(f"ПАМЯТЬ GPU [{phase.upper()}]:")
    write_log(f"  Аллоцировано: {mem['allocated_mb']:.2f} MB ({mem['utilization_percent']:.1f}%)")
    write_log(f"  Зарезервировано: {mem['reserved_mb']:.2f} MB")
    write_log(f"  Свободно: {mem['free_mb']:.2f} MB")
    write_log(f"  Всего: {mem['total_mb']:.2f} MB")
    write_log(f"  Пиковое использование (за сессию): {mem['max_allocated_mb']:.2f} MB")
    if "gpu_name" in mem:
        write_log(f"  GPU: {mem['gpu_name']}")
    write_log('='*60 + "\n")


class DetailedMemoryCallback(TrainerCallback):
    """
    Измерение пиковой памяти GPU на каждой эпохе.
    """
    
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
            "reserved": mem["reserved_mb"]
        }
        
        write_log(f"  Пиковая память за эпоху {epoch}: {mem['max_allocated_mb']:.2f} MB")
    
    def get_memory_stats(self):
        """Возвращает статистику по памяти."""
        if not self.epoch_memory:
            return {
                "peak_memory_overall": 0,
                "avg_epoch_peak": 0,
            }
        
        peak_values = [m["peak"] for m in self.epoch_memory.values()]
        
        return {
            "peak_memory_overall": max(peak_values),
            "avg_epoch_peak": np.mean(peak_values),
            "epoch_memory": self.epoch_memory,
        }


def log_classification_report(model, eval_dataset, tokenizer, class_names, device="cuda"):
    """
    Генерация и логирование classification report для финальной эпохи.
    
    Args:
        model: обученная модель
        eval_dataset: валидационный датасет
        tokenizer: токенизатор
        class_names: названия классов
        device: устройство
    
    Returns:
        словарь с метриками
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    # Создаем DataLoader для инференса с DataCollatorWithPadding
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(tokenizer, padding=True, pad_to_multiple_of=8)
    dataloader = DataLoader(eval_dataset, batch_size=64, shuffle=False, collate_fn=data_collator)
    
    write_log("\n" + "="*70)
    write_log("CLASSIFICATION REPORT - ФИНАЛЬНАЯ ЭПОХА")
    write_log("="*70)
    
    with torch.no_grad():
        for batch in dataloader:
            # Перемещаем данные на устройство
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass с измерением памяти
            reset_gpu_memory_stats()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Получаем предсказания
            preds = outputs.logits.argmax(dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Конвертируем в numpy массивы
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Вычисляем метрики
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    write_log(f"\nОБЩИЕ МЕТРИКИ:")
    write_log(f"  Accuracy:  {accuracy:.4f}")
    write_log(f"  F1-macro:  {f1_macro:.4f}")
    write_log(f"  F1-weighted: {f1_weighted:.4f}")
    
    write_log(f"\nДЕТАЛЬНЫЙ ОТЧЁТ ПО КЛАССАМ:")
    
    # Получаем classification report как словарь
    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4,
        zero_division=0,
        output_dict=True
    )
    
    # Выводим построчно для лучшего форматирования
    report_str = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
    for line in report_str.split('\n'):
        if line.strip():
            write_log(f"  {line}")
    
    # Дополнительная статистика по каждому классу
    class_metrics = {}
    
    for i, class_name in enumerate(class_names):
    # Пробуем оба варианта ключа: индекс и имя класса
        class_key = str(i)
        if class_key in report:
            class_data = report[class_key]
        elif class_name in report:
            class_data = report[class_name]
        else:
            write_log(f"  ВНИМАНИЕ: класс {class_name} (индекс {i}) не найден в report")
            continue
        
        # Сохраняем метрики
        class_metrics[f"class_{i+1}_accuracy"] = class_data['precision']
        class_metrics[f"class_{i+1}_precision"] = class_data['precision']
        class_metrics[f"class_{i+1}_recall"] = class_data['recall']
        class_metrics[f"class_{i+1}_f1"] = class_data['f1-score']
        class_metrics[f"class_{i+1}_count"] = int(class_data['support'])
    
    # Формируем словарь с метриками
    metrics = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "classification_report": report,
        **class_metrics
    }
    
    return metrics


class EpochTrainingTimeCallback(TrainerCallback):
    """
    Измерение времени обучения за эпоху и статистика по стабильной фазе.
    
    Исключаются только первые эпохи системного прогрева оборудования
    (кэширование CUDA, инициализация буферов). Warmup расписания обучения не влияет на время.
    """

    def __init__(self, system_warmup_epochs=2):
        """
        Args:
            system_warmup_epochs: количество первых эпох для исключения (системный прогрев)
        """
        self.epoch_start_time = None
        self.epoch_train_times = {}  # epoch_int → время обучения в секундах
        self.system_warmup_epochs = system_warmup_epochs
        self.epoch_metrics = []  # Для хранения метрик по эпохам

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()
        # Измеряем память в начале эпох

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.epoch_start_time is not None:
            epoch_duration = time.time() - self.epoch_start_time
            epoch_int = int(round(state.epoch)) if state.epoch is not None else 0
            self.epoch_train_times[epoch_int] = epoch_duration
            
            write_log(f"[Эпоха {epoch_int}] Время обучения: {epoch_duration:.2f} сек (без валидации)")
            
            # Сохраняем метрики эпохи
            if hasattr(state, 'log_history') and state.log_history:
                for log in state.log_history:
                    if log.get('epoch', 0) == epoch_int and 'eval_loss' in log:
                        self.epoch_metrics.append({
                            'epoch': epoch_int,
                            'train_time': epoch_duration,
                            'eval_loss': log.get('eval_loss'),
                            'eval_accuracy': log.get('eval_accuracy'),
                            'eval_f1_macro': log.get('eval_f1_macro'),
                            'eval_mae': log.get('eval_mae', None)
                        })
            
            self.epoch_start_time = None

    def get_epoch_train_time(self, epoch_int):
        """Время обучения для конкретной эпохи."""
        return self.epoch_train_times.get(epoch_int, 0.0)

    def get_stable_epoch_stats(self):
        """
        Статистика по стабильной фазе (после системного прогрева).
        
        Returns:
            avg_time: среднее время эпохи (сек)
            std_time: стандартное отклонение (сек)
            excluded_epochs: количество исключённых эпох
            stable_epochs: список эпох в расчёте
        """
        if not self.epoch_train_times:
            return 0.0, 0.0, 0, []

        excluded_epochs = self.system_warmup_epochs
        stable_epochs = [e for e in sorted(self.epoch_train_times.keys()) if e >= excluded_epochs]

        # Защита: если эпох меньше, чем исключаемых
        if not stable_epochs:
            stable_epochs = sorted(self.epoch_train_times.keys())
            excluded_epochs = 0

        times = [self.epoch_train_times[e] for e in stable_epochs]
        avg_time = sum(times) / len(times)
        std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5 if len(times) > 1 else 0.0

        return avg_time, std_time, excluded_epochs, stable_epochs
    
    def get_epoch_metrics(self):
        """Возвращает метрики по эпохам."""
        return self.epoch_metrics


class PositionalParamsCallback(TrainerCallback):
    """Логирование параметров позиционного кодирования после каждой эпохи."""

    def __init__(self):
        write_log("\n" + "=" * 60)
        write_log("ЛОГ ПАРАМЕТРОВ ПОЗИЦИОННОГО КОДИРОВАНИЯ")
        write_log("=" * 60)
        self.positional_params = {}

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        model_ref = model if model is not None else getattr(self, 'model', None)
        if model_ref is None:
            return

        epoch = int(state.epoch) if state.epoch is not None else 0
        write_log(f"\n--- Эпоха {epoch} ---")

        epoch_params = {}

        for name, module in model_ref.named_modules():
            # Проверяем наличие атрибута head_scales – это надёжнее, чем проверка имени класса
            if hasattr(module, 'head_scales'):
                # Определяем номер слоя (упрощённо)
                layer_num = name.split('.')[-3] if 'layer' in name else "final"
                head_scales = module.head_scales.data.cpu().numpy()

                write_log(f"Слой {layer_num}:")
                write_log(f"  head_scales = {head_scales}")

                for i, scale in enumerate(head_scales):
                    epoch_params[f"layer_{layer_num}_head_{i}_scale"] = float(scale)

        write_log(f"--- Конец эпохи {epoch} ---")
        self.positional_params[epoch] = epoch_params


def train_custom_bert(
    train_dataset_path: str,
    val_dataset_path: str,
    num_layers_to_replace: int,
    num_layers_to_add: int,
    num_layers_to_remove: int,
    bert_loader: TransformerLoader,
    run_log_dir: str,
    training_args_config: Dict[str, Any],
    bert_config_params: Dict[str, Any],
    class_names: Optional[list],
    experiment_id: str,
    output_path: str,
    n_folds: int = 1,
    attention_class = LinearContextAttentionDilated,
    random_state: int = 42,
) -> Tuple[TransformerLoader, torch.nn.Module, Dict[str, Any]]:
    """
    Дообучение BERT с возможностью кросс-валидации.
    
    Args:
        ... (все прежние аргументы)
        attention_class: класс реализации линейного внимания (base, pos-enc, dilated)
    """
    set_seed(random_state)
    base_model = bert_loader.model

    write_log(
        f"Конфиг модели: num_labels={base_model.config.num_labels}, "
        f"hidden_size={base_model.config.hidden_size}, "
        f"num_attention_heads={base_model.config.num_attention_heads}"
    )

    # Удаление слоёв (общее для всех режимов)
    if num_layers_to_remove > 0:
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
            write_log(
                f"Удаление слоёв: всего было {total_layers}, удалено {num_layers_to_remove}, осталось {num_layers_to_keep}"
            )

    # Проверка наличия train датасета
    if not os.path.exists(train_dataset_path):
        write_log(f"Train dataset {train_dataset_path} не найден.")
        return bert_loader, base_model, {}

    # Загрузка train датасета (нужен для обоих режимов)
    train_dataset = ReviewDataset(
        csv_path=train_dataset_path,
        tokenizer=bert_loader.tokenizer,
        max_length=bert_config_params.get("max_position_embeddings", 768)
    )

    # Кросс-валидация
    if n_folds > 1:
        write_log(f"\n{'='*60}")
        write_log(f"Запуск {n_folds}-fold кросс-валидации на train датасете")
        write_log(f"Размер train датасета: {len(train_dataset)}")
        write_log('='*60)

        # Получаем метки для стратификации
        labels = [train_dataset[i]['labels'] for i in range(len(train_dataset))]
        
        # Стратифицированное разбиение
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
            write_log(f"\n{'='*60}")
            write_log(f"FOLD {fold+1}/{n_folds}")
            write_log('='*60)
            
            # Создаём подмножества
            train_subset = torch.utils.data.Subset(train_dataset, train_idx)
            val_subset = torch.utils.data.Subset(train_dataset, val_idx)
            
            # Создаём новую модель для каждого фолда
            config = BertConfig.from_pretrained(
                bert_config_params["pretrained_model_name"],
                num_hidden_layers=bert_config_params["num_hidden_layers"],
                hidden_size=bert_config_params["hidden_size"],
                num_attention_heads=bert_config_params["num_attention_heads"],
                intermediate_size=bert_config_params["intermediate_size"],
                num_labels=bert_config_params["num_labels"],
                max_position_embeddings=bert_config_params["max_position_embeddings"],
                hidden_dropout_prob=bert_config_params["hidden_dropout_prob"],
                attention_probs_dropout_prob=bert_config_params["attention_probs_dropout_prob"],
                problem_type=bert_config_params["problem_type"]
            )
            fold_model = BertForSequenceClassification(config)
            torch.manual_seed(random_state + fold)
            
            # Применяем кастомное внимание, если нужно (передаём класс)
            if num_layers_to_replace > 0 or num_layers_to_add > 0:
                fold_model = BertWithCustomAttention(
                    fold_model,
                    num_layers_to_replace=num_layers_to_replace,
                    num_layers_to_add=num_layers_to_add,
                    attention_class=attention_class,  # ← передаём
                )
            
            # Создаём временный контейнер
            fold_loader = SimpleModelContainer(model=fold_model, tokenizer=bert_loader.tokenizer)
            
            # Директория для логов фолда
            fold_log_dir = os.path.join(run_log_dir, f"fold_{fold+1}")
            os.makedirs(fold_log_dir, exist_ok=True)
            
            # Параметры обучения
            batch_size = training_args_config["per_device_train_batch_size"]
            steps_per_epoch = max(1, len(train_subset) // batch_size)
            logging_steps = max(1, int(steps_per_epoch / 10))

            seed_value = training_args_config.get("seed", random_state)
            
            training_args = TrainingArguments(
                output_dir=os.path.join(fold_log_dir, "checkpoints"),
                optim=training_args_config["optim"],
                num_train_epochs=training_args_config["num_train_epochs"],
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=training_args_config["per_device_eval_batch_size"],
                gradient_accumulation_steps=training_args_config["gradient_accumulation_steps"],
                max_grad_norm=training_args_config["max_grad_norm"],
                learning_rate=training_args_config["learning_rate"],
                warmup_ratio=training_args_config["warmup_ratio"],
                lr_scheduler_type=training_args_config["lr_scheduler_type"],
                weight_decay=training_args_config["weight_decay"],
                eval_strategy=training_args_config["eval_strategy"],
                save_strategy=training_args_config["save_strategy"],
                load_best_model_at_end=training_args_config["load_best_model_at_end"],
                metric_for_best_model=training_args_config["metric_for_best_model"],
                greater_is_better=training_args_config["greater_is_better"],
                fp16=training_args_config.get("fp16", torch.cuda.is_available()),
                bf16=training_args_config.get("bf16", False),
                logging_dir=fold_log_dir,
                logging_steps=logging_steps,
                save_total_limit=training_args_config["save_total_limit"],
                report_to=training_args_config["report_to"],
                dataloader_num_workers=training_args_config["dataloader_num_workers"],
                dataloader_pin_memory=training_args_config["dataloader_pin_memory"],
                disable_tqdm=training_args_config["disable_tqdm"],
                seed=seed_value,
                data_seed=seed_value, 
            )
            
            # Колбэки (упрощённые для CV)
            memory_callback = DetailedMemoryCallback()
            time_callback = EpochTrainingTimeCallback(system_warmup_epochs=training_args_config.get("system_warmup_epochs", 2))
            
            # Trainer
            trainer = Trainer(
                model=fold_model,
                args=training_args,
                train_dataset=train_subset,
                eval_dataset=val_subset,
                compute_metrics=compute_metrics,
                data_collator=DataCollatorWithPadding(
                    tokenizer=bert_loader.tokenizer,
                    padding=True,
                    pad_to_multiple_of=8
                ),
                callbacks=[memory_callback, time_callback],
            )
            
            # Обучение
            write_log(f"Начало обучения фолда {fold+1}...")
            train_output = trainer.train()
            
            # Оценка
            eval_metrics = trainer.evaluate()
            
            # Сохраняем результаты фолда
            fold_result = {
                'fold': fold+1,
                'eval_accuracy': eval_metrics.get('eval_accuracy'),
                'eval_f1_macro': eval_metrics.get('eval_f1_macro'),
                'eval_loss': eval_metrics.get('eval_loss'),
                'train_loss': train_output.metrics.get('train_loss') if hasattr(train_output, 'metrics') else None,
            }
            fold_results.append(fold_result)
            
            write_log(f"Фолд {fold+1} завершён. Accuracy: {fold_result['eval_accuracy']:.4f}, F1: {fold_result['eval_f1_macro']:.4f}")
        
        # Усреднение результатов по фолдам
        avg_accuracy = np.mean([r['eval_accuracy'] for r in fold_results])
        std_accuracy = np.std([r['eval_accuracy'] for r in fold_results])
        avg_f1 = np.mean([r['eval_f1_macro'] for r in fold_results])
        std_f1 = np.std([r['eval_f1_macro'] for r in fold_results])
        
        write_log(f"\n{'='*60}")
        write_log(f"РЕЗУЛЬТАТЫ {n_folds}-FOLD КРОСС-ВАЛИДАЦИИ")
        write_log(f"Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
        write_log(f"F1-macro: {avg_f1:.4f} ± {std_f1:.4f}")
        write_log('='*60)
        
        # Формируем итоговые метрики
        final_metrics = {
            'cv_accuracy_mean': avg_accuracy,
            'cv_accuracy_std': std_accuracy,
            'cv_f1_macro_mean': avg_f1,
            'cv_f1_macro_std': std_f1,
            'n_folds': n_folds,
        }
        
        # Параметры запуска
        seed = training_args_config.get("seed", None)
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
        
        # Сохраняем результаты в CSV
        all_results = {
            "run_params": run_params,
            "training_args": training_args_config,
            "bert_config": bert_config_params,
            "final_metrics": final_metrics,
            "fold_results": fold_results,
            "system_info": get_system_info(),
        }
        
        return bert_loader, base_model, all_results

    # Обычное обучение с train/val split (n_folds == 1)
    else:
        write_log("Запуск обычного обучения с train/val split")
        
        # Проверка наличия val датасета
        if not os.path.exists(val_dataset_path):
            write_log(f"Val dataset {val_dataset_path} не найден. Используем train_dataset для оценки.")
            val_dataset_path = train_dataset_path

        # Загрузка валидационного датасета
        eval_dataset = ReviewDataset(
            csv_path=val_dataset_path,
            tokenizer=bert_loader.tokenizer,
            max_length=bert_config_params.get("max_position_embeddings", 768)
        )

        # Выбор модели: обычный BERT или с кастомным вниманием
        if num_layers_to_replace == 0 and num_layers_to_add == 0:
            write_log("Все параметры replace/add = 0: используем BERT без изменений")
            model_to_train = base_model
        else:
            model_to_train = BertWithCustomAttention(
                base_model,
                num_layers_to_replace=num_layers_to_replace,
                num_layers_to_add=num_layers_to_add,
                attention_class=attention_class,  # ← передаём
            )

        # Директории для чекпоинтов и логов
        checkpoint_dir = os.path.join(run_log_dir, "checkpoints")
        logging_dir = run_log_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Вычисляем logging_steps на основе batch_size
        batch_size = training_args_config["per_device_train_batch_size"]
        steps_per_epoch = max(1, len(train_dataset) // batch_size)
        logging_steps = max(1, int(steps_per_epoch / 10))
        
        # Параметры обучения из JSON
        training_args = TrainingArguments(
            output_dir=checkpoint_dir,
            optim=training_args_config["optim"],
            num_train_epochs=training_args_config["num_train_epochs"],
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=training_args_config["per_device_eval_batch_size"],
            gradient_accumulation_steps=training_args_config["gradient_accumulation_steps"],
            max_grad_norm=training_args_config["max_grad_norm"],
            learning_rate=training_args_config["learning_rate"],
            warmup_ratio=training_args_config["warmup_ratio"],
            lr_scheduler_type=training_args_config["lr_scheduler_type"],
            weight_decay=training_args_config["weight_decay"],
            eval_strategy=training_args_config["eval_strategy"],
            save_strategy=training_args_config["save_strategy"],
            group_by_length=training_args_config.get("group_by_length", False),
            load_best_model_at_end=training_args_config["load_best_model_at_end"],
            metric_for_best_model=training_args_config["metric_for_best_model"],
            greater_is_better=training_args_config["greater_is_better"],
            fp16=training_args_config.get("fp16", torch.cuda.is_available()),
            bf16=training_args_config.get("bf16", False),
            logging_dir=logging_dir,
            logging_steps=logging_steps,
            save_total_limit=training_args_config["save_total_limit"],
            report_to=training_args_config["report_to"],
            dataloader_num_workers=training_args_config["dataloader_num_workers"],
            dataloader_pin_memory=training_args_config["dataloader_pin_memory"],
            disable_tqdm=training_args_config["disable_tqdm"],
        )

        class LogLossCallback(TrainerCallback):
            """Логирование loss на каждом шаге."""

            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs and "loss" in logs and "eval_loss" not in logs:
                    loss_val = logs["loss"]
                    epoch = logs.get("epoch", 0)
                    step = logs.get("step", 0)
                    write_log(f"[Epoch {epoch:.2f}, Step {step}] loss={loss_val:.4f}")

        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=training_args_config["early_stopping_patience"],
            early_stopping_threshold=training_args_config["early_stopping_threshold"],
        )

        def compute_metrics(p):
            """Вычисление метрик для оценки качества."""
            preds = p.predictions.argmax(-1)
            labels = p.label_ids

            from sklearn.metrics import precision_score, recall_score

            mae = np.mean(np.abs(preds - labels))
            precision = precision_score(labels, preds, average="macro", zero_division=0)
            recall = recall_score(labels, preds, average="macro", zero_division=0)
            f1_macro = f1_score(labels, preds, average="macro", zero_division=0)

            print("\n" + "=" * 60)
            print(f"MAE: {mae:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1-macro: {f1_macro:.3f}")
            print("-" * 60)
            
            target_names = class_names[:len(set(labels))]
            
            print("Детальный отчёт по классам:")
            print(classification_report(
                labels,
                preds,
                target_names=target_names,
                digits=3,
                zero_division=0
            ))
            print("=" * 60 + "\n")

            return {
                "accuracy": accuracy_score(labels, preds),
                "f1_macro": f1_macro,
                "mae": mae,
                "precision_macro": precision,
                "recall_macro": recall,
            }

        data_collator = DataCollatorWithPadding(
            tokenizer=bert_loader.tokenizer,
            padding=True,
            pad_to_multiple_of=8
        )

        # Измерение памяти до обучения
        reset_gpu_memory_stats()
        log_detailed_memory("до обучения")

        # Колбэки (УДАЛЁН LengthMetricsCallback)
        detailed_memory_callback = DetailedMemoryCallback()
        time_callback = EpochTrainingTimeCallback(system_warmup_epochs=training_args_config.get("system_warmup_epochs", 2))
        positional_params_callback = PositionalParamsCallback()

        trainer = Trainer(
            model=model_to_train,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            callbacks=[LogLossCallback(), positional_params_callback, early_stopping, detailed_memory_callback, time_callback],
        )

        # Обучение
        write_log("Начало дообучения...")
        training_start = time.time()
        reset_gpu_memory_stats()
        train_output = trainer.train()
        training_duration = time.time() - training_start
        write_log("Обучение завершено!")
        write_log(f"Общее время обучения ({training_args.num_train_epochs} эпох): {training_duration:.2f} сек")

        # Статистика времени
        avg_time, std_time, warmup_epochs, stable_epochs = time_callback.get_stable_epoch_stats()
        write_log(
            f"Статистика времени обучения:\n"
            f"  Прогрев: первые {warmup_epochs} эпох (исключены из расчёта)\n"
            f"  Среднее время эпохи: {avg_time:.2f} ± {std_time:.2f} сек\n"
            f"  Относительное отклонение: {(std_time / avg_time * 100):.1f}%"
        )

        # Память после обучения
        log_detailed_memory("после обучения")

        # Финальный classification report
        device = next(model_to_train.parameters()).device
        classification_metrics = log_classification_report(
            model=model_to_train,
            eval_dataset=eval_dataset,
            tokenizer=bert_loader.tokenizer,
            class_names=class_names,
            device=device
        )

        # Инференс и память при валидации
        reset_gpu_memory_stats()
        eval_start = time.time()
        eval_metrics = trainer.evaluate()
        eval_duration = time.time() - eval_start
        log_detailed_memory("во время валидации")

        # Финальные метрики
        train_metrics = getattr(train_output, "metrics", {}) if train_output is not None else {}
        final_train_loss = train_metrics.get("train_loss")
        final_eval_loss = eval_metrics.get("eval_loss")
        final_eval_accuracy = eval_metrics.get("eval_accuracy")
        final_eval_f1_macro = eval_metrics.get("eval_f1_macro")
        final_eval_mae = eval_metrics.get("eval_mae")
        final_eval_precision = eval_metrics.get("eval_precision_macro")
        final_eval_recall = eval_metrics.get("eval_recall_macro")

        write_log(
            f"train_loss={final_train_loss}, eval_loss={final_eval_loss}, "
            f"accuracy={final_eval_accuracy}, f1_macro={final_eval_f1_macro}, "
            f"mae={final_eval_mae}, "
            f"train_time={training_duration:.2f}s"
        )

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        # Сохранение пошаговых результатов
        log_history = getattr(trainer.state, "log_history", [])
        last_train_loss_by_epoch = {}
        for log_item in log_history:
            if "loss" in log_item and "eval_loss" not in log_item:
                epoch_val = log_item.get("epoch")
                if epoch_val is None:
                    continue
                epoch_int = int(round(epoch_val))
                last_train_loss_by_epoch[epoch_int] = log_item["loss"]

        epoch_metrics_list = []
        # Создаём отдельный путь для файла с метриками по эпохам
        epoch_output_path = output_path.replace('.csv', '_epochs.csv')

        for log_item in log_history:
            if "eval_loss" not in log_item:
                continue
            epoch_val = log_item.get("epoch")
            if epoch_val is None:
                continue
            epoch_int = int(round(epoch_val))

            train_loss_epoch = last_train_loss_by_epoch.get(epoch_int, "")
            eval_loss_epoch = log_item.get("eval_loss", "")
            eval_acc_epoch = log_item.get("eval_accuracy", "")
            eval_f1_epoch = log_item.get("eval_f1_macro", "")
            eval_mae_epoch = log_item.get("eval_mae", "")
            epoch_train_time = time_callback.get_epoch_train_time(epoch_int)

            epoch_metrics = {
                "epoch": epoch_int,
                "timestamp": timestamp,
                "train_loss": train_loss_epoch,
                "eval_loss": eval_loss_epoch,
                "train_time_sec": round(epoch_train_time, 2),
                "eval_accuracy": eval_acc_epoch,
                "eval_f1_macro": eval_f1_epoch,
                "eval_mae": eval_mae_epoch,
                "num_layers_replace": num_layers_to_replace,
                "num_layers_add": num_layers_to_add,
                "num_layers_remove": num_layers_to_remove,
            }
            
            epoch_metrics_list.append(epoch_metrics)
            # Используем отдельный файл для эпох
            append_train_results(epoch_metrics, epoch_output_path)

        epochs_completed = getattr(trainer.state, "epoch", None)
        if epochs_completed is not None:
            epochs_completed = round(epochs_completed, 2)

        # Финальные метрики
        best_metric = trainer.state.best_metric if hasattr(trainer.state, 'best_metric') else None

        final_metrics = {
            "train_loss": final_train_loss if final_train_loss is not None else "",
            "eval_loss": final_eval_loss if final_eval_loss is not None else "",
            "total_training_time_sec": round(training_duration, 2),
            "avg_epoch_time_sec": round(avg_time, 2),
            "std_epoch_time_sec": round(std_time, 2),
            "eval_accuracy": final_eval_accuracy if final_eval_accuracy is not None else "",
            "eval_f1_macro": final_eval_f1_macro if final_eval_f1_macro is not None else "",
            "eval_mae": final_eval_mae if final_eval_mae is not None else "",
            "eval_precision_macro": final_eval_precision if final_eval_precision is not None else "",
            "eval_recall_macro": final_eval_recall if final_eval_recall is not None else "",
            "epochs_completed": epochs_completed if epochs_completed is not None else "",
            "best_metric": best_metric if best_metric is not None else "",
            **classification_metrics
        }

        write_log(
            f"Параметры обучения: replace={num_layers_to_replace}, add={num_layers_to_add}, "
            f"remove={num_layers_to_remove}, learning_rate={training_args.learning_rate}, "
            f"epochs={training_args.num_train_epochs}"
        )
        write_log(f"Размер train: {len(train_dataset)}, val: {len(eval_dataset)}")
        if hasattr(trainer, 'state') and trainer.state.best_metric is not None:
            write_log(f"Лучшая метрика: {trainer.state.best_metric}")

        trainer.save_model("./custom_bert_finetuned")

        # Собираем все результаты (БЕЗ ДУБЛИРУЮЩЕГО append_train_results)
        seed = training_args_config.get("seed", None)
        run_params = {
            "experiment_id": experiment_id,
            "num_layers_to_replace": num_layers_to_replace,
            "num_layers_to_add": num_layers_to_add,
            "num_layers_to_remove": num_layers_to_remove,
            "train_dataset_size": len(train_dataset),
            "eval_dataset_size": len(eval_dataset),
        }
        if seed is not None:
            run_params["seed"] = seed

        all_results = {
            "run_params": run_params,
            "training_args": training_args_config,
            "bert_config": bert_config_params,
            "final_metrics": final_metrics,
            "epoch_metrics": epoch_metrics_list,
            "memory_stats": {
                "peak_memory_overall": detailed_memory_callback.get_memory_stats().get("peak_memory_overall", 0),
                "avg_epoch_peak": detailed_memory_callback.get_memory_stats().get("avg_epoch_peak", 0),
            },
            "system_info": get_system_info(),
            "positional_params": positional_params_callback.positional_params,
        }

        return bert_loader, model_to_train, all_results


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
    attention_type: str = 'dilated',
    random_state: int = 42,
):
    set_seed(random_state)
    # Базовая директория для логов
    base_log_dir = f"./logs/{config_name}"
    os.makedirs(base_log_dir, exist_ok=True)

    # Уникальная директория для текущего запуска
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    experiment_id = f"{config_name}_{timestamp}"
    run_name = f"run_replace{num_layers_to_replace}_add{num_layers_to_add}_remove{num_layers_to_remove}_{timestamp}"
    run_log_dir = os.path.join(base_log_dir, run_name)
    os.makedirs(run_log_dir, exist_ok=True)

    # Корректировка путей
    log_path = os.path.join(run_log_dir, os.path.basename(log_path))
    output_path = os.path.join(run_log_dir, os.path.basename(output_path))

    # Настройка логирования (один раз!)
    set_log_file(log_path)

    # Загрузка конфигурации из JSON
    if not config_json_path or not config_name:
        raise ValueError("Необходимо указать config_json_path и config_name")

    write_log(f"Загрузка конфигурации '{config_name}' из {config_json_path}")
    config_data = load_config_from_json(config_json_path, config_name)
    training_args_config = config_data["training_args"]
    bert_config_params = config_data["bert_config"]
    class_names = training_args_config.get("class_names")

    # Подробный вывод загруженной конфигурации
    write_log("\n" + "="*60)
    write_log("ЗАГРУЖЕННАЯ КОНФИГУРАЦИЯ:")
    write_log("="*60)

    write_log("\n--- Training Arguments ---")
    for key, value in training_args_config.items():
        write_log(f"  {key}: {value}")

    write_log("\n--- Bert Config ---")
    for key, value in bert_config_params.items():
        write_log(f"  {key}: {value}")

    if class_names:
        write_log(f"\n--- Class Names ---")
        write_log(f"  {class_names}")

    write_log("="*60 + "\n")

    write_log(f"Запуск обучения: train={train_path}, output={output_path}, log={log_path}")
    write_log(f"Директория запуска: {run_log_dir}")
    write_log(f"Параметры слоёв: replace={num_layers_to_replace}, add={num_layers_to_add}, remove={num_layers_to_remove}")
    write_log(f"Используемая конфигурация: {config_name}")
    write_log(f"Режим: {'кросс-валидация' if n_folds > 1 else 'обычное обучение'} с n_folds={n_folds}")

    # Тип внимания
    attention_class = ATTENTION_CLASSES.get(attention_type)
    if attention_class is None:
        raise ValueError(f"Неизвестный тип внимания: {attention_type}. Допустимые: {list(ATTENTION_CLASSES.keys())}")
    write_log(f"Тип внимания: {attention_type} -> {attention_class.__name__}")

    # Проверка наличия подготовленных датасетов
    datasets_dir = "datasets"
    expected_files = [os.path.join(datasets_dir, f) for f in ("train_dataset.csv", "val_dataset.csv", "test_dataset.csv")]
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

    # Создание директории для выходного файла если её нет
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Создание выходного файла
    if not os.path.exists(output_path):
        try:
            with open(output_path, "a", encoding="utf-8"):
                pass
        except Exception as e:
            raise RuntimeError(f"Не удалось создать выходной файл {output_path}: {e}")

    # Определение путей к валидации и тесту
    if "train" in os.path.basename(train_path):
        val_path = train_path.replace("train", "val")
        test_path = train_path.replace("train", "test")
    else:
        val_path = "datasets/val_dataset.csv"
        test_path = "datasets/test_dataset.csv"

    # Конфигурация модели из JSON
    config = BertConfig.from_pretrained(
        bert_config_params["pretrained_model_name"],
        num_hidden_layers=bert_config_params["num_hidden_layers"],
        hidden_size=bert_config_params["hidden_size"],
        num_attention_heads=bert_config_params["num_attention_heads"],
        intermediate_size=bert_config_params["intermediate_size"],
        num_labels=bert_config_params["num_labels"],
        max_position_embeddings=bert_config_params["max_position_embeddings"],
        hidden_dropout_prob=bert_config_params["hidden_dropout_prob"],
        attention_probs_dropout_prob=bert_config_params["attention_probs_dropout_prob"],
        problem_type=bert_config_params["problem_type"]
    )

    base_model = BertForSequenceClassification(config)

    num_params = sum(p.numel() for p in base_model.parameters())
    write_log(f"Размер модели: {num_params/1e6:.1f}M параметров")

    embedding_mean = base_model.bert.embeddings.word_embeddings.weight.mean().item()
    write_log(f"Проверка инициализации: среднее эмбеддингов = {embedding_mean:.6f} (ожидаемо ≈ 0.0)")

    tokenizer = AutoTokenizer.from_pretrained(bert_config_params["pretrained_model_name"])
    tokenizer.model_max_length = bert_config_params["max_position_embeddings"]
    tokenizer.init_kwargs["model_max_length"] = bert_config_params["max_position_embeddings"]

    pos_emb_size = base_model.bert.embeddings.position_embeddings.num_embeddings
    write_log(f"Размер позиционных эмбеддингов: {pos_emb_size}")

    model_container = SimpleModelContainer(model=base_model, tokenizer=tokenizer)

    # Запуск обучения
    bert_loader, model_trained, all_results = train_custom_bert(
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

    # Оценка на тесте (если n_folds == 1)
    test_length_metrics = None
    if n_folds == 1 and test_path and os.path.exists(test_path):
        test_dataset = ReviewDataset(
            csv_path=test_path,
            tokenizer=tokenizer,
            max_length=bert_config_params["max_position_embeddings"]
        )
        device = next(model_trained.parameters()).device
        test_length_metrics = evaluate_by_length_bins(
            model=model_trained,
            dataset=test_dataset,
            tokenizer=tokenizer,
            class_names=class_names,
            device=device,
            max_len=bert_config_params["max_position_embeddings"],
            step=128
        )
        write_log("Оценка на тесте по группам длины завершена.")
        save_all_results_to_csv(
            output_path=output_path,
            run_params=all_results["run_params"],
            training_args_config=training_args_config,
            bert_config_params=bert_config_params,
            final_metrics=all_results["final_metrics"],
            epoch_metrics=all_results.get("epoch_metrics", []),
            memory_stats=all_results["memory_stats"],
            system_info=all_results["system_info"],
            test_length_metrics=test_length_metrics,
            attention_type=attention_type
        )
    
    return bert_loader, model_trained, all_results