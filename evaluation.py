import csv
import os
import time
from collections import defaultdict
from itertools import product

import torch

from dataset_utils import create_data_loader
from logging_utils import write_log


def evaluate_transformer_model(
    transformer_loader,
    input_csv: str,
    output_csv: str,
    devices=("cpu", "cuda"),
    batch_sizes=(1, 2, 4, 8, 16, 32, 64),
    torch_optimizations=(0, 1),
):
    # перебирает комбинации параметров и записывает результаты инференса в CSV
    # model = transformer_loader.model - получение модели
    model = transformer_loader.model
    # tokenizer = transformer_loader.tokenizer - получение токенизатора
    tokenizer = transformer_loader.tokenizer
    # config = transformer_loader.config - получение конфига
    config = transformer_loader.config
    # max_length = config.max_position_embeddings -
    # получение максимальной длины последовательности
    max_length = config.max_position_embeddings


    fieldnames = [
        "line_index",
        "batch_number",
        "device",
        "torch_optimizations",
        "batch_size",
        "max_length",
        "true_label",
        "pred_label",
        "length_label",
        "total_time_sec",
        "time_per_sample_sec",
    ]
# проверка на существование output_csv  
    if not os.path.exists(output_csv):
        with open(output_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="|")
            writer.writeheader()
# создание комбинаций параметров
    param_combinations = list(product(devices, torch_optimizations, batch_sizes))
# перебирает комбинации параметров
    for device_name_tuple, opt_flag, bs in param_combinations:
        # device_name = device_name_tuple - получение имени устройства
        device_name = device_name_tuple
        # device = torch.device(device_name) - создание устройства
        device = torch.device(device_name)

        # bool(opt_flag) - установка benchmark для cuDNN
        # подбор самой эффективеой реализации свертки
        torch.backends.cudnn.benchmark = bool(opt_flag)
        #установка allow_tf32 для cuDNN
        # 10 бит мантиссы для ускорения вычислений
        torch.backends.cuda.matmul.allow_tf32 = bool(opt_flag)
        # установка градиента в False
        # экономит память и время
        torch.set_grad_enabled(False)
        # model.to(device) - перемещение модели на устройство

        model.to(device)
        # model.eval() - установка модели в режим оценки
        model.eval()

        write_log(f"===> Тест: device={device_name}, optim={opt_flag}, batch_size={bs}")
        # создание dataloader
        dataloader = create_data_loader(
            transformer_loader=transformer_loader,
            csv_path=input_csv,
            batch_size=bs,
            max_length=max_length,
        )

        line_index = 0
        all_results = []
        # перебор батчей
        for batch_idx, batch in enumerate(dataloader, start=1):
            batch_start_time = time.time()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["rating_labels"].to(device)
            length_labels = batch["length_labels"].to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                if isinstance(outputs, dict):
                    logits = outputs.get("logits")
                else:
                    logits = getattr(outputs, "logits", None)

                if logits is None:
                    raise RuntimeError("Модель не вернула logits")

                preds = torch.argmax(logits, dim=1)

            batch_time = time.time() - batch_start_time
            batch_size_actual = len(labels)
            time_per_sample = batch_time / batch_size_actual if batch_size_actual > 0 else 0.0

            preds_list = preds.cpu().tolist()
            labels_list = labels.cpu().tolist()
            lengths_list = length_labels.cpu().tolist()

            device_code = 0 if device_name == "cpu" else 1

            for i in range(batch_size_actual):
                line_index += 1
                row = defaultdict(str)
                row["line_index"] = line_index
                row["batch_number"] = batch_idx
                row["device"] = device_code
                row["torch_optimizations"] = opt_flag
                row["batch_size"] = bs
                row["max_length"] = max_length
                row["true_label"] = labels_list[i]
                row["pred_label"] = preds_list[i]
                row["length_label"] = lengths_list[i]
                row["total_time_sec"] = round(batch_time, 4)
                row["time_per_sample_sec"] = round(time_per_sample, 6)
                all_results.append(row)

                if line_index % 100 == 0:
                    write_log(f"Обработано {line_index} строк (индекс батча: {batch_idx})")

        with open(output_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="|")
            writer.writerows(all_results)

        write_log(
            f"Комбинация завершена: device={device_name}, batch_size={bs}, "
            f"optim={opt_flag}, строк записано: {len(all_results)}"
        )

