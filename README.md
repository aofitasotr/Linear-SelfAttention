# Linear Attention BERT for Sentiment Analysis

Исследовательский проект по замене стандартного `self-attention` в BERT на семейство линейных аппроксимаций с акцентом на:

- позиционную модуляцию значений;
- разрежённые dilated-схемы;
- локальные окна внимания;
- сравнение с классическим BERT на реальных и синтетических задачах.

Проект оформлен как единый CLI с отдельными сценариями для:

- обучения на датасете отзывов;
- синтетической задачи `positional lookup`;
- синтетической задачи `consecutive ones`.

## Ключевые возможности

- Единая точка входа через `main.py`.
- Конфигурирование экспериментов через `config.json`.
- Частичная замена attention-слоёв BERT на кастомные реализации.
- Поддержка кросс-валидации по фолдам.
- Отдельные synthetic-бенчмарки для анализа индуктивных bias.
- Логирование метрик, времени и параметров запуска.

## Структура проекта

```text
.
├── custom_attention.py          # кастомные блоки внимания и интеграция в BERT
├── main.py                      # верхнеуровневый CLI
├── config.json                  # конфигурации текстовых экспериментов
├── dataset_utils.py             # подготовка train/val/test и датасет отзывов
├── evaluation.py                # отдельные утилиты оценки и инференса
├── logging_utils.py             # логирование и сохранение результатов
├── synthetic/
│   ├── dataset.py               # positional lookup dataset
│   ├── consecutive_ones_dataset.py
│   ├── model.py                 # synthetic-модели и фабрики
│   ├── train_positional.py
│   └── train_consecutive_ones.py
└── training/
    ├── callbacks.py             # Trainer callbacks и custom trainer
    ├── config.py                # загрузка и типизация конфигов
    ├── cv.py                    # cross-validation
    ├── eval.py                  # оценка на length bins и сохранение метрик
    ├── model_factory.py         # фабрики моделей и attention-реестры
    ├── pipeline.py              # основной train/eval pipeline для отзывов
    ├── synthetic_pipeline.py
    ├── consecutive_ones_pipeline.py
    └── schemas.py               # dataclass-схемы конфигов и артефактов
```

## Установка

Минимальные зависимости:

- Python 3.11+
- PyTorch
- Transformers
- scikit-learn
- pandas

Пример:

```bash
pip install torch transformers scikit-learn pandas
```

## Быстрый старт

### 1. Подготовьте данные и конфиг

Для текстового сценария нужен CSV-файл с отзывами. Если в проекте ещё нет папки `datasets/` с готовыми `train/val/test`, код сам попытается собрать их из исходного CSV.

Пример:

```text
datasets/amazon2023_225.csv
```

Конфигурация эксперимента задаётся в `config.json`, например `amazon_5class`.

### 2. Запустите обучение

Классический BERT без замены attention:

```bash
python main.py reviews \
  --train datasets/amazon2023_225.csv \
  --output results_original.csv \
  --log train_original.log \
  --replace 0 \
  --add 0 \
  --remove 0 \
  --config-file config.json \
  --config-name amazon_5class
```

Линейное внимание с позиционной модуляцией:

```bash
python main.py reviews \
  --train datasets/amazon2023_225.csv \
  --output results_posenc.csv \
  --log train_posenc.log \
  --replace 4 \
  --add 0 \
  --remove 0 \
  --config-file config.json \
  --config-name amazon_5class \
  --attention-type pos-enc
```

Кросс-валидация на 3 фолдах:

```bash
python main.py reviews \
  --train datasets/amazon2023_225.csv \
  --output results_cv3_local_window.csv \
  --log train_cv3_local_window.log \
  --replace 4 \
  --add 0 \
  --remove 0 \
  --config-file config.json \
  --config-name amazon_5class \
  --attention-type local-window \
  --n-folds 3
```

### 3. Проверьте результаты

После запуска:

- подробный лог будет лежать в `logs/...`;
- итоговые метрики сохраняются в CSV, указанный через `--output`;
- checkpoint-ы сохраняются в директории конкретного запуска.

## Синтетические задачи

### Positional Lookup

Синтетическая задача на позиционную адресацию относительно фиксированного маркера.

```bash
python main.py synthetic \
  --mode single \
  --attention_type pos-enc \
  --vocab_size 10 \
  --k 4 \
  --hidden_size 96 \
  --num_heads 4 \
  --num_layers 2
```

### Consecutive Ones

Синтетическая задача на максимальную длину подряд идущих единиц в бинарной последовательности.

```bash
python main.py synthetic-consecutive-ones \
  --mode single \
  --attention_type local-window \
  --context_len 64 \
  --hidden_size 96 \
  --num_heads 4 \
  --num_layers 2
```

## Реализованные типы внимания

- `base` — базовая линейная агрегация контекста без позиционного учёта;
- `pos-enc` — линейное внимание с позиционной модуляцией значений;
- `dilated` — разрежённое внимание по подрешёткам с разным шагом;
- `local-window` — локальные окна внимания с несколькими масштабами;
- `weighted` — взвешенная модификация линейного внимания.

Важно:

- `--attention-type base` — это не классический BERT;
- классический BERT запускается через `--replace 0 --add 0 --remove 0`.

## Архитектурные принципы

- Верхнеуровневый CLI отделён от train-pipeline.
- Конфиги экспериментов типизированы через dataclass-схемы.
- Фабрики моделей и attention-реестр собраны в `training/model_factory.py`.
- Synthetic-модели используют единый типизированный конфиг в `synthetic/model.py`.

## Что смотреть в коде

Если нужен быстрый вход по основным модулям:

- `custom_attention.py` — все реализации внимания;
- `training/model_factory.py` — создание моделей и выбор типа внимания;
- `training/pipeline.py` — основной сценарий обучения на отзывах;
- `synthetic/model.py` — синтетические модели и pooling-логика;
- `training/schemas.py` — dataclass-схемы конфигов.

## Разработка

В репозитории добавлены базовые настройки качества кода через `pyproject.toml`.

Проверить синтаксис и smoke-тесты:

```bash
python -m py_compile main.py custom_attention.py
python -m pytest
```

Запустить линтер и форматирование:

```bash
ruff check .
black .
```
