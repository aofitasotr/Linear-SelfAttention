import argparse

from logging_utils import LOG_FILE
from training import custom_model_train


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Обучение и оценка кастомной BERT-модели с модифицированным вниманием."
    )
    parser.add_argument(
        "--train",
        "-t",
        required=True,
        help=(
            "Путь к исходному CSV (например, полный набор отзывов). Если указать этот файл "
            "и папки 'datasets' нет, скрипт создаст train/val/test в 'datasets/'."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        default="results_custom.csv",
        help="Путь к выходному CSV с результатами (будет создан, если отсутствует)",
    )
    parser.add_argument(
        "--log",
        "-l",
        default=LOG_FILE,
        help="Путь к файлу логов (будет создан, если отсутствует)",
    )
    parser.add_argument(
        "--replace",
        "--rp",
        "-r",
        type=int,
        default=0,
        help="Количество последних слоёв для замены на кастомные (алиасы: --rp, -r)",
    )
    parser.add_argument(
        "--add",
        "-a",
        type=int,
        default=0,
        help="Количество новых кастомных слоёв для добавления",
    )
    parser.add_argument(
        "--remove",
        "--rm",
        "-R",
        type=int,
        default=0,
        help="Количество верхних слоёв BERT encoder для удаления (алиасы: --rm, -R)",
    )
    parser.add_argument(
        "--config-file",
        "--cf",
        required=True,
        help="Путь к JSON файлу с конфигурациями (обязательно)"
    )
    parser.add_argument(
        "--config-name",
        "--cn",
        required=True,
        help="Имя конфигурации в JSON файле (обязательно)"
    )
    # Новый аргумент для выбора типа внимания
    parser.add_argument(
        "--attention-type",
        "--at",
        type=str,
        default="dilated",
        choices=["base", "pos-enc", "dilated"],
        help="Тип линейного внимания: base (без позиций), pos-enc (с синусоидами), dilated (дилатированное)"
    )
    return parser