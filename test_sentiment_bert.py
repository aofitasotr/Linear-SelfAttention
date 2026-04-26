import argparse

from logging_utils import LOG_FILE


def build_arg_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Обучение и оценка кастомной BERT-модели с модифицированным вниманием.",
        add_help=add_help,
    )
    parser.add_argument(
        "--train",
        "-t",
        required=True,
        help=(
            "Путь к исходному CSV. Если указать этот файл и папки 'datasets' нет, "
            "скрипт создаст train/val/test в 'datasets/'."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        default="results_custom.csv",
        help="Путь к выходному CSV с результатами.",
    )
    parser.add_argument(
        "--log",
        "-l",
        default=LOG_FILE,
        help="Путь к файлу логов.",
    )
    parser.add_argument(
        "--replace",
        "--rp",
        "-r",
        type=int,
        default=0,
        help="Количество последних слоёв для замены на кастомные.",
    )
    parser.add_argument(
        "--add",
        "-a",
        type=int,
        default=0,
        help="Количество новых кастомных слоёв для добавления.",
    )
    parser.add_argument(
        "--remove",
        "--rm",
        "-R",
        type=int,
        default=0,
        help="Количество верхних слоёв BERT encoder для удаления.",
    )
    parser.add_argument(
        "--config-file",
        "--cf",
        required=True,
        help="Путь к JSON файлу с конфигурациями.",
    )
    parser.add_argument(
        "--config-name",
        "--cn",
        required=True,
        help="Имя конфигурации в JSON файле.",
    )
    parser.add_argument(
        "--attention-type",
        "--at",
        type=str,
        default="dilated",
        choices=["base", "pos-enc", "dilated", "local-window", "weighted"],
        help="Тип линейного внимания: base, pos-enc, dilated, local-window или weighted.",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=1,
        help="Количество фолдов для кросс-валидации. Значение 1 означает обычное обучение без CV.",
    )
    return parser


def main(argv=None):
    from training import custom_model_train

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    custom_model_train(
        train_path=args.train,
        output_path=args.output,
        log_path=args.log,
        num_layers_to_replace=args.replace,
        num_layers_to_add=args.add,
        num_layers_to_remove=args.remove,
        config_json_path=args.config_file,
        config_name=args.config_name,
        n_folds=args.n_folds,
        attention_type=args.attention_type,
    )


if __name__ == "__main__":
    main()
