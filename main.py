from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True, slots=True)
class CommandSpec:
    """Описание одной CLI-команды верхнего уровня."""

    name: str
    help_text: str
    loader: Callable[[], Callable[[list[str] | None], None]]


def _load_reviews_entrypoint() -> Callable[[list[str] | None], None]:
    from test_sentiment_bert import main as reviews_main

    return reviews_main


def _load_synthetic_positional_entrypoint() -> Callable[[list[str] | None], None]:
    from synthetic.train_positional import main as synthetic_main

    return synthetic_main


def _load_synthetic_consecutive_ones_entrypoint() -> Callable[[list[str] | None], None]:
    from synthetic.train_consecutive_ones import main as consecutive_ones_main

    return consecutive_ones_main


COMMANDS: tuple[CommandSpec, ...] = (
    CommandSpec(
        name="reviews",
        help_text="Обучение и оценка на текстовом датасете отзывов.",
        loader=_load_reviews_entrypoint,
    ),
    CommandSpec(
        name="synthetic",
        help_text="Синтетическая задача positional lookup.",
        loader=_load_synthetic_positional_entrypoint,
    ),
    CommandSpec(
        name="synthetic-consecutive-ones",
        help_text="Синтетическая задача maximum consecutive ones.",
        loader=_load_synthetic_consecutive_ones_entrypoint,
    ),
)
COMMAND_INDEX = {command.name: command for command in COMMANDS}


def build_main_parser() -> argparse.ArgumentParser:
    """Строит верхнеуровневый CLI с подкомандами проекта."""
    parser = argparse.ArgumentParser(
        description="Единая точка входа для текстовых и синтетических экспериментов.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")
    for command in COMMANDS:
        subparsers.add_parser(command.name, help=command.help_text)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Диспетчеризует выполнение в соответствующий модуль по имени команды."""
    argv = sys.argv[1:] if argv is None else argv
    parser = build_main_parser()

    if not argv or argv == ["-h"] or argv == ["--help"]:
        parser.print_help()
        return

    first_argument = argv[0]
    if first_argument.startswith("-"):
        _load_reviews_entrypoint()(argv)
        return

    command = COMMAND_INDEX.get(first_argument)
    if command is None:
        parser.print_help()
        return

    command.loader()(argv[1:])


if __name__ == "__main__":
    main()
