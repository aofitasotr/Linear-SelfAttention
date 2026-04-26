import argparse

from logging_utils import write_log
from synthetic.model import SYNTHETIC_ATTENTION_TYPES
from training.consecutive_ones_pipeline import run_consecutive_ones_attention_sweep, train_consecutive_ones_model


def build_arg_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Обучение на синтетической задаче consecutive ones",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=add_help,
    )
    parser.add_argument("--mode", type=str, default="single", choices=["single", "all_attentions"])
    parser.add_argument("--attention_type", type=str, default="dilated", choices=SYNTHETIC_ATTENTION_TYPES)
    parser.add_argument("--use_original_model", action="store_true")
    parser.add_argument("--context_len", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout_prob", type=float, default=0.4)
    parser.add_argument("--warmup_ratio", type=float, default=0.25)
    parser.add_argument("--train_samples", type=int, default=50000)
    parser.add_argument("--eval_samples", type=int, default=5000)
    parser.add_argument("--test_samples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./synthetic_consecutive_ones_output")
    parser.add_argument("--log_dir", type=str, default="./synthetic_consecutive_ones_logs")
    parser.add_argument("--results_csv", type=str, default="./synthetic_consecutive_ones_results.csv")
    parser.add_argument("--early_stop_metric_threshold", type=float, default=1.0)
    parser.add_argument("--early_stopping_patience", type=int, default=0)
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.mode == "single":
        results = train_consecutive_ones_model(
            attention_type=args.attention_type,
            use_original_model=args.use_original_model,
            context_len=args.context_len,
            hidden_size=args.hidden_size,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout_prob=args.dropout_prob,
            warmup_ratio=args.warmup_ratio,
            train_samples=args.train_samples,
            eval_samples=args.eval_samples,
            test_samples=args.test_samples,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
            output_dir=args.output_dir,
            log_dir=args.log_dir,
            results_csv_path=args.results_csv,
            early_stop_metric_threshold=args.early_stop_metric_threshold,
            early_stopping_patience=args.early_stopping_patience,
        )
        write_log("\n" + "=" * 70)
        write_log("CONSECUTIVE ONES RESULTS")
        write_log("=" * 70)
        write_log(f"Model: {'original' if args.use_original_model else args.attention_type}")
        write_log(f"Accuracy: {results['test_accuracy']:.4f}")
        write_log(f"F1-macro: {results['test_f1_macro']:.4f}")
        write_log(f"Time per sample: {results['test_time_per_sample_ms']:.2f} ms")
        write_log("=" * 70)
        return

    df = run_consecutive_ones_attention_sweep(
        attention_types=SYNTHETIC_ATTENTION_TYPES,
        use_original_model=args.use_original_model,
        context_len=args.context_len,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout_prob=args.dropout_prob,
        base_seed=args.seed,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        results_csv_path=args.results_csv,
        early_stop_metric_threshold=args.early_stop_metric_threshold,
        early_stopping_patience=args.early_stopping_patience,
    )
    write_log("\n" + "=" * 70)
    write_log("CONSECUTIVE ONES RESULTS")
    write_log("=" * 70)
    write_log(df[["attention_type", "test_accuracy", "test_f1_macro"]].to_string())
    write_log("=" * 70)


if __name__ == "__main__":
    main()
