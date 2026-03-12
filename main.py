from test_sentiment_bert import build_arg_parser
from training import custom_model_train


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    custom_model_train(
        train_path=args.train,
        output_path=args.output,
        log_path=args.log,
        num_layers_to_replace=args.replace,
        num_layers_to_add=args.add,
        num_layers_to_remove=args.remove,
        config_json_path=args.config_file,
        config_name=args.config_name,
        attention_type=args.attention_type,   # ← добавлено
    )


if __name__ == "__main__":
    main()