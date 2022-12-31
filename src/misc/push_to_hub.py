from src.utils import Config, init_model_tokenizer


def main():
    model, tokenizer = init_model_tokenizer(
        Config.MODEL_CHECKPOINT_BASE_PATH / "code-berta-large-experiment-4",
        tokenizer_name="microsoft/codebert-base-mlm",
        is_cuda_required=False,
    )

    model.push_to_hub("mamiksik/CommitPredictor", private=True)
    tokenizer.push_to_hub("mamiksik/CommitPredictor", private=True)


if __name__ == "__main__":
    main()
