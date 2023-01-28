from transformers import AutoTokenizer, T5ForConditionalGeneration


def main():
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained("mamiksik/T5-commit-message-generator")
    model: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained("mamiksik/T5-commit-message-generator")

    model.push_to_hub("mamiksik/T5-commit-message-generation", commit_message="Experiment ID: 3a050tsi", use_auth_token=True)
    tokenizer.push_to_hub("mamiksik/T5-commit-message-generation", commit_message="Experiment ID: 3a050tsi")

if __name__ == "__main__":
    main()