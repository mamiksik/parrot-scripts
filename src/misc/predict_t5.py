from datasets import load_dataset
from transformers import T5ForConditionalGeneration, RobertaTokenizer


def main():
    model_path = '../../output-model/t5-pl'
    tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-small")
    tokenizer.add_tokens(["<ide>", "<add>", "<del>"], special_tokens=True)

    model = T5ForConditionalGeneration.from_pretrained(model_path)
    dataset = load_dataset("mamiksik/CommitDiffs", use_auth_token=True)

    test_example = dataset["test"][70]

    input_ids = tokenizer(f"Summarize {test_example['language']}:" + test_example['patch'], return_tensors='pt').input_ids
    outputs = model.generate(input_ids)
    print("Input:", test_example['patch'])
    print("Commit message:", test_example['message'])
    print("Generated docstring:", tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
