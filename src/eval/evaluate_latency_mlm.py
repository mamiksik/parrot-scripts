import argparse
import timeit

import numpy as np
import torch
from datasets import load_dataset
from transformers import RobertaTokenizerFast, RobertaForMaskedLM, pipeline


def predict(args, model, tokenizer, patch, message):
    with torch.no_grad():
        inputs = tokenizer(
            patch, message, truncation=True, truncation_strategy="only_first", padding=True, return_tensors="pt"
        )

        mask_token_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]

        inputs = inputs.to(args.device)
        logits = model(**inputs).logits
        mask_token_logits = logits[0, mask_token_index, :]
        top_prediction = torch.topk(mask_token_logits, 1, dim=1).indices[:, 0].tolist()[0]


def main(args):
    dataset = load_dataset("mamiksik/processed-commit-diffs")
    patches = []
    messages = []
    for message, patch in zip(dataset['test']['message'], dataset['test']['patch']):
        words = message.split()
        words[np.random.randint(0, len(words))] = '<mask>'
        messages.append(' '.join(words))
        patches.append(patch)

    tokenizer = RobertaTokenizerFast.from_pretrained("mamiksik/CodeBERTa-commit-message-autocomplete")
    model = RobertaForMaskedLM.from_pretrained("mamiksik/CodeBERTa-commit-message-autocomplete").to(args.device)

    def run():
        item = np.random.randint(0, len(patches))
        predict(args, model, tokenizer, patches[item], messages[item])

    t = timeit.Timer(run)
    times = 1000

    device = args.device
    if device == "cuda":
        device = torch.cuda.get_device_name(0)

    print(f"Latency for {device}: {t.timeit(times) / times} sec [{times} loops]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"])

    main(parser.parse_args())
