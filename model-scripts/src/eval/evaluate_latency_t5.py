import argparse
import timeit

import numpy as np
import torch
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, RobertaTokenizerFast


def predict(model, tokenizer, message, args):
    with torch.no_grad():
        input_ids = tokenizer(
            message,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(args.device).input_ids

        outputs = model.generate(
            input_ids,
            max_length=120,
            min_length=5,
            num_beams=7,
            num_return_sequences=5,
        )


def main(args):
    dataset = load_dataset("mamiksik/processed-commit-diffs")
    tokenizer = RobertaTokenizerFast.from_pretrained("mamiksik/CommitPredictorT5PL", revision="fb08d01")
    model = T5ForConditionalGeneration.from_pretrained("mamiksik/CommitPredictorT5PL", revision="fb08d01").to(args.device)

    def run():
        item = np.random.randint(0, len(dataset['test']))
        predict(model, tokenizer, dataset['test']['patch'][item], args)

    print("Running...")
    t = timeit.Timer(run)
    times = 1_000

    device = args.device
    if device == "cuda":
        device = torch.cuda.get_device_name(0)

    print(f"Latency for {device}: {t.timeit(times) / times} sec [{times} loops]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"])

    main(parser.parse_args())