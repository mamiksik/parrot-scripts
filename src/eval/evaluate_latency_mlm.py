import argparse
import timeit

import numpy as np
import torch
from datasets import load_dataset
from transformers import RobertaTokenizerFast, RobertaForMaskedLM, pipeline


def predict(pipe, message):
    _ = pipe(message)


def main(args):
    dataset = load_dataset("mamiksik/processed-commit-diffs")
    patches = []
    for patch in dataset['test']['patch']:
        words = patch.split()
        words[np.random.randint(0, len(words))] = '<mask>'
        patches.append(' '.join(words))

    tokenizer = RobertaTokenizerFast.from_pretrained("CodeBERTa-commit-message-autocomplete")
    model = RobertaForMaskedLM.from_pretrained("CodeBERTa-commit-message-autocomplete").to(args.device)
    pipe = pipeline("fill-mask", model=model, tokenizer=tokenizer)

    def run():
        item = np.random.randint(0, len(patches))
        predict(pipe, patches[item])

    t = timeit.Timer(run)
    times = 1000000

    device = args.device
    if device == "cuda":
        device = torch.cuda.get_device_name(0)

    print(f"Latency for {device}: {t.timeit(times) / times} sec [{times} loops]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"])

    main(parser.parse_args())
