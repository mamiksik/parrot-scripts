import re
from copy import copy
from math import log
from pprint import pprint

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm

# from typing import Self

from transformers import RobertaTokenizer, RobertaForMaskedLM, pipeline


class PredictionCandidate:
    def __init__(self, mask=None):
        self.mask = [] if mask is None else mask

    def copy_with(self, score: float, token: str) -> "PredictionCandidate":
        return type(self)(mask=self.mask.copy() + [(score, token)])

    def score(self) -> int:
        return sum(map(lambda x: log(x[0]), self.mask))

    def masked(self):
        return str(self) + "".join(["<mask>" for _ in range(20 - len(self.mask))])

    def __str__(self) -> str:
        return " ".join(map(lambda x: x[1], self.mask))


def predict(tokenizer, model, text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    # outputs = model(**inputs)
    # predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    logits = model(**inputs).logits
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[0]

    mask_token_logits = logits[0, mask_token_index, :]
    top_3_tokens = torch.topk(mask_token_logits, 3, dim=1).indices[0].tolist()

    # values, predictions = predictions.topk(5)
    # for i, (_values, _predictions) in enumerate(zip(values.tolist(), predictions.tolist())):
    #     for v, p in zip(_values, _predictions):
    #         print(tokenizer.decode(p))

    return outputs


def main():
    path = "/home/dron/work/temp/BscModel/output-model/mlm/checkpoint-2260-best-bleu"
    path = "/home/dron/work/temp/BscModel/output-model/mlm/checkpoint-452"
    path = "/home/dron/work/temp/BscModel/output-model/mlm/checkpoint-3164"
    path = "/home/dron/work/temp/BscModel/output-model/mlm/checkpoint-1356"
    path = "mamiksik/CommitPredictor"

    model = RobertaForMaskedLM.from_pretrained(path, revision="c653c13")
    # path = '/home/dron/work/temp/BscModel/output-model/mlm/'
    tokenizer = RobertaTokenizer.from_pretrained(
        path, revision="c653c13", truncation=True
    )

    # Create a torch.device object for the GPU
    # gpu_index = torch.cuda.current_device()
    # device = torch.device("cuda:" + str(gpu_index))
    device = "cpu"
    # device = torch.device("mps")
    pipe = pipeline("fill-mask", model=model, tokenizer=tokenizer, device=device)

    dataset = load_dataset("mamiksik/CommitDiffs", use_auth_token=False)
    df = pd.DataFrame.from_dict(dataset["train"])
    print(df)

    message = (
        "<keep>def main():" '<keep>   print("Hello World!")' '<remove>   name = "John"'
    )
    length = 20

    text = message + f" </s></s> <msg>" + " <mask>" * length

    n_beams = 7
    best_n_results = 4
    len_multipliers = [1.5, 2, 3, 4]
    for i, (_, message, patch) in tqdm(list(df.iloc[::47].iterrows())):
        length = len(message.split())
        commits = []
        print(f'\nPatch: \n{"=" * 70}\n{patch[:1000]}\n{"=" * 70}\n')
        for len_multiplier in len_multipliers:
            commits += predict_commit(
                pipe, patch, int(length * len_multiplier), n_beams
            )[:best_n_results]
        # person_message = input()
        print(f"\nCommit: {message}")
        pprint(commits)
        print()


def predict_commit(pipe, message, length, n_beams=7):
    beams_prob = [0.0, 0.0]
    text = (
        message + f" </s></s> <msg>" + " <mask>" * length
    )  # so that last dim still predicts
    beams = [
        text,
    ] * 2
    for i in range(length):
        beams = beams[:n_beams]
        beams_prob = beams_prob[:n_beams]
        try:
            r = pipe(list(beams))  # (inputs, #mask, order){score:, sequence:, ...}
        except RuntimeError as e:
            print(e)
            break
        # handle last <mask>
        last_mask = True if isinstance(r[0][0], dict) else False
        # get sequences and their probs
        new_beams = []
        new_beams_prob = []
        for beam_prob, input_result in zip(beams_prob, r):
            for prediction in (
                input_result if last_mask else input_result[0]
            ):  # only 1st mask
                if (
                    last_mask
                ):  # Avoid overcompensating branches where last token is obvisous (period etc.)
                    new_beams.append(prediction["sequence"])
                    new_beams_prob.append(beam_prob)
                    break  # add only the most probable last word
                else:
                    new_beams_prob.append(beam_prob + log(prediction["score"]))
                    new_beams.append(prediction["sequence"][4:-4])

        beams = np.array(new_beams)[np.argsort(new_beams_prob)][::-1]
        beams_prob = np.sort(new_beams_prob)[::-1]  # TODO: check correct sorting
        # TODO: try moving cutoff to the front of the function, before pipeline

        # commits = [re.findall(r"<msg> ?(.+)(<mask>)?", b)[0] for b in beams]
        # pprint(dict(zip(beams_prob, commits)))
    commits = [re.findall(r"<msg> ?(.*?)(<mask>|$)", b)[0][0] for b in beams]
    return commits


if __name__ == "__main__":
    main()
