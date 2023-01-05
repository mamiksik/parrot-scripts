import re
import warnings
from math import log
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset, DatasetDict
from transformers import RobertaTokenizer

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

relative_root = Path(__file__).parent.resolve() / ".."

hyperparameter_defaults = dict(
    learning_rate=2e-5,
    weight_decay=0.01,
    epochs=10,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Config:
    MODEL_CHECKPOINT_BASE_PATH = relative_root / "output-model"


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


def prepare_dataset(tokenizer: RobertaTokenizer, preprocess) -> DatasetDict:
    dataset = load_dataset("mamiksik/CommitDiffs", use_auth_token=True)
    tokenized_datasets = dataset.map(
        lambda x: preprocess(tokenizer, x),
        batched=True,
        remove_columns=["message", "patch"],
    )
    tokenized_datasets.set_format("torch")
    return tokenized_datasets


def predict_commit(pipe, message, length, n_beams=7):
    beams_prob = [0.0, 0.0]
    text = message + f' </s></s> <msg>' + ' <mask>' * length  # so that last dim still predicts
    beams = [text, ] * 2
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
            for prediction in input_result if last_mask else input_result[0]:  # only 1st mask
                if last_mask:   # Avoid overcompensating branches where last token is obvisous (period etc.)
                    new_beams.append(prediction['sequence'])
                    new_beams_prob.append(beam_prob)
                    break   # add only the most probable last word
                else:
                    new_beams_prob.append(beam_prob + log(prediction['score']))
                    new_beams.append(prediction['sequence'][4:-4])

        beams = np.array(new_beams)[np.argsort(new_beams_prob)][::-1]
        beams_prob = np.sort(new_beams_prob)[::-1]  # TODO: check correct sorting
        # TODO: try moving cutoff to the front of the function, before pipeline

        # commits = [re.findall(r"<msg> ?(.+)(<mask>)?", b)[0] for b in beams]
        # pprint(dict(zip(beams_prob, commits)))
    commits = [re.findall(r"<msg> ?(.*?)(<mask>|$)", b)[0][0] for b in beams]
    return commits
