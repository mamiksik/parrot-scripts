import re
import warnings
from dataclasses import dataclass, field
from math import log
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset, DatasetDict, Dataset
from transformers import RobertaTokenizer, HfArgumentParser

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

relative_root = Path(__file__).parent.resolve() / ".."

accelerator = "cpu"
if torch.cuda.is_available():
    accelerator = "cuda"
elif torch.backends.mps.is_available():
    accelerator = "mps"

device = torch.device(accelerator)


class Config:
    MODEL_CHECKPOINT_BASE_PATH = relative_root / "output-model"
    GEN_EVAL_OUTPUT = relative_root / 'gen_eval_result.csv'
    COM_EVAL_OUTPUT = relative_root / 'com_eval_result.csv'


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


def prepare_dataset(
    training_args: "RunArgs", tokenizer: RobertaTokenizer, preprocess
) -> DatasetDict:
    dataset = load_dataset(
        "mamiksik/processed-commit-diffs", revision=training_args.dataset_rev
    )

    if training_args.debug:
        dataset = DatasetDict(
            {
                "train": Dataset.from_dict(dataset["train"][:2]),
                "test": Dataset.from_dict(dataset["test"][:2]),
                "valid": Dataset.from_dict(dataset["valid"][:2]),
            }
        )

    if not training_args.allow_multi_file_commits:
        dataset = dataset.filter(lambda x: x["file_count"] == 1, keep_in_memory=True)

    if not training_args.allow_mixed_files:
        dataset = dataset.filter(
            lambda x: x["content_type"] != "Mixed", keep_in_memory=True
        )

    tokenized_datasets = dataset.map(
        lambda x: preprocess(training_args, tokenizer, x),
        batched=True,
        remove_columns=["message", "patch", "content_type", "file_count", "main_lang", "sha"],
        num_proc=4,
    )
    tokenized_datasets.set_format("torch")
    return tokenized_datasets


@dataclass
class RunArgs:
    train_bs: int = field(default=8, metadata={"help": "Train batch size"})
    eval_bs: int = field(default=8, metadata={"help": "Eval batch size"})
    acc_grad_steps: int = field(
        default=3, metadata={"help": "Gradient accumulation steps"}
    )
    max_input_size: int = field(default=512, metadata={"help": "Max input size"})
    max_target_size: int = field(default=128, metadata={"help": "Max target size"})

    allow_multi_file_commits: bool = field(
        default=False,
        metadata={"help": "Train model on commits consisting of " "multiple files"},
    )

    allow_mixed_files: bool = field(
        default=False,
        metadata={"help": "Train model on commits consisting of " "mixed file types"},
    )

    dataset_rev: str = field(default=None, metadata={"help": "Dataset revision"})
    debug: bool = field(
        default=False,
        metadata={
            "help": "For running the model locally (e.g. restring dataset to "
            "2 elements)"
        },
    )

    run_name: str = field(
        default=None, metadata={"help": "What makes this run special?"}
    )
    output_to_custom_dir: bool = field(
        default=False,
        metadata={"help": "Output to custom directory? (wanadb run name)"},
    )

    @staticmethod
    def build() -> "RunArgs":
        parser = HfArgumentParser(RunArgs)
        return parser.parse_args_into_dataclasses()[0]


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


def search_sequence_numpy(arr,seq):
    """ Find sequence in an array using NumPy only.

    Parameters
    ----------
    arr    : input 1D array
    seq    : input 1D array

    Output
    ------
    Output : 1D Array of indices in the input array that satisfy the
    matching of input sequence in the input array.
    In case of no match, an empty list is returned.
    """

    # Store sizes of input array and sequence
    Na, Nseq = arr.size, seq.size

    # Range of sequence
    r_seq = np.arange(Nseq)

    # Create a 2D array of sliding indices across the entire length of input array.
    # Match up with the input sequence & get the matching starting indices.
    M = (arr[np.arange(Na-Nseq+1)[:,None] + r_seq] == seq).all(1)

    # Get the range of those indices as final output
    if M.any() >0:
        return np.where(np.convolve(M,np.ones((Nseq),dtype=int))>0)[0]
    else:
        return []         # No match found