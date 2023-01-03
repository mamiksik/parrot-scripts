import warnings
from pathlib import Path

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
    dataset = load_dataset("mamiksik/CommitDiffs", use_auth_token=False)
    tokenized_datasets = dataset.map(
        lambda x: preprocess(tokenizer, x),
        batched=True,
        remove_columns=["message", "patch"],
    )
    tokenized_datasets.set_format("torch")
    return tokenized_datasets
