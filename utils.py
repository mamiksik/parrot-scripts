from pathlib import Path

import torch
from transformers import RobertaForMaskedLM, RobertaTokenizer

relative_root = Path(__file__).parent.resolve()


class Config:
    MODEL_CHECKPOINT_BASE_PATH = relative_root / 'output-model'


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(metric, eval_pred):
    preds, labels = eval_pred

    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics
    labels = labels.reshape(-1)
    preds = preds.reshape(-1)
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]

    return metric.compute(predictions=preds, references=labels)

def init_model_tokenizer(model_name: str, *, tokenizer_name=None, is_cuda_required=True) -> (RobertaForMaskedLM, RobertaTokenizer):
    if tokenizer_name is None:
        tokenizer_name = model_name

    if not torch.cuda.is_available():
        print("🚨 CUDA is not avaible")
        if is_cuda_required:
            exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RobertaForMaskedLM.from_pretrained(model_name)
    model.to(device)

    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_tokens(["<keep>", "<add>", "<remove>", "<msg>"], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer
