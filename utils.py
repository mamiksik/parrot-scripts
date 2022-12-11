from pathlib import Path

import torch
from transformers import RobertaForMaskedLM, RobertaTokenizer

relative_root = Path(__file__).parent.resolve()


class Config:
    MODEL_CHECKPOINT_BASE_PATH = relative_root / 'output-model'


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
