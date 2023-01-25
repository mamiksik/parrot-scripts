import gc
import itertools
from abc import ABC
import random

import numpy as np
import pandas as pd
from datasets import load_dataset

import evaluate
from tqdm import tqdm

import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM, pipeline

from src.utils import Config

SUPPORTED_LANGUAGES = ["Java", "Go", "PHP", "Javascript", "Ruby", "Python"]

accelerator = "cpu"
if torch.cuda.is_available():
    accelerator = "cuda"

device = torch.device(accelerator)
metric = evaluate.load("accuracy")


def process(dataset, compatibility_mode, for_lang):
    def map_entry(entry):
        message = entry['message'].split()

        message = f"<msg> {' '.join(message)}"
        patch = entry["patch"]
        for special_token in ["<msg>", "<pad>", "<s>", "</s>", "<mask>", "<unk>"]:
            patch = patch.replace(special_token, '')

        if compatibility_mode:
            patch = patch.replace("<del>", '-')
            patch = patch.replace("<add>", '+')
            patch = patch.replace("<ide>", '')

        return {'patch': patch, 'message': message, 'main_lang': entry['main_lang']}

    if for_lang is not None:
        dataset = dataset.filter(lambda x: x["main_lang"] == for_lang, keep_in_memory=True)

    return dataset.map(map_entry, batched=False, keep_in_memory=True)


def calculate_accuracy(model, tokenizer, dataset) -> float:
    predictions = []
    references = []
    for patch, message in tqdm(zip(dataset['patch'], dataset['message'])):
        inputs = tokenizer(
            patch, message, truncation=True, truncation_strategy="only_first", padding=True, return_tensors="pt"
        )

        sep_tokens = torch.argwhere(inputs["input_ids"] == tokenizer.sep_token_id)
        start_msg_index = sep_tokens[1][1] + 2
        end_msg_index = sep_tokens[2][1]
        mask_token_index = torch.randint(start_msg_index, end_msg_index, (1,))[0].item()
        references.append(inputs["input_ids"][0][mask_token_index].item())

        inputs["input_ids"][0][mask_token_index] = tokenizer.mask_token_id
        with torch.no_grad():
            mask_token_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]

            inputs = inputs.to(device)
            logits = model(**inputs).logits
            mask_token_logits = logits[0, mask_token_index, :]
            top_prediction = torch.topk(mask_token_logits, 1, dim=1).indices[:, 0].tolist()[0]

        predictions.append(top_prediction)

    return metric.compute(predictions=predictions, references=references)['accuracy']


def evaluate(dataset, model_url, model_name, compatibility_mode) -> dict:
    tokenizer = RobertaTokenizer.from_pretrained(model_url)
    model = RobertaForMaskedLM.from_pretrained(model_url).to(device)
    result = {"Model": model_name}

    data = process(dataset, compatibility_mode=compatibility_mode, for_lang=None)
    result['Overall'] = calculate_accuracy(model, tokenizer, data)

    for main_lang in SUPPORTED_LANGUAGES:
        data = process(dataset, compatibility_mode=compatibility_mode, for_lang=main_lang)
        result[main_lang] = calculate_accuracy(model, tokenizer, data)
        print(f"For {main_lang}: {result[main_lang]}")

    return result


def main():
    dataset = load_dataset("mamiksik/processed-commit-diffs")
    dataset = dataset['test']

    models = [
        ("RoBERTa", "roberta-base", True),
        ("CodeBERT", "microsoft/codebert-base-mlm", True),
        ("Parrot", "mamiksik/CodeBERTa-commit-message-autocomplete", False)
    ]

    acc = []
    for model_name, model_url, compatibility_mode in models:
        acc.append(evaluate(dataset, model_url, model_name, compatibility_mode))

    df = pd.DataFrame.from_records(acc, columns=["Model", "Overall"] + SUPPORTED_LANGUAGES)
    df.to_csv(Config.COM_EVAL_OUTPUT, index=False)

    # # for model, tokenizer in models:
    # # compatibility_mode = True
    # # tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")
    # # model = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm").to(device)
    #
    # compatibility_mode = False
    # tokenizer = RobertaTokenizer.from_pretrained("mamiksik/CodeBERTa-commit-message-autocomplete")
    # model = RobertaForMaskedLM.from_pretrained("mamiksik/CodeBERTa-commit-message-autocomplete").to(device)
    #
    # pipe = pipeline("fill-mask", model=model, tokenizer=tokenizer, device=device)
    #
    # patches = []
    # messages = []
    # references = []
    # for example in tqdm(dataset):
    #     message = example['message'].split()
    #     # mask_index = np.random.randint(0, len(message))
    #     # references.append(message[mask_index])
    #     # message[mask_index] = tokenizer.mask_token
    #
    #     message = f"<msg> {' '.join(message)}"
    #     patch = example["patch"]
    #     for special_token in ["<msg>", "<pad>", "<s>", "</s>", "<mask>", "<unk>"]:
    #         patch = patch.replace(special_token, '')
    #
    #     if compatibility_mode:
    #         patch = patch.replace("<del>", '-')
    #         patch = patch.replace("<add>", '+')
    #         patch = patch.replace("<ide>", '')
    #
    #     patches.append(patch)
    #     messages.append(message)
    #
    # predictions = []
    # for i in tqdm(range(0, 200)):
    #     pt_batch = tokenizer(
    #         patches[i], messages[i], truncation=True, truncation_strategy="only_first", padding=True, return_tensors="pt"
    #     )
    #
    #     sep_tokens = torch.argwhere(pt_batch["input_ids"] == tokenizer.sep_token_id)
    #     start_msg_index = sep_tokens[1][1] + 2
    #     end_msg_index = sep_tokens[2][1]
    #     random_index = torch.randint(start_msg_index, end_msg_index, (1,))[0].item()
    #     references.append(pt_batch["input_ids"][0][random_index].item())
    #     pt_batch["input_ids"][0][random_index] = tokenizer.mask_token_id
    #
    #     with torch.no_grad():
    #         mask_token_index = torch.where(pt_batch['input_ids'] == tokenizer.mask_token_id)[1]
    #         logits = model(**pt_batch).logits
    #         mask_token_logits = logits[0, mask_token_index, :]
    #         top_prediction = torch.topk(mask_token_logits, 1, dim=1).indices[:, 0].tolist()[0]
    #
    #     predictions.append(top_prediction)
    #
    # accuracy = metric.compute(predictions=predictions, references=references)['accuracy']
    # print("Overall accuracy:", accuracy)


if __name__ == '__main__':
    main()
