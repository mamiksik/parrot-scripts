import gc

import pandas as pd
from datasets import load_dataset

import evaluate
from tqdm import tqdm

import torch
from transformers import RobertaTokenizer, T5ForConditionalGeneration

from utils import Config

SUPPORTED_LANGUAGES = ["Java", "Go", "PHP", "Javascript", "Ruby", "Python"]
DEVICE = torch.device("cuda")


class BleuT5:
    def __init__(self, tokenizer_name, model_name, *, model_revision='main'):
        self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
        self._model = T5ForConditionalGeneration.from_pretrained(model_name, revision=model_revision)

    def preprocess_batch(self, batch):
        return [patch for patch in batch['patch']]
        # return [f"Summarize {lang}:" + patch for patch, lang in zip(batch['patch'], batch['language'])]

    def predict_batch(self, batch):
        batch = self.preprocess_batch(batch)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                input = self.tokenizer(batch, truncation=True, padding=True, return_tensors='pt').to(DEVICE)
                outputs = self.model.generate(input.input_ids, max_length=100, min_length=10, num_beams=7)
                del input

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def predict_one(self, lang, patch):
        with torch.no_grad():
            input_ids = self.tokenizer(f"{patch}", truncation=True, padding=True, return_tensors='pt').to(DEVICE).input_ids
            outputs = self.model.generate(input_ids, max_length=100, min_length=10, num_beams=7)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def __enter__(self):
        self.model = self._model.to(DEVICE)

    def __exit__(self, type, value, traceback):
        self._model = self._model.to('cpu')
        del self.model
        with torch.no_grad():
            torch.cuda.empty_cache()
        gc.collect()


def evaluate_batch(metric, batch, model):
    labels = batch['message']
    predictions = model.predict_batch(batch)
    return metric.compute(predictions=predictions, references=labels, smooth=True)['bleu']


def evaluate_model(dataset, metric, model, *, for_lang=None, multi_file=False):
    batch_size = 64
    test_ds = dataset['test']
    if for_lang is not None:
        test_ds = test_ds.filter(lambda x: x["main_lang"] == for_lang, keep_in_memory=True)

    if multi_file:
        test_ds = test_ds.filter(lambda x: x["file_count"] == 1, keep_in_memory=True)

    to_process = len(test_ds)
    trials = []
    for i in tqdm(range(0, to_process, batch_size)):
        trials.append(evaluate_batch(metric, test_ds[i:i+batch_size], model))

    avg = sum(trials) / len(trials)
    return round(avg * 100, 2)


def main():
    metric = evaluate.load("bleu")
    dataset = load_dataset("mamiksik/processed-commit-diffs")

    models = {
        "T5 Pytorch": BleuT5('mamiksik/CommitPredictorT5', 'mamiksik/CommitPredictorT5PL', model_revision="5dc6ce5"),
        "T5 Multifile": BleuT5('mamiksik/CommitPredictorT5', 'mamiksik/CommitPredictorT5PL', model_revision="fb08d01"),
        "T5 Base Multi Sum": BleuT5('Salesforce/codet5-base-multi-sum', 'Salesforce/codet5-base-multi-sum')
    }

    results = []
    for input_type in ["Multi File", "Single File"]:
        is_multifile = input_type == "Multi File"
        for model_name, model in models.items():
            print(f"===== {input_type}-{model_name} =====")
            result = dict()
            result['Model'] = model_name
            result['Input'] = input_type

            with model:
                result['Overall'] = evaluate_model(dataset, metric, model, multi_file=is_multifile)
                print(f"Overall: {result['Overall']}")

            for main_lang in SUPPORTED_LANGUAGES:
                with model:
                    result[main_lang] = evaluate_model(dataset, metric, model, for_lang=main_lang, multi_file=is_multifile)
                    print(f"For {main_lang}: {result[main_lang]}")

            results.append(result)

            print(f"===== END {model_name} =====")

    df = pd.DataFrame.from_records(results, columns=["Model", "Overall"] + SUPPORTED_LANGUAGES)
    df.to_csv(Config.GEN_EVAL_OUTPUT, index=False)


if __name__ == '__main__':
    main()
