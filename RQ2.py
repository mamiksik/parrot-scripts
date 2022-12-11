import numpy as np

from transformers import RobertaConfig, RobertaTokenizer, RobertaForMaskedLM, pipeline, TrainingArguments, Trainer, DataCollatorForLanguageModeling, TFAutoModelForMaskedLM
from datasets import load_dataset, DatasetDict

import evaluate
import torch

from utils import Config

def init_model_tokenizer(model_name: str) -> (RobertaForMaskedLM, RobertaTokenizer):
    if not torch.cuda.is_available():
        print("🚨 CUDA is not avaible")
        exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RobertaForMaskedLM.from_pretrained(model_name)
    model.to(device)

    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    tokenizer.add_tokens(["<keep>", "<add>", "<remove>", "<msg>"], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer

def prepare_dataset(tokenizer: RobertaTokenizer) -> DatasetDict:
    prepath = Config.PREPROCESSED_DIFFS_PATH
    dataset = load_dataset("csv", data_files=[
        str(prepath / "facebook-react.csv"),
        str(prepath / "huggingface-transformers.csv"),
        str(prepath / "huggingface-datasets.csv"),
        str(prepath / "atom-atom.csv"),
        str(prepath / "mrdoob-three.js.csv"),
        str(prepath / "nicolargo-glances.csv"),
        str(prepath / "spring-projects-spring-framework.csv"),
        str(prepath / "vercel-next.js.csv"),
        str(prepath / "chartjs-Chart.js.csv"),
    ])

    dataset = dataset.remove_columns("Unnamed: 0")
    dataset = dataset['train']
    dataset = dataset.train_test_split(test_size=0.3)

    def preprocess(examples):
        outputs = []
        for patch, message in zip(examples["patch"], examples["message"]):
            fragment = f"<msg> {message}\n{patch}"
            outputs.append(fragment)

        inputs = tokenizer(outputs, padding="max_length", truncation=True)
        inputs['labels'] = inputs['input_ids'].copy()
        return inputs

    tokenized_datasets = dataset.map(preprocess, batched=True, remove_columns=["message", "patch"])
    tokenized_datasets.set_format("torch")

    return tokenized_datasets


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


def main():
    model_name = "microsoft/codebert-base-mlm" #"huggingface/CodeBERTa-small-v1" #
    model_output_path = Config.MODEL_CHECKPOIN_BASE_PATH / 'code-berta-large-experiment-3'
    print(f'▶️  Model name: {model_name}')
    print(f'▶️  Output path: {str(model_output_path)}')

    print(f'ℹ️  Loading Model and Tokenizer')
    model, tokenizer = init_model_tokenizer(model_name)

    print(f'ℹ️  Loading Metrics')
    metric = evaluate.load("accuracy")

    print(f'ℹ️  Loading Dataset')
    tokenized_dataset = prepare_dataset(tokenizer)

    print(f'ℹ️  Initializing Trainer')
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    training_args = TrainingArguments(
        output_dir=str(model_output_path),
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        # per_device_train_batch_size=1,
        # per_device_eval_batch_size=1,
        # gradient_accumulation_steps=2,
        # eval_accumulation_steps=2,
        save_strategy="epoch",
        save_total_limit=15, # Only last 5 models are saved. Older ones are deleted.
        load_best_model_at_end=True,
        num_train_epochs=5
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        compute_metrics=lambda eval_pred: compute_metrics(metric, eval_pred),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        data_collator=data_collator,
    )

    print(f'🏋️‍♂️  Training')
    trainer.train()
    trainer.save_model(model_output_path)
    print(f'🏁  Training Done')


def predict():
    device = torch.device("cpu")
    model_name = Config.MODEL_CHECKPOIN_BASE_PATH / 'code-berta-large-experiment-2'

    local_model = RobertaForMaskedLM.from_pretrained(model_name)
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")

    tokenizer.add_tokens(["<keep>", "<add>", "<remove>", "<msg>"], special_tokens=True)
    local_model.resize_token_embeddings(len(tokenizer))
    local_model.to(device)

    # org_msg: remove unused variable
    test = """<msg> <mask> unused variable
    <keep> import {hydrateRoot} from 'react-dom';
    <keep> import App from './App';
    <remove> const root = hydrateRoot(document, <App assets={window.assetManifest} />);
    <add> hydrateRoot(document, <App assets={window.assetManifest} />);
    """

    local_fill_mask = pipeline('fill-mask', model=local_model, tokenizer=tokenizer)
    for prediction in local_fill_mask(test):
        print(f"'{prediction['token_str']}' with core: {prediction['score']}")

if __name__ == "__main__":
    # main()
    predict()