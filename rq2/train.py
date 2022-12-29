import evaluate
import torch
import wandb
import warnings

from transformers import RobertaTokenizer, RobertaForMaskedLM, pipeline, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling
from datasets import load_dataset, DatasetDict
from utils import Config, init_model_tokenizer, compute_metrics, preprocess_logits_for_metrics

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def prepare_dataset(tokenizer: RobertaTokenizer) -> DatasetDict:
    dataset = load_dataset("mamiksik/CommitDiffs", use_auth_token=True)

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


def main():
    model_name = "microsoft/codebert-base-mlm"  # "huggingface/CodeBERTa-small-v1" #
    model_output_path = Config.MODEL_CHECKPOINT_BASE_PATH / 'code-berta-large-experiment-3'
    print(f'▶️  Model name: {model_name}')
    print(f'▶️  Output path: {str(model_output_path)}')

    print(f'ℹ️  Loading Model and Tokenizer')
    model, tokenizer = init_model_tokenizer(model_name, is_cuda_required=False)

    print(f'ℹ️  Loading Metrics')
    metric = evaluate.load("accuracy")

    print(f'ℹ️  Loading Dataset')
    tokenized_dataset = prepare_dataset(tokenizer)

    print(f'ℹ️  Initialize wandb')
    wandb.init(project="CommitPredictor")

    print(f'ℹ️  Initializing Trainer')
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.20)
    training_args = TrainingArguments(
        output_dir=str(model_output_path),
        overwrite_output_dir=True,
        push_to_hub=True,
        hub_model_id="mamiksik/CommitPredictor",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=15,  # Only last 5 models are saved. Older ones are deleted.
        load_best_model_at_end=True,
        num_train_epochs=3,
        report_to=["wandb"]
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
    # trainer.save_model(model_output_path)

    print(f'Pushing model to HuggingFace Hub')
    trainer.push_to_hub()
    print(f'🏁  Training Done')


def predict():
    device = torch.device("cpu")
    model_name = Config.MODEL_CHECKPOINT_BASE_PATH / 'code-berta-large-experiment-2'

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
    main()
