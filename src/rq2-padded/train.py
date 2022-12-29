import evaluate
import wandb
import warnings

from transformers import RobertaTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, \
    EarlyStoppingCallback
from datasets import load_dataset, DatasetDict
from src.utils import Config, init_model_tokenizer, compute_metrics, preprocess_logits_for_metrics

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def prepare_dataset(tokenizer: RobertaTokenizer) -> DatasetDict:
    dataset = load_dataset("mamiksik/CommitDiffs", use_auth_token=True)

    def preprocess(examples):
        messages = [f"<msg> {message}" for message in examples["message"]]
        inputs = tokenizer(examples["patch"], messages, padding="max_length", truncation='only_first')
        inputs['labels'] = inputs['input_ids'].copy()
        return inputs

    tokenized_datasets = dataset.map(preprocess, batched=True, remove_columns=["message", "patch"])
    tokenized_datasets.set_format("torch")

    return tokenized_datasets


def main():
    model_name = "microsoft/codebert-base-mlm"
    model_output_path = Config.MODEL_CHECKPOINT_BASE_PATH / wandb.run.name
    print(f'▶️  Model name: {model_name}')
    print(f'▶️  Output path: {str(model_output_path)}')

    print(f'ℹ️  Loading Model and Tokenizer')
    model, tokenizer = init_model_tokenizer(model_name, is_cuda_required=False)

    print(f'ℹ️  Loading Metrics')
    metric = evaluate.load("accuracy")

    print(f'ℹ️  Loading Dataset')
    tokenized_dataset = prepare_dataset(tokenizer)

    print(f'ℹ️  Initializing Trainer')
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.20)
    training_args = TrainingArguments(
        output_dir=str(model_output_path),
        hub_model_id="mamiksik/CommitPredictor",
        report_to=["wandb"],
        push_to_hub=True,
        overwrite_output_dir=True,
        load_best_model_at_end=True,

        save_strategy="epoch",
        evaluation_strategy="epoch",
        save_total_limit=5,

        learning_rate=wandb.config["learning_rate"],
        weight_decay=wandb.config["weight_decay"],
        num_train_epochs=wandb.config["epochs"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        compute_metrics=lambda eval_pred: compute_metrics(metric, eval_pred),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    print(f'🏋️‍♂️  Training')
    trainer.train()
    trainer.save_model(model_output_path)

    print(f'Pushing model to HuggingFace Hub')
    trainer.push_to_hub()
    print(f'🏁  Training Done')


if __name__ == "__main__":
    main()
