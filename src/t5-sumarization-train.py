import evaluate
import nltk as nltk
import numpy as np
import wandb

from transformers import (
    RobertaTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
)

from utils import Config, hyperparameter_defaults, prepare_dataset, device

wandb.init(config=hyperparameter_defaults, project="CommitPredictorT5")


def preprocess(tokenizer: RobertaTokenizer, examples):
    model_inputs = tokenizer(
        examples["patch"], max_length=412, padding="max_length", truncation=True
    )
    labels = tokenizer(
        text_target=examples["message"],
        max_length=100,
        padding="max_length",
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


def compute_metrics(metric, tokenizer, eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = [
        "\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds
    ]
    decoded_labels = [
        "\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels
    ]

    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    # Add mean generated length
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
    ]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


def load_model_and_tokenizer(model_name: str, tokenizer_name: str):
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.to(device)

    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_tokens(["<keep>", "<add>", "<remove>"], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


def main():
    tokenizer_name = "Salesforce/codet5-base"
    model_name = "Salesforce/codet5-base-multi-sum"

    model_output_path = Config.MODEL_CHECKPOINT_BASE_PATH / wandb.run.name
    print(f"▶️  Model name: {model_name}")
    print(f"▶️  Output path: {str(model_output_path)}")

    print(f"ℹ️  Loading Model and Tokenizer")
    model, tokenizer = load_model_and_tokenizer(model_name, tokenizer_name)

    print(f"ℹ️  Loading Metrics")
    metric = evaluate.load("rouge")

    print(f"ℹ️  Loading Dataset")
    tokenized_dataset = prepare_dataset(tokenizer, preprocess)

    print(f"ℹ️  Initializing Trainer")
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir=str(model_output_path),
        hub_model_id="CommitPredictorT5",
        report_to=["wandb"],
        push_to_hub=True,
        hub_strategy="end",
        overwrite_output_dir=True,
        # load_best_model_at_end=True,
        save_strategy="no",
        evaluation_strategy="epoch",
        save_total_limit=5,
        learning_rate=wandb.config["learning_rate"],
        weight_decay=wandb.config["weight_decay"],
        num_train_epochs=wandb.config["epochs"],
        auto_find_batch_size=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=lambda eval_pred: compute_metrics(metric, tokenizer, eval_pred),
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print(f"🏋️‍♂️  Training")
    trainer.train()
    trainer.save_model(model_output_path)

    print(f"🚀  Pushing model to HuggingFace Hub")
    commit_id = trainer.push_to_hub(f"End of training {wandb.run.name}", blocking=True)
    print(f"🎉  Model pushed to HuggingFace Hub: {commit_id}")

    print(f"🏁  Training Done")


if __name__ == "__main__":
    main()
