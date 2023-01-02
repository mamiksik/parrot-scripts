import evaluate
import nltk as nltk
import numpy as np
from datasets import DatasetDict, Dataset

import wandb

from transformers import (
    RobertaTokenizer,
    TrainingArguments,
    Trainer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq, EarlyStoppingCallback, Seq2SeqTrainingArguments,
)

from utils import Config, hyperparameter_defaults, prepare_dataset, device, preprocess_logits_for_metrics

wandb.init(config=hyperparameter_defaults, project="CommitPredictorT5")


def preprocess(tokenizer: RobertaTokenizer, examples):

    messages = [f"Generate commit message:\n{patch}" for patch in examples["patch"]]
    model_inputs = tokenizer(
        messages, max_length=412, padding="max_length", truncation=True
    )

    labels = tokenizer(
        text_target=examples["message"],
        max_length=100,
        padding="max_length",
        truncation=True,
    ).input_ids

    labels_with_ignore_index = []
    for labels_example in labels:
        labels_example = [label if label != 0 else -100 for label in labels_example]
        labels_with_ignore_index.append(labels_example)

    model_inputs["labels"] = labels_with_ignore_index

    return model_inputs


def compute_metrics(metrics, tokenizer, eval_pred):
    predictions, labels = eval_pred
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = [
        "\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds
    ]
    decoded_labels = [
        "\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels
    ]

    result = metrics['rouge'].compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )

    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    output = {k: round(v, 4) for k, v in result.items()}
    output['bleu'] = metrics['bleu4'].compute(predictions=decoded_preds, references=decoded_labels, smooth=True)["bleu"]
    return output


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
    metrics = {
        'rouge': evaluate.load("rouge"),
        'bleu4': evaluate.load("bleu")
    }

    print(f"ℹ️  Loading Dataset")
    tokenized_dataset = prepare_dataset(tokenizer, preprocess)

    # tokenized_dataset = DatasetDict({
    #     "train": Dataset.from_dict(tokenized_dataset["train"][:2]),
    #     "test": Dataset.from_dict(tokenized_dataset["test"][:2]),
    # })

    print(f"ℹ️  Initializing Trainer")
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(model_output_path),
        hub_model_id="CommitPredictorT5",
        report_to=["wandb"],
        push_to_hub=True,
        hub_strategy="end",
        overwrite_output_dir=True,
        load_best_model_at_end=True,

        save_strategy="epoch",
        evaluation_strategy="epoch",
        save_total_limit=50,

        learning_rate=wandb.config["learning_rate"],
        weight_decay=wandb.config["weight_decay"],

        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,

        num_train_epochs=30,
        bf16=True,
        metric_for_best_model="eval_bleu",

        predict_with_generate=True,
        auto_find_batch_size=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=lambda eval_pred: compute_metrics(metrics, tokenizer, eval_pred),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=7)],
    )

    print(f"🏋️‍♂️  Training")
    trainer.train()

    print(f"🚀  Pushing model to HuggingFace Hub")
    commit_id = trainer.push_to_hub(f"End of training (patience=7, prefix) {wandb.run.name}", blocking=True)
    print(f"🎉  Model pushed to HuggingFace Hub: {commit_id}")

    print(f"🏁  Training Done")


if __name__ == "__main__":
    main()
