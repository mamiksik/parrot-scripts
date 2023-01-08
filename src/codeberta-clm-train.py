import evaluate
import torch
from datasets import DatasetDict, Dataset

import wandb

from transformers import (
    RobertaTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    RobertaForCausalLM,
    EarlyStoppingCallback,
    RobertaConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from utils import (
    Config,
    preprocess_logits_for_metrics,
    hyperparameter_defaults,
    prepare_dataset,
    device,
)

wandb.init(config=hyperparameter_defaults, project="CodeBertaCLM")


def preprocess(tokenizer: RobertaTokenizer, examples):
    messages = [f"<msg> {message}" for message in examples["message"]]
    inputs = tokenizer(
        examples["patch"] + messages, padding="max_length", truncation=True
    )
    inputs["labels"] = inputs["input_ids"].copy()
    return inputs


def init_model_tokenizer(
    model_name: str, tokenizer_name, *, is_cuda_required=True
) -> (RobertaForCausalLM, RobertaTokenizer):
    if not torch.cuda.is_available():
        print("🚨 CUDA is not available")
        if is_cuda_required:
            exit(1)

    config = RobertaConfig.from_pretrained(model_name)
    config.is_decoder = True
    model = RobertaForCausalLM.from_pretrained(model_name, config=config)
    model.to(device)

    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_tokens(["<keep>", "<add>", "<remove>", "<msg>"], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


def compute_metrics(tokenizer, metric, eval_pred):
    preds, labels = eval_pred

    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics
    labels = labels.reshape(-1)
    preds = preds.reshape(-1)
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    return {
        "accuracy": metric["accuracy"].compute(predictions=preds, references=labels)[
            "accuracy"
        ],
        "f1": metric["f1"].compute(
            predictions=preds, references=labels, average="micro"
        )["f1"],
        "bleu4": metric["bleu4"].compute(
            predictions=decoded_preds, references=decoded_labels, smooth=True
        )["bleu"],
    }


def main():
    model_name = "microsoft/codebert-base"
    model_output_path = Config.MODEL_CHECKPOINT_BASE_PATH / wandb.run.name
    print(f"▶️  Model name: {model_name}")
    print(f"▶️  Output path: {str(model_output_path)}")

    print(f"ℹ️  Loading Model and Tokenizer")
    model, tokenizer = init_model_tokenizer(
        model_name, model_name, is_cuda_required=False
    )

    print(f"ℹ️  Loading Metrics")
    metric = {
        "accuracy": evaluate.load("accuracy"),
        "f1": evaluate.load("f1"),
        "bleu4": evaluate.load("bleu"),
    }

    print(f"ℹ️  Loading Dataset")
    tokenized_dataset = prepare_dataset(tokenizer, preprocess)
    # tokenized_dataset = DatasetDict({
    #     "train": Dataset.from_dict(tokenized_dataset["train"][:5]),
    #     "test": Dataset.from_dict(tokenized_dataset["test"][:5]),
    # })

    print(f"ℹ️  Initializing Trainer")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(model_output_path),
        overwrite_output_dir=True,
        hub_model_id="mamiksik/CodeBertaCLM",
        report_to=["wandb"],
        push_to_hub=True,
        hub_strategy="end",
        load_best_model_at_end=True,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        save_total_limit=100,
        learning_rate=wandb.config["learning_rate"],
        weight_decay=wandb.config["weight_decay"],
        num_train_epochs=200,
        metric_for_best_model="eval_bleu4",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=lambda eval_pred: compute_metrics(tokenizer, metric, eval_pred),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=25)],
    )

    print(f"🏋️‍♂️  Training")
    trainer.train()

    print(f"🚀  Pushing model to HuggingFace Hub")
    commit_id = trainer.push_to_hub(
        f"End of training (Seq2SeqTraining) {wandb.run.name}", blocking=True
    )
    print(f"🎉  Model pushed to HuggingFace Hub: {commit_id}")

    print(f"🏁  Training Done")


if __name__ == "__main__":
    main()
