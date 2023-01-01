import evaluate
import torch

import wandb

from transformers import (
    RobertaTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    RobertaForCausalLM,
)

from utils import (
    Config,
    preprocess_logits_for_metrics,
    hyperparameter_defaults,
    prepare_dataset,
    device,
)

wandb.init(config=hyperparameter_defaults, project="CodeBertaCLM")


def preprocess(tokenizer, examples):
    messages = [f"<msg>{message}" for message in examples["message"]]
    inputs = tokenizer(
        examples["patch"], messages, padding="max_length", truncation="only_first"
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

    model = RobertaForCausalLM.from_pretrained(model_name)
    model.to(device)

    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_tokens(["<keep>", "<add>", "<remove>", "<msg>"], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


def compute_metrics(metric, eval_pred):
    preds, labels = eval_pred

    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics
    labels = labels.reshape(-1)
    preds = preds.reshape(-1)
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]
    return {
        'accuracy': metric['accuracy'].compute(predictions=preds, references=labels),
        'f1': metric['f1'].compute(predictions=preds, references=labels)['f1'],
        'bleu4': metric['bleu4'].compute(predictions=preds, references=labels, smooth=True)["bleu"],
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
        'accuracy': evaluate.load("accuracy"),
        'f1': evaluate.load("f1"),
        'bleu4': evaluate.load("bleu4"),
    }

    print(f"ℹ️  Loading Dataset")
    tokenized_dataset = prepare_dataset(tokenizer, preprocess)

    print(f"ℹ️  Initializing Trainer")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=0.20
    )

    training_args = TrainingArguments(
        output_dir=str(model_output_path),
        overwrite_output_dir=True,
        hub_model_id="mamiksik/CodeBertaCLM",
        report_to=["wandb"],
        push_to_hub=True,
        hub_strategy="end",
        load_best_model_at_end=False,
        save_strategy="no",
        evaluation_strategy="epoch",
        save_total_limit=50,
        learning_rate=wandb.config["learning_rate"],
        weight_decay=wandb.config["weight_decay"],
        # num_train_epochs=wandb.config["epochs"],
        num_train_epochs=5,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=lambda eval_pred: compute_metrics(metric, eval_pred),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print(f"🏋️‍♂️  Training")
    trainer.train()

    print(f"🚀  Pushing model to HuggingFace Hub")
    commit_id = trainer.push_to_hub(f"End of training (f1/bleu4) {wandb.run.name}", blocking=True)
    print(f"🎉  Model pushed to HuggingFace Hub: {commit_id}")

    print(f"🏁  Training Done")


if __name__ == "__main__":
    main()
