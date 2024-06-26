from typing import Any

import evaluate
import numpy as np
import torch
import wandb

from transformers import (
    RobertaTokenizer,
    TrainingArguments,
    Trainer,
    RobertaForMaskedLM,
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling,
    RobertaTokenizerFast,
)

from utils import (
    Config,
    preprocess_logits_for_metrics,
    prepare_dataset,
    device,
    accelerator, RunArgs,
)

wandb.init(project="CommitPredictor")


def preprocess(training_args: RunArgs, tokenizer: RobertaTokenizer, examples):
    messages = [f"<msg> {message}" for message in examples["message"]]
    inputs = tokenizer(
        examples["patch"],
        messages,
        padding="max_length",
        truncation="only_first",
        max_length=training_args.max_input_size,
        # https://github.com/neulab/code-bert-score/blob/main/run_mlm.py#L448
        # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
        # receives the `special_tokens_mask`.
        return_special_tokens_mask=True,
        return_tensors="np"
    )

    # labels = []
    inputs["labels"] = inputs["input_ids"].copy()
    for idx in range(len(inputs["input_ids"])):
        pad_tokens = np.argwhere(inputs["input_ids"][idx] == tokenizer.sep_token_id)
        start_msg_index = pad_tokens[1][0] + 2
        end_msg_index = pad_tokens[2][0]

        inputs["labels"][idx][:start_msg_index] = -100
        inputs["labels"][idx][end_msg_index:] = -100

        # words_ids = np.asarray(inputs.word_ids(0))
        # inputs["labels"][idx].word_ids()
        inputs["special_tokens_mask"][idx][:start_msg_index] = 1
        inputs["special_tokens_mask"][idx][end_msg_index:] = 1

    return inputs


def init_model_tokenizer(
    model_name: str, tokenizer_name, *, is_acceleration_required=True
) -> (RobertaForMaskedLM, RobertaTokenizer):
    if accelerator == "cpu":
        print("🚨 Acceleration is not available")
        if is_acceleration_required:
            exit(1)

    model = RobertaForMaskedLM.from_pretrained(model_name)
    model.to(device)

    tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_name, use_fast=True)
    tokenizer.add_tokens(["<ide>", "<add>", "<del>", "<msg>", "<path>"], special_tokens=True)
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
        "accuracy": metric["accuracy"].compute(predictions=preds, references=labels)[
            "accuracy"
        ]
    }


# def msg_masking_collator(tokenizer, features: list[list[int] | Any | dict[str, Any]]):
#     batch = tokenizer.pad(features, return_tensors="pt", pad_to_multiple_of=None)
#     # features = [torch.tensor(e, dtype=torch.long) for e in features]
#
#     inputs = batch["input_ids"].clone()
#     labels = batch["labels"].clone()
#     probability_matrix = torch.full(labels.shape, 0.5)
#
#     special_tokens_mask = batch.pop("special_tokens_mask")
#     special_tokens_mask = special_tokens_mask.bool()
#     probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
#     probability_matrix.masked_fill_(labels == -100, value=0.0)
#
#     # masking_mask = batch.pop("masking_mask")
#     # masking_mask = masking_mask.bool()
#     # probability_matrix.masked_fill_(~masking_mask, value=0.0)
#
#     masked_indices = torch.bernoulli(probability_matrix).bool()
#     labels[~masked_indices] = -100  # We only compute loss on masked tokens
#
#     # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
#     indices_replaced = (
#         torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
#     )
#     inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
#
#     # 10% of the time, we replace masked input tokens with random word
#     indices_random = (
#         torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
#         & masked_indices
#         & ~indices_replaced
#     )
#     random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
#     inputs[indices_random] = random_words[indices_random]
#
#     # The rest of the time (10% of the time) we keep the masked input tokens unchanged
#     batch["input_ids"] = inputs
#     batch["labels"] = labels
#
#     return batch


def main():
    training_args = RunArgs.build()

    model_name = "microsoft/codebert-base-mlm"
    model_output_path = Config.MODEL_CHECKPOINT_BASE_PATH / "mlm"
    if training_args.output_to_custom_dir:
        model_output_path = Config.MODEL_CHECKPOINT_BASE_PATH / wandb.run.name

    print(f"▶️  Model name: {model_name}")
    print(f"▶️  Output path: {str(model_output_path)}")

    print(f"ℹ️  Loading Model and Tokenizer")
    model, tokenizer = init_model_tokenizer(
        model_name, model_name, is_acceleration_required=False
    )

    print(f"ℹ️  Loading Metrics")
    metric = {"accuracy": evaluate.load("accuracy")}

    print(f"ℹ️  Loading Dataset")
    tokenized_dataset = prepare_dataset(training_args, tokenizer, preprocess)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.5
    )

    training_args = TrainingArguments(
        output_dir=str(model_output_path),
        overwrite_output_dir=True,
        hub_model_id="mamiksik/CodeBERTa-commit-message-autocomplete",
        report_to=["wandb"],
        push_to_hub=True,
        hub_strategy="end",
        load_best_model_at_end=True,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        save_total_limit=50,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=1000,
        metric_for_best_model='eval_loss',
        num_train_epochs=50,
        fp16=accelerator == "cuda",
        per_device_train_batch_size=training_args.train_bs,
        per_device_eval_batch_size=training_args.eval_bs,
        gradient_accumulation_steps=training_args.acc_grad_steps,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["valid"],
        compute_metrics=lambda eval_pred: compute_metrics(metric, eval_pred),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        # data_collator=lambda inputs: msg_masking_collator(tokenizer, inputs),
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print(f"🏋️‍♂️  Training")
    trainer.train()

    print(f"🚀  Pushing model to HuggingFace Hub")
    tokenizer.push_to_hub("mamiksik/CodeBERTa-commit-message-autocomplete", commit_message=f"End of training {wandb.run.name}")
    commit_id = trainer.push_to_hub(f"End of training {wandb.run.name}", blocking=True)
    print(f"🎉  Model pushed to HuggingFace Hub: {commit_id}")

    print(f"🏁  Training Done")


if __name__ == "__main__":
    main()
