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
        # https://github.com/neulab/code-bert-score/blob/main/run_mlm.py#L448
        # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
        # receives the `special_tokens_mask`.
        return_special_tokens_mask=True,
    )

    masking_mask = []
    for input_ids in inputs["input_ids"]:
        start_msg_index = input_ids.index(tokenizer.sep_token_id) + 3  # </s></s><msg>
        try:
            end_msg_index = input_ids.index(tokenizer.pad_token_id)
        except ValueError:
            end_msg_index = len(input_ids)

        mask = np.zeros(len(input_ids), dtype=bool)
        mask[start_msg_index:end_msg_index] = True
        masking_mask.append(mask)

    inputs["labels"] = inputs["input_ids"].copy()
    inputs["masking_mask"] = masking_mask

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

    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
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


def msg_masking_collator(tokenizer, features: list[list[int] | Any | dict[str, Any]]):
    batch = tokenizer.pad(features, return_tensors="pt", pad_to_multiple_of=None)
    # features = [torch.tensor(e, dtype=torch.long) for e in features]

    inputs = batch["input_ids"].clone()
    labels = batch["input_ids"].clone()
    probability_matrix = torch.full(labels.shape, 0.2)

    special_tokens_mask = batch.pop("special_tokens_mask")
    special_tokens_mask = special_tokens_mask.bool()
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    masking_mask = batch.pop("masking_mask")
    masking_mask = masking_mask.bool()
    probability_matrix.masked_fill_(~masking_mask, value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = (
        torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    )
    inputs[masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    batch["input_ids"] = inputs
    batch["labels"] = labels

    return batch


def main():
    training_args = RunArgs.build()

    model_name = "microsoft/codebert-base-mlm"
    model_output_path = Config.MODEL_CHECKPOINT_BASE_PATH / "mlm"
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

    training_args = TrainingArguments(
        output_dir=str(model_output_path),
        overwrite_output_dir=True,
        hub_model_id="mamiksik/CommitPredictor",
        report_to=["wandb"],
        push_to_hub=True,
        hub_strategy="end",
        load_best_model_at_end=True,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        save_total_limit=50,
        learning_rate=2e-5,
        weight_decay=0.01,
        metric_for_best_model='eval_loss',
        num_train_epochs=50,
        fp16=accelerator == "cuda",
        per_device_train_batch_size=training_args.train_bs,
        per_device_eval_batch_size=training_args.eval_bs,
        gradient_accumulation_steps=training_args.acc_grad_steps,
        remove_unused_columns=False,
    )

    # with AsyncBleu4Callback(eval_dataset=tokenized_dataset["test"], run=wandb.run) as bleu4_callback:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["valid"],
        compute_metrics=lambda eval_pred: compute_metrics(metric, eval_pred),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        data_collator=lambda inputs: msg_masking_collator(tokenizer, inputs),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print(f"🏋️‍♂️  Training")
    trainer.train()

    print(f"🚀  Pushing model to HuggingFace Hub")
    commit_id = trainer.push_to_hub(f"End of training {wandb.run.name}", blocking=True)
    print(f"🎉  Model pushed to HuggingFace Hub: {commit_id}")

    print(f"🏁  Training Done")


if __name__ == "__main__":
    main()
