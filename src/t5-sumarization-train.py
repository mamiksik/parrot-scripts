import evaluate
import numpy as np

import wandb

from transformers import (
    RobertaTokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq, EarlyStoppingCallback, Seq2SeqTrainingArguments, Seq2SeqTrainer,
)

from utils import Config, hyperparameter_defaults, prepare_dataset, device, preprocess_logits_for_metrics, preprocess_t5

wandb.init(config=hyperparameter_defaults, project="CommitPredictorT5")


def compute_metrics(metrics, tokenizer, eval_pred):
    predictions, labels = eval_pred
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    return metrics['bleu4'].compute(predictions=decoded_preds, references=decoded_labels, smooth=True)


def load_model_and_tokenizer(model_name: str, tokenizer_name: str):
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.to(device)

    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_tokens(["<ide>", "<add>", "<del>"], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def main():
    tokenizer_name = "Salesforce/codet5-base-multi-sum"
    model_name = "Salesforce/codet5-base-multi-sum"

    model_output_path = Config.MODEL_CHECKPOINT_BASE_PATH / 't5-hf'
    print(f"▶️  Model name: {model_name}")
    print(f"▶️  Output path: {str(model_output_path)}")

    print(f"ℹ️  Loading Model and Tokenizer")
    model, tokenizer = load_model_and_tokenizer(model_name, tokenizer_name)

    print(f"ℹ️  Loading Metrics")
    metrics = {
        'bleu4': evaluate.load("bleu")
    }

    print(f"ℹ️  Loading Dataset")
    tokenized_dataset = prepare_dataset(tokenizer, preprocess_t5)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

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

        learning_rate=2e-5,
        weight_decay=0.01,

        per_device_train_batch_size=21,
        per_device_eval_batch_size=21,
        gradient_accumulation_steps=3,

        bf16=True,
        num_train_epochs=100,
        metric_for_best_model="eval_loss",
        predict_with_generate=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["valid"],
        compute_metrics=lambda eval_pred: compute_metrics(metrics, tokenizer, eval_pred),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        data_collator=data_collator,
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
