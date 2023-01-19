from dataclasses import asdict

import evaluate
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from transformers import (
    RobertaTokenizer,
    T5ForConditionalGeneration,
    AdamW,
    get_linear_schedule_with_warmup,
)
import pytorch_lightning as pl

from utils import prepare_dataset, Config, accelerator, RunArgs


def preprocess(training_args: RunArgs, tokenizer: RobertaTokenizer, examples):
    # encode the code-docstring pairs
    patch = np.array(examples["patch"])
    commit_message = np.array(examples["message"])

    # Some wired stuff with commit message being none
    ii = np.where(commit_message == None)[0]
    commit_message = np.delete(commit_message, ii)
    patch = np.delete(patch, ii)

    inputs = [code for code in patch]
    model_inputs = tokenizer(
        inputs,
        max_length=training_args.max_input_size,
        padding="max_length",
        truncation=True,
    )

    # encode the summaries
    labels = tokenizer(
        list(commit_message),
        max_length=training_args.max_target_size,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    ).input_ids

    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels

    return model_inputs


class CodeT5(pl.LightningModule):
    model: T5ForConditionalGeneration

    def __init__(
        self,
        dataset,
        tokenizer: RobertaTokenizer,
        train_bs,
        eval_bs,
        lr=5e-5,
        num_train_epochs=15,
        warmup_steps=1000,
    ):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(
            "Salesforce/codet5-base-multi-sum"
        )
        self.model.resize_token_embeddings(len(tokenizer))

        self.dataset = dataset
        self.tokenizer = tokenizer

        self.save_hyperparameters(ignore=["dataset", "tokenizer"])

        # Setup metrics
        self.metrics = evaluate.load("bleu")

        self.targets = []
        self.predictions = []

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return outputs

    def training_step(self, batch, batch_idx):
        loss = self(**batch).loss
        self.log("training_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log("validation_loss", outputs.loss, on_epoch=True)

        predictions = self.tokenizer.batch_decode(
            outputs.logits.argmax(-1), skip_special_tokens=True
        )
        targets = batch["labels"].cpu()
        targets = np.where(targets != -100, targets, self.tokenizer.pad_token_id)
        targets = self.tokenizer.batch_decode(targets, skip_special_tokens=True)

        self.targets.extend(targets)
        self.predictions.extend(predictions)
        return outputs.loss

    def validation_epoch_end(self, outputs):
        result = self.metrics.compute(
            predictions=self.predictions, references=self.targets, smooth=True
        )
        self.log("eval/bleu", result["bleu"])

        self.targets.clear()
        self.predictions.clear()

    def test_step(self, batch, batch_idx):
        loss = self(**batch).loss
        return loss

    def configure_optimizers(self):
        # create optimizer
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        # create learning rate scheduler
        num_train_optimization_steps = self.hparams.num_train_epochs * len(
            self.train_dataloader()
        )
        lr_scheduler = {
            "scheduler": get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=num_train_optimization_steps,
            ),
            "name": "learning_rate",
            "interval": "step",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"], shuffle=True, batch_size=self.hparams.train_bs
        )

    def val_dataloader(self):
        return DataLoader(self.dataset["valid"], batch_size=self.hparams.eval_bs)


def main():
    training_args = RunArgs.build()
    model_output_path = Config.MODEL_CHECKPOINT_BASE_PATH / "t5-pl-hub"

    wandb_logger = WandbLogger(
        project="CommitPredictorT5PL", name=training_args.run_name
    )
    wandb_logger.log_hyperparams(asdict(training_args))

    if training_args.output_to_custom_dir:
        model_output_path = (
            Config.MODEL_CHECKPOINT_BASE_PATH / wandb_logger.experiment.id
        )

    print(f"▶️ Run name: {wandb_logger.experiment.name} [{wandb_logger.experiment.id}]")
    print(f"▶️ Output path: {str(model_output_path)}")
    print(f"🚨 Running on {accelerator}")

    print(f"ℹ️ Loading Tokenizer")
    tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base-multi-sum")
    tokenizer.add_tokens(["<ide>", "<add>", "<del>", "<path>"], special_tokens=True)

    print(f"ℹ️ Loading Dataset")
    dataset = prepare_dataset(training_args, tokenizer, preprocess)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    print(f"ℹ️ Loading Model")
    model = CodeT5(
        dataset=dataset,
        tokenizer=tokenizer,
        train_bs=training_args.train_bs,
        eval_bs=training_args.eval_bs,
    )

    print(f"ℹ️ Setting up trainer")
    lr_monitor = LearningRateMonitor(logging_interval="step")
    early_stop_callback = EarlyStopping(
        monitor="validation_loss", patience=3, strict=False, verbose=False, mode="min"
    )

    trainer = Trainer(
        accelerator=accelerator if not training_args.debug else None,
        precision=16,
        default_root_dir=model_output_path / "checkpoints",
        logger=wandb_logger,
        callbacks=[early_stop_callback, lr_monitor],
        accumulate_grad_batches=training_args.acc_grad_steps,
    )

    trainer.fit(model)
    if not training_args.debug:
        model.model.save_pretrained(model_output_path)
        new_model = T5ForConditionalGeneration.from_pretrained(model_output_path)
        new_model.push_to_hub("CodeT5-commit-message-generator", commit_message=f"Experiment ID: {wandb_logger.experiment.id}")
        tokenizer.push_to_hub("CodeT5-commit-message-generator", commit_message=f"Experiment ID: {wandb_logger.experiment.id}")


if __name__ == "__main__":
    main()
