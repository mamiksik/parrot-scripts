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

import wandb
from utils import hyperparameter_defaults, prepare_dataset, Config, accelerator

# wandb.init(config=hyperparameter_defaults, project="CommitPredictorT5PL")

max_input_length = 256
max_target_length = 128


def preprocess(tokenizer: RobertaTokenizer, examples):
    # encode the code-docstring pairs
    language = examples["language"]
    patch = np.array(examples["patch"])
    commit_message = np.array(examples["message"])

    ii = np.where(commit_message == None)[0]
    commit_message = np.delete(commit_message, ii)
    patch = np.delete(patch, ii)

    inputs = [f"Summarize {lang}: " + code for lang, code in zip(language, patch)]
    model_inputs = tokenizer(
        inputs, max_length=max_input_length, padding="max_length", truncation=True
    )

    # encode the summaries
    labels = tokenizer(
        list(commit_message),
        max_length=max_target_length,
        padding="max_length",
        truncation=True,
    ).input_ids

    # important: we need to replace the index of the padding tokens by -100
    # such that they are not taken into account by the CrossEntropyLoss
    labels_with_ignore_index = []
    for labels_example in labels:
        labels_example = [label if label != 0 else -100 for label in labels_example]
        labels_with_ignore_index.append(labels_example)

    model_inputs["labels"] = labels_with_ignore_index

    return model_inputs


class CodeT5(pl.LightningModule):
    def __init__(self, *, dataset, tokenizer: RobertaTokenizer, lr=5e-5, num_train_epochs=15, warmup_steps=1000):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(
            "Salesforce/codet5-small"
        )
        self.model.resize_token_embeddings(len(tokenizer))

        self.dataset = dataset
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return outputs

    def common_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)

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
        return DataLoader(self.dataset["train"], shuffle=True, batch_size=8)

    def val_dataloader(self):
        return DataLoader(self.dataset["valid"], batch_size=4)

    def test_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=4)


def main():
    model_output_path = Config.MODEL_CHECKPOINT_BASE_PATH / 't5-pl'
    wandb_logger = WandbLogger(project="CommitPredictorT5PL")
    print(f"🚨 Running on {accelerator}")

    tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-small")
    tokenizer.add_tokens(["<ide>", "<add>", "<del>"], special_tokens=True)
    dataset = prepare_dataset(tokenizer, preprocess)

    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    model = CodeT5(dataset=dataset, tokenizer=tokenizer)

    early_stop_callback = EarlyStopping(
        monitor="validation_loss", patience=3, strict=False, verbose=False, mode="min"
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = Trainer(
        accelerator=accelerator,
        precision=16,
        default_root_dir=model_output_path / "checkpoints",
        logger=wandb_logger,
        callbacks=[early_stop_callback, lr_monitor],
    )

    trainer.fit(model)
    model.model.save_pretrained(model_output_path)


if __name__ == "__main__":
    main()
