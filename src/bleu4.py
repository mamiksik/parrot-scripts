from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import evaluate
import wandb
from datasets import Dataset
from transformers import pipeline, RobertaTokenizer, RobertaForMaskedLM, Pipeline, DefaultFlowCallback, \
    TrainingArguments, TrainerState, TrainerControl
from wandb.apis.public import Run

from src.utils import predict_commit


@dataclass
class CheckpointDescription:
    output_dir: Path
    checkpoint_name: str
    eval_dataset: Dataset
    run: Run


class AsyncBleu4Callback(DefaultFlowCallback):
    pool: Pool
    run: Run
    eval_dataset: Dataset

    def __init__(self, *, eval_dataset: Dataset, run: Run):

        self.run = run
        self.eval_dataset = eval_dataset

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        checkpoint = CheckpointDescription(
            output_dir=Path(args.output_dir),
            checkpoint_name=state.trial_name,
            run=self.run,
            eval_dataset=self.eval_dataset
        )
        self.pool.apply_async(evaluate_worker, (checkpoint, ))

    def __enter__(self):
        self.pool = Pool(processes=2)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pool.join()
        self.pool.close()


def evaluate_worker(checkpoint: CheckpointDescription):
    model = RobertaForMaskedLM.from_pretrained(checkpoint.output_dir / checkpoint.checkpoint_name)
    tokenizer = RobertaTokenizer.from_pretrained(checkpoint.output_dir)
    metric = evaluate.load("bleu")

    pipe = pipeline("fill-mask", model=model, tokenizer=tokenizer)

    ground_truth = []
    predictions = []
    for eval_row in checkpoint.eval_dataset:
        message = eval_row["message"]
        patch = eval_row["patch"]

        ground_truth.append(message)
        predictions.append(predict_commit(pipe, message, patch)[0])

    bleu = metric.compute(predictions=predictions, references=ground_truth, smooth=True)["bleu"]

    checkpoint.run.log({"bleu4": bleu},)





