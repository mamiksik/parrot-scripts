import logging
import os

import pandas as pd
from datasets import load_dataset, DatasetDict, Dataset
from tqdm import tqdm

from d_preprocess_diffs import Commit
from utils import Config

logger = logging.getLogger("final-dataset")

recognized_extensions = {
    ".java": "Java",
    ".py": "Python",
    ".js": "Javascript",
    ".php": "PHP",
    ".rb": "Ruby",
    ".go": "Go",
}


def push_analysed_metadata(repos):
    dfs = []
    for owner, repo in tqdm(repos):
        logger.info(f"Reading analysed metada for {owner}/{repo}")

        df = pd.read_csv(Config.ANNOTATED_METADATA_PATH / f"{owner}-{repo}.csv")
        df["owner"] = owner
        df["repo"] = repo
        dfs.append(df)

    df = pd.concat(dfs)
    df["subject_length"] = pd.to_numeric(df["subject_length"])

    dataset = Dataset.from_pandas(df)

    dataset_id = "analysed-diff-metadata"
    logger.info(f"💾 Dataset published to hub at {dataset_id}")
    dataset.push_to_hub(dataset_id)


def load_raw_diffs(items, owner, repo, allowed_shas):
    logger.info(f"Reading diffs for {owner}/{repo}")
    diff_root = Config.RAW_DIFFS_PATH / f"{owner}-{repo}"

    # Note that local_filename == commit sha + .json
    for local_filename in tqdm(os.listdir(diff_root)):
        if not os.path.isfile(diff_root / local_filename):
            continue

        # Check that commit is in prefiltered dataset (in case we updated constrains)
        if local_filename.removesuffix(".json") not in allowed_shas:
            continue

        commit: Commit = Commit.parse_file(diff_root / local_filename)
        is_multipart = len(commit.files) > 1
        for file in commit.files:
            extension = file.filename.rsplit(".", 1)[-1]
            language = recognized_extensions.get(extension, "Other")

            items.append(
                {
                    "language": language,
                    "owner": owner,
                    "repo": repo,
                    "sha": local_filename,
                    "message": commit.message,
                    "path": file.filename,
                    "patch": file.patch,
                    "is_multipart": is_multipart,
                }
            )


def push_raw_diffs_to_hub(repositories):
    # df = pd.DataFrame(columns=["language", "owner", "repo", "sha", "message", "path", "patch", "is_multipart"])
    items = []
    for owner, repo in repositories:
        df_prefiltered = pd.read_csv(
            Config.ANNOTATED_METADATA_PATH / f"{owner}-{repo}.csv"
        )
        df_prefiltered = df_prefiltered[df_prefiltered["fits_requirements"] == True]
        load_raw_diffs(items, owner, repo, set(df_prefiltered["sha"]))

    dataset = Dataset.from_list(items)

    dataset_id = "raw-commit-diffs"
    logger.info(f"💾 Dataset published to hub at {dataset_id}")
    dataset.push_to_hub(dataset_id)


def push_final_dataset_to_hub():
    dataset = load_dataset("csv", data_dir=Config.PREPROCESSED_DIFFS_PATH)
    dataset = dataset.shuffle()
    dataset = dataset["train"]

    # https://discuss.huggingface.co/t/how-to-split-main-dataset-into-train-dev-test-as-datasetdict/1090/2
    # 90% train, 10% test + validation
    train_test_dataset = dataset.train_test_split(test_size=0.2)
    # Split the 10% test + valid in half test, half valid
    test_valid = train_test_dataset["test"].train_test_split(test_size=0.5)
    # gather everyone if you want to have a single DatasetDict

    dataset = DatasetDict(
        {
            "train": train_test_dataset["train"],
            "test": test_valid["test"],
            "valid": test_valid["train"],
        }
    )

    dataset_id = "processed-commit-diffs"
    logger.info(f"💾 Dataset published to hub at {dataset_id}")
    dataset.push_to_hub(dataset_id)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    push_final_dataset_to_hub()
