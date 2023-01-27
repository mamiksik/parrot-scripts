import logging

import pandas as pd
from datasets import load_dataset

from b_annotate_metadata import RepositoryMetadata
from utils import Config

logger = logging.getLogger("final-dataset")


def main(repositories: list[tuple[str, str]]):
    # df = pd.DataFrame(columns=)
    for (owner, repo) in repositories:
        all_commits = RepositoryMetadata.parse_file(
            Config.METADATA_PATH / f"{owner}-{repo}.json"
        ).commits_to_df()
        filtered_commits = pd.read_csv(
            Config.ANNOTATED_METADATA_PATH / f"{owner}-{repo}.csv"
        )["sha"]

        all_commits[""]

        df.append(repo_metadata.commits_to_df())
