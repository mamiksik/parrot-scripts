import logging
import os
import time
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import pandas as pd
import requests
import spacy
import stanza
from requests import Response
from tqdm import trange, tqdm
from dotenv import load_dotenv
from transformers import HfArgumentParser

load_dotenv()


logger = logging.getLogger("utils")


class Config:
    DATASET_ROOT = Path(os.getenv("OUTPUT_ROOT"))
    METADATA_PATH = DATASET_ROOT / "a-metadata"
    ANNOTATED_METADATA_PATH = DATASET_ROOT / "b-prefiltered-metadata"
    RAW_DIFFS_PATH = DATASET_ROOT / "c-raw-diffs"
    PREPROCESSED_DIFFS_PATH = DATASET_ROOT / "d-preprocessed-diffs"
    FINAL_DATASET_PATH = DATASET_ROOT / "e-final-dataset"

    TOP_REPOS_PATH = DATASET_ROOT / '..' / 'top_repos'
    REPOSITORIES_PATH = DATASET_ROOT / "repositories"
    COMMITS_PATH = DATASET_ROOT / "commits"

    API_ENDPOINT = "https://api.github.com/repos"
    HEADERS = {
        "User-Agent": os.getenv("GIT_USER"),
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {os.getenv('GIT_BEARER_TOKEN')}",
    }

    COMMITS_PER_PAGE = 90


def ensure_paths_exist():
    Config.DATASET_ROOT.mkdir(exist_ok=True)
    Config.METADATA_PATH.mkdir(exist_ok=True)
    Config.ANNOTATED_METADATA_PATH.mkdir(exist_ok=True)
    Config.RAW_DIFFS_PATH.mkdir(exist_ok=True)
    Config.PREPROCESSED_DIFFS_PATH.mkdir(exist_ok=True)
    Config.REPOSITORIES_PATH.mkdir(exist_ok=True)
    Config.COMMITS_PATH.mkdir(exist_ok=True)


def get_content_from_url(url: str) -> Response | None:
    response = requests.get(url, headers=Config.HEADERS)

    if response.status_code == 403:
        logger.info(f"💤 Process Suspended (API Limit Reached) ({url})")
        while response.status_code == 403:
            for _ in trange(
                1200, desc="💤 Process Suspended (API Limit Reached)", leave=False
            ):
                time.sleep(1)

            response = requests.get(url, headers=Config.HEADERS)  # Retry

        logger.info(f"👍 Process Resumed ({url})")

    if response.status_code != 200:
        logger.error(f"❌ Status code {response.status_code} for {url}")
        return None

    return response


# @cache
def get_nlp_models() -> tuple[stanza.Pipeline, spacy.Language]:
    # Its very unlikely we would need 2 instances of npl pipelines, especialy since stancy depends on GPU
    stanza_nlp = stanza.Pipeline(processors="tokenize,pos,lemma,depparse", download_method=None)
    spacy_nlp = spacy.load("en_core_web_sm")
    return stanza_nlp, spacy_nlp


@dataclass
class MainArgs:
    scrape_metadata: bool = field(default=False)
    annotate_metadata: bool = field(default=False)
    scrape_diffs: bool = field(default=False)
    preprocess_diffs: bool = field(default=False)
    push_to_hub: bool = field(default=False)
    push_raw_diffs_to_hub: bool = field(default=False)
    push_analysed_metadata: bool = field(default=False)

    nproc: int = field(
        default=1,
        metadata={
            "help": "How many processes to use for filtering metadata (num proc should == gpu count)"
        },
    )

    rerun_ruled_based_filter: bool = field(
        default=False,
        metadata={
            "help": "If true, will rerun the rule based filters on (already) prefiltered metadata. (So that you don't "
            "have to run spacy and stanza again). overwrite_existing_files must be set to True for this to work."
            "param filter_metadata and rerun_ruled_based_filter does not make sense together."
        },
    )

    overwrite_existing_files: bool = field(
        default=False,
        metadata={
            "help": "Overwrite existing files. It does not apply for scraping commits."
        },
    )


def parse_args() -> MainArgs:
    parser = HfArgumentParser(MainArgs)
    return parser.parse_args_into_dataclasses()[0]


def extract_subject(commit_message: pd.Series) -> pd.Series:
    subject = commit_message.map(lambda msg: msg.split("\n", 1)[0])
    subject = subject.str.lower()

    # Remove conventional commit prefixes
    re = r"^[^\n]+(:)"
    subject = subject.str.replace(re, "")

    # Remove issues id
    re = r"(#[0-9]{4,5})|(\(#[0-9]{4,5}\))"
    subject = subject.str.replace(re, "")

    # Remove trailing punctuation
    subject = subject.str.removesuffix(".")

    # Remove trailing whitespaces
    subject = subject.str.strip()

    return subject


def read_repositories_txt():
    with open("repositories.txt", "r") as f:
        repos_url = f.read().splitlines()
    repos = []
    for url in repos_url:
        if url.startswith("#"):
            continue

        owner, repo = url.split("/")[-2:]
        repos.append((url, owner, repo))
    return repos
