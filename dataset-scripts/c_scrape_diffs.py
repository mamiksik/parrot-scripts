import logging

import pandas as pd

import os

from pydantic import BaseModel, root_validator, ValidationError
from typing import List
from tqdm import tqdm
from utils import Config, get_content_from_url, MainArgs

logger = logging.getLogger("diff-scraper")


# Models
class FileDiff(BaseModel):
    filename: str
    previous_filename: str | None
    status: str
    patch: str


class Commit(BaseModel):
    message: str
    files: List[FileDiff]

    @root_validator(pre=True)
    def extract_message(cls, values):
        values["message"] = values["commit"]["message"]
        return values


def scrape_repository(owner: str, repo: str):
    df = pd.read_csv(Config.ANNOTATED_METADATA_PATH / f"{owner}-{repo}.csv")
    shas = df[df["fits_requirements"] == True]["sha"]

    diffdir = Config.RAW_DIFFS_PATH / f"{owner}-{repo}"

    if not os.path.exists(diffdir):
        diffdir.mkdir(parents=True)

    to_download = []
    for sha in tqdm(shas, desc="Checking for already downloaded commits", leave=False):
        if not os.path.exists(diffdir / f"{sha}.json"):
            to_download.append(sha)

    logger.info(f"💾 {len(to_download)}/{len(shas)} to download")

    for sha in (pbar := tqdm(to_download)):
        pbar.set_description(f"Downloading commit {sha}")
        logger.debug(f"📄 Downloading commit {sha}")

        try:
            response = get_content_from_url(
                f"{Config.API_ENDPOINT}/{owner}/{repo}/commits/{sha}"
            )
            commit = Commit.parse_raw(response.content)
        except (ValidationError, AttributeError):
            logger.warning(
                f"Skipping commit {sha} due to error (most likely the commit is too large)"
            )
            continue
        else:
            with open(diffdir / f"{sha}.json", "w") as file:
                file.write(commit.json())
