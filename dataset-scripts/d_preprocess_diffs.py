import logging
import re
from pathlib import Path

import pandas as pd

import os

from pydantic import BaseModel
from typing import List
from tqdm import tqdm
from utils import Config, MainArgs, extract_subject

logger = logging.getLogger("diff-preprocessor")

SUPPORTED_EXTENSIONS = {
    ".java": "Java",
    ".py": "Python",
    ".js": "Javascript",
    ".php": "PHP",
    ".rb": "Ruby",
    ".go": "Go",
    ".md": "Text",
}


# Models
class FileDiff(BaseModel):
    filename: str
    previous_filename: str | None
    status: str
    patch: str


class Commit(BaseModel):
    message: str
    files: List[FileDiff]


def parse_file(accumulator: list[str], file_diff: FileDiff):
    if file_diff.previous_filename is None:
        accumulator.append(f"<ide><path>{file_diff.filename}")
    else:
        accumulator.append(f"<add><path>{file_diff.filename}")
        accumulator.append(f"<del><path>{file_diff.previous_filename}")

    for line in file_diff.patch.splitlines():
        line = re.sub("@@[^@@]*@@", "", line)
        if len(line) == 0:
            continue

        if line[0] == "+":
            line = line.replace("+", "<add>", 1)
        elif line[0] == "-":
            line = line.replace("-", "<del>", 1)
        else:
            line = f"<ide>{line}"

        accumulator.append(line)


def parse_content_type(commit: Commit):
    files_extension = map(lambda file: Path(file.filename).suffix, commit.files)
    per_file_content_type = list(
        map(lambda ext: SUPPORTED_EXTENSIONS.get(ext, None), files_extension)
    )

    if any(map(lambda x: x is None, per_file_content_type)) or len(per_file_content_type) == 0:
        return None, None

    main_lang = max(set(per_file_content_type), key=per_file_content_type.count)
    if len(set(per_file_content_type)) > 1:
        return "Mixed", main_lang

    return per_file_content_type[0], main_lang


def preprocess(diff_root: Path, passing_shas: set[str]) -> pd.DataFrame:
    items = []
    for filename in tqdm(os.listdir(diff_root)):
        if not os.path.isfile(diff_root / filename):
            continue
        sha = filename.removesuffix(".json")
        # Check that commit is in prefiltered dataset (in case we updated constrains)
        if sha not in passing_shas:
            continue

        commit: Commit = Commit.parse_file(diff_root / filename)
        content_type, main_lang = parse_content_type(commit)

        # If there is at least one unsupported file type in the commit, skip it
        if content_type is None:
            continue

        line_accumulator = []
        for file_diff in commit.files:
            parse_file(line_accumulator, file_diff)

        patch = "\n".join(line_accumulator)
        items.append(
            {
                "content_type": content_type,
                "main_lang": main_lang,
                "message": commit.message,
                "sha": sha,
                "patch": patch,
                "file_count": len(commit.files),
            }
        )

    df = pd.DataFrame.from_records(
        items, columns=["content_type", "main_lang", "message", "sha", "patch", "file_count"]
    )
    df["message"] = extract_subject(df["message"])
    return df


def preprocess_diffs(owner: str, repo: str, args: MainArgs):
    output_file = Config.PREPROCESSED_DIFFS_PATH / f"{owner}-{repo}.csv"
    diff_root = Config.RAW_DIFFS_PATH / f"{owner}-{repo}"

    if output_file.exists() and not args.overwrite_existing_files:
        logger.info(f"📣 {owner}/{repo} already preprocessed")
        return

    df_prefiltered = pd.read_csv(Config.ANNOTATED_METADATA_PATH / f"{owner}-{repo}.csv")
    df_prefiltered = df_prefiltered[df_prefiltered["fits_requirements"] == True]

    df = preprocess(diff_root, set(df_prefiltered["sha"]))

    df = df.dropna(axis="columns", how="any")
    if len(df) == 0:
        logger.info(f"❓{owner}/{repo} has no diffs. File not saved")
        return

    df.to_csv(output_file, index=False)
    logger.info(f"💾 Dataset of {len(df)} commits saved to {output_file}")
