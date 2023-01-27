import os
from argparse import ArgumentParser
from pathlib import Path
from typing import List

import pandas as pd
# from huggingface_hub.utils import tqdm
from tqdm import tqdm

from b_annotate_metadata import VerbObjectAnnotation
from utils import ensure_paths_exist
from pydrill import Commits, Commit, FileDiff
import re
import itertools

prefix_re = re.compile('r"^[^\n]+(:)')
issue_re = re.compile('r"(#[0-9]{4,5})|(\(#[0-9]{4,5}\))')


def extract_subject(message: str) -> str:
    subject = message.splitlines()
    if len(subject) == 0:
        return ""

    subject = subject[0].lower()
    subject = re.sub(prefix_re, "", subject)
    subject = re.sub(issue_re, "", subject)
    subject = subject.removesuffix(".").strip()

    return subject


def commit_quality_filter(commit: Commit) -> bool:
    subject = extract_subject(commit.message)
    if commit.author.endswith("[bot]"):
        return False

    if subject.startswith(("merge", "bump", "rollback")):
        return False

    if len(subject) > 50 or len(subject) == 0:
        return False

    if len(subject.split()) < 3:
        return False

    # We compute the vo-annotation, last to save time since it's the most
    # expensive operation.
    vo_filter = VerbObjectAnnotation.shared()
    if not vo_filter.is_verb_object_spacy(
        subject
    ) and not vo_filter.is_verb_object_stanza(subject):
        return False

    return True


def combine_patches(files: List[FileDiff]) -> str:
    accumulator = []
    for file_diff in files:
        if file_diff.path_after != file_diff.path_before:
            accumulator.append(f"<add><path>{file_diff.path_after}")
            accumulator.append(f"<del><path>{file_diff.path_before}")
        else:
            accumulator.append(f"<ide><path>{file_diff.path_before}")

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
    return "\n".join(accumulator)


def main(args):
    print("Loading commits...")

    print("Init VO-Filters")
    _ = VerbObjectAnnotation.shared()

    accumulator = []
    source_path = Path(args.source)

    for filename in (pbar := tqdm(os.listdir(args.source))):
        pbar.set_description(f"Processing {filename}")
        if not filename.endswith(".json"):
            continue

        source = Commits.parse_file(source_path / filename)
        commits = filter(commit_quality_filter, source.commits)
        # commits = tqdm(
        #     filter(commit_quality_filter, source.commits),
        #     desc="Filtering commits",
        #     total=len(source.commits),
        #     leave=None
        # )
        accumulator.append(list(map(lambda commit: {
            "repository": commit.repository,
            "mixed_content": commit.mixed_content,
            "message": commit.message,
            "hash": commit.hash,
            "patch": combine_patches(commit.files),
        }, commits)))

    df = pd.DataFrame.from_records(itertools.chain(*accumulator))

    print(f"Saving {len(df)} commits")
    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    _parser = ArgumentParser()
    _parser.add_argument("--output", dest="output", type=str)
    _parser.add_argument("--source_dir", dest="source", type=str)
    _args = _parser.parse_args()
    ensure_paths_exist()
    main(_args)
