import logging
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import cache
from typing import List

import numpy as np
from pydantic import BaseModel
from tqdm import tqdm

import stanza
import pandas as pd
import spacy

from utils import Config, get_nlp_models, MainArgs, extract_subject

logger = logging.getLogger("metadata-annotation")


# Models
class CommitMetadata(BaseModel):
    sha: str
    author: str
    committer: str
    message: str


class RepositoryMetadata(BaseModel):
    owner: str
    name: str
    commits_metadata: List[CommitMetadata]

    def commits_to_df(self):
        return pd.DataFrame([item.__dict__ for item in self.commits_metadata])


class VerbObjectAnnotation:
    def __init__(self):
        self.stanza_nlp, self.spacy_nlp = get_nlp_models()

    def is_verb_object_spacy(self, subject: str) -> bool:
        tokens = self.spacy_nlp(subject)
        if len(tokens) <= 0:
            return False

        # First word has to be verb in base form
        if tokens[0].dep_ != "ROOT" or tokens[0].tag_ != "VB":
            return False

        # Get first _object_ in a sentance
        idx, val = next(
            (
                (idx, token.text)
                for idx, token in enumerate(tokens)
                if token.dep_ == "dobj"
            ),
            (None, None),
        )
        if val is None:
            return False

        # All tokens between verb <-> object must relate to the object
        return all(token.head.text == val for token in tokens[1:idx])

    def is_verb_object_stanza(self, subject: str) -> bool:
        sentences = self.stanza_nlp(subject).sentences
        if len(sentences) <= 0:
            return False

        tokens = sentences[0].words
        if len(tokens) <= 0:
            return False

        # First word has to be verb in base form
        if tokens[0].deprel != "root" or tokens[0].xpos != "VB":
            return False

        # Get first _object_ in a sentence
        idx, id = next(
            (
                (idx, token.id)
                for idx, token in enumerate(tokens)
                if token.deprel == "obj"
            ),
            (None, None),
        )
        if idx is None:
            return False

        # All tokens between verb <-> object must relate to the object
        return all(token.head == id for token in tokens[1:idx])

    @staticmethod
    @cache
    def shared():
        return VerbObjectAnnotation()


def rule_based_annotations(df: pd.DataFrame):
    logger.info(f"📐 Processing static rules")

    df["subject_length"] = df["subject"].str.len()
    df["is_chore"] = df["subject"].str.startswith(("merge", "bump", "rollback"))

    df["is_bot"] = df["author"].str.lower().str.endswith("[bot]")

    df["subject_word_count"] = df["subject"].str.split().str.len()
    return df


def ml_annotations(df: pd.DataFrame):
    tqdm.pandas()
    verb_object_filter = VerbObjectAnnotation.shared()

    logger.info(f"📐 Processing Verb-Object Rules(Spacy)")
    df["verb_object_spacy"] = df["subject"].progress_map(
        verb_object_filter.is_verb_object_spacy
    )

    logger.info(f"📐 Processing Verb-Object Rules(Stanza)")
    df["verb_object_stanza"] = df["subject"].progress_map(
        verb_object_filter.is_verb_object_stanza
    )
    return df


def requirements_mask(df: pd.DataFrame):
    return (
        (df["is_bot"] == False)  # Exclude bots
        & (df["is_chore"] == False)  # Exclude chore commits -> merge etc
        & (df["subject_length"] <= 50)  # Exclude commits with subject line too long
        & (
            df["subject_word_count"] > 2
        )  # Exclude commits such as "update text.php" or "remove variables"
        & (  # Include commits that pass at least one vo filter
            df["verb_object_spacy"] == True | (df["verb_object_stanza"] == True)
        )
    )


def filtered(commits_df: pd.DataFrame):
    commits_df["subject"] = extract_subject(commits_df["message"])
    commits_df = rule_based_annotations(commits_df)
    commits_df = ml_annotations(commits_df)

    logger.info(f"Applying filters")

    commits_df["fits_requirements"] = requirements_mask(commits_df)
    return commits_df


def annotate_metadata(owner: str, repo: str, args: MainArgs):
    source_file = Config.METADATA_PATH / f"{owner}-{repo}.json"
    output_file = Config.ANNOTATED_METADATA_PATH / f"{owner}-{repo}.csv"
    if output_file.exists() and not args.overwrite_existing_files:
        logger.info(f"📣 {owner}/{repo} already processed")
        return

    repo_metadata = RepositoryMetadata.parse_file(source_file)
    df = filtered(repo_metadata.commits_to_df())

    logger.info(
        f"🔍 {df[df['fits_requirements'] == True].shape[0]} commits fit requirements"
    )
    df = df.dropna()
    df.to_csv(output_file, index=False)


def rerun_rule_based_filters(owner: str, repo: str, args: MainArgs):
    path = Config.ANNOTATED_METADATA_PATH / f"{owner}-{repo}.csv"
    if path.exists() and not args.overwrite_existing_files:
        logger.info(f"📣 {owner}/{repo} already processed")
        return

    df = pd.read_csv(path)
    df = rule_based_annotations(df)
    df["fits_requirements"] = requirements_mask(df)
    df = df.dropna()
    df.to_csv(path, index=False)
