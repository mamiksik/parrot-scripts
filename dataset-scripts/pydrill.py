from argparse import ArgumentParser
from typing import Literal, TypeAlias

import pandas as pd
from pydantic import BaseModel
from pydriller import Repository
import pydriller
from tqdm import tqdm

from utils import Config, ensure_paths_exist

PROGRAMING_LANGUAGE_CONTENT = {
    "py": "Python",
    "java": "Java",
    "js": "JavaScript",
    "go": "Go",
    "php": "PHP",
    "rb": "Ruby",
}

NATURAL_LANGUAGE_CONTENT = {
    "md": "Markdown",
}

SUPPORTED_CONTENT = NATURAL_LANGUAGE_CONTENT | PROGRAMING_LANGUAGE_CONTENT
MixedContentType: TypeAlias = Literal["NO", "NL_ONLY", "PL"]
ChangeType: TypeAlias = Literal["ADD", "MODIFY", "DELETE", "RENAME", "COPY", "UNKNOWN"]
JSON: TypeAlias = str


class FileDiff(BaseModel):
    status: ChangeType
    content_type: str
    patch: str

    path_before: str | None
    path_after: str | None

    content_before: str | None
    content_after: str | None


class Commit(BaseModel):
    repository: str
    author: str
    committer: str
    message: str
    hash: str
    files: list[FileDiff]
    mixed_content: MixedContentType


class Commits(BaseModel):
    commits: list[Commit]


def parse_content_type(filename: str) -> str | None:
    ext = filename.rsplit(".")[-1]
    return SUPPORTED_CONTENT.get(ext, None)


def parse_files(commit: pydriller.Commit) -> tuple[list[FileDiff], MixedContentType] | None:
    files = []
    nl_type = set()
    pl_type = set()

    for file in commit.modified_files:
        if (content_type := parse_content_type(file.filename)) is None:
            return None

        if content_type in NATURAL_LANGUAGE_CONTENT.values():
            nl_type.add(content_type)
        else:
            pl_type.add(content_type)

        files.append(
            FileDiff(
                status=file.change_type.name,
                patch=file.diff,
                content_type=content_type,
                path_before=file.old_path,
                path_after=file.new_path,
                content_before=file.content_before,
                content_after=file.content,
            )
        )

    mixed_content: MixedContentType = "NO"
    if len(pl_type) > 1:
        mixed_content = "PL"
    elif len(nl_type) > 0:
        mixed_content = "NL_ONLY"

    return files, mixed_content


def parse_repository(repository: Repository) -> Commits:
    commits = []
    for commit in tqdm(repository.traverse_commits(), desc="Commits"):
        if (result := parse_files(commit)) is None:
            continue
        files, mixed_content = result
        commits.append(Commit(
            repository=repository.git.project_name,
            author=commit.author.name,
            committer=commit.committer.name,
            message=commit.msg,
            hash=commit.hash,
            files=files,
            mixed_content=mixed_content
        ))

    return Commits(commits=commits)


def main(args):
    df = pd.read_csv(args.source, encoding='latin-1')
    df = df[['name', 'defaultBranch', 'mainLanguage']]
    df = df.rename(columns={'name': 'repo', 'defaultBranch': 'main_branch', 'mainLanguage': 'language'})

    for row in tqdm(df.itertuples(), desc="Repositories: "):
        repo_id = row.repo.replace('/', '-')
        tqdm.write(f"Processing {repo_id}")

        output_dir = Config.COMMITS_PATH / row.language.lower()
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"{repo_id}.json"
        if output_file.exists():
            tqdm.write(f"Skipping {repo_id}")
            continue

        try:

            repository = Repository(
                f"https://github.com/{row.repo}",
                only_in_branch=row.main_branch,
                only_no_merge=True,
                skip_whitespaces=True,
                include_deleted_files=True,
                include_remotes=True
            )
            commits = parse_repository(repository)
            with open(output_file, "w") as f:
                f.write(commits.json())
            tqdm.write(f"Saved {output_file}")
        except Exception as e:
            tqdm.write(f"Error processing {repo_id}: {e}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--source", dest="source", type=str)
    args = parser.parse_args()
    ensure_paths_exist()
    main(args)
