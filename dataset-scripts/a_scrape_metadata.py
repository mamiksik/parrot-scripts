import logging

import tqdm as tqdm
from pydantic import BaseModel, parse_raw_as, root_validator
from tqdm import trange, tqdm

from utils import Config, get_content_from_url, MainArgs

logger = logging.getLogger("metadata-screper")


class CommitMetadata(BaseModel):
    sha: str
    author: str
    committer: str
    message: str

    @root_validator(pre=True)
    def extract_metadata(cls, values):
        values["author"] = values["commit"]["author"]["name"]
        values["committer"] = values["commit"]["committer"]["name"]
        values["message"] = values["commit"]["message"]
        return values


class RepositoryMetadata(BaseModel):
    owner: str
    name: str
    commits_metadata: list[CommitMetadata]


def get_commit_count(owner: str, repo: str) -> int:
    response = get_content_from_url(
        f"{Config.API_ENDPOINT}/{owner}/{repo}/commits?per_page=1&page=1"
    )
    if response is None:
        return -1

    return int(
        response.headers.get("Link").split(",")[1].split("&page=")[1].split(">;")[0]
    )


def get_metadata_batch(
    owner: str, repo: str, per_page: int, page: int
) -> list[CommitMetadata]:
    response = get_content_from_url(
        f"{Config.API_ENDPOINT}/{owner}/{repo}/commits?per_page={per_page}&page={page}"
    )
    if response is None:
        return []

    return parse_raw_as(list[CommitMetadata], response.content)


def scrape_metadata(owner: str, repo: str, args: MainArgs):
    filepath = Config.METADATA_PATH / f"{owner}-{repo}.json"
    if filepath.exists() and not args.overwrite_existing_files:
        logger.info(f"📣 {owner}/{repo} already scraped")
        return

    commits_metadata = []
    commit_count = get_commit_count(owner, repo)
    if commit_count == -1:
        logger.error(f"❌ {owner}/{repo} failed to get commit count")
        return

    batches = commit_count // Config.COMMITS_PER_PAGE + 1
    batches = 2 if batches == 1 else batches

    for page in trange(1, batches, desc="Batch", leave=False):
        commits_metadata.extend(
            get_metadata_batch(owner, repo, Config.COMMITS_PER_PAGE, page)
        )

    repo_metadata = RepositoryMetadata(
        owner=owner, name=repo, commits_metadata=commits_metadata
    )

    with open(filepath, "w") as f:
        f.write(repo_metadata.json())

    logger.info(f"📊 {owner}/{repo} scraped {len(commits_metadata)} commits")
