import logging
import os
import warnings
from multiprocessing import Pool, set_start_method

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from a_scrape_metadata import scrape_metadata
from b_annotate_metadata import annotate_metadata, rerun_rule_based_filters
from c_scrape_diffs import scrape_repository
from d_preprocess_diffs import preprocess_diffs
from e_push_to_hub import (
    push_final_dataset_to_hub,
    push_raw_diffs_to_hub,
    push_analysed_metadata,
)
from utils import ensure_paths_exist, parse_args, MainArgs

logger = logging.getLogger("main")
warnings.simplefilter(action="ignore", category=FutureWarning)


def process_repository(owner: str, repo: str, args: MainArgs):
    logger.info(f"⌛️ Processing repository {owner}/{repo} [{os.getpid()}]")
    try:
        if args.scrape_metadata:
            logger.info("📥 Scraping metadata")
            scrape_metadata(owner, repo, args)

        if args.annotate_metadata:
            logger.info(f"🕵  Filtering commit messages")
            annotate_metadata(owner, repo, args)

        if args.rerun_ruled_based_filter:
            logger.info(f"📐  Re-running rule-based filters")
            rerun_rule_based_filters(owner, repo, args)

        if args.scrape_diffs:
            logger.info("📥 Scraping commit diffs")
            scrape_repository(owner, repo)

        if args.preprocess_diffs:
            logger.info("🧹 Preprocessing diffs")
            preprocess_diffs(owner, repo, args)

        logger.info(f"🟢 Repository {owner}/{repo} done...")
        tqdm.write("")
    except Exception as e:
        logger.error(f"❌ Repository {owner}/{repo} failed with {e} [{os.getpid()}]")


def main():
    main_args = parse_args()
    logger.info(f"⭐ Starting process")
    logger.debug(f"GIT USER: {os.getenv('GIT_USER')}")
    logger.debug(f"GIT TOKEN: {os.getenv('GIT_BEARER_TOKEN')}")
    ensure_paths_exist()

    # https://github.com/github/gh-ost/commits/master
    with open("repositories.txt", "r") as f:
        repos_url = f.read().splitlines()

    repos = []
    for url in repos_url:
        if url.startswith("#"):
            continue

        owner, repo = url.split("/")[-2:]
        repos.append((owner, repo))

    tqdm.write("\n")

    if main_args.nproc > 1:
        with Pool(processes=main_args.nproc) as pool:
            tasks = [
                pool.apply_async(process_repository, (owner, repo, main_args))
                for owner, repo in repos
            ]
            [task.get() for task in tasks]
    else:
        for owner, repo in repos:
            process_repository(owner, repo, main_args)

    if main_args.push_analysed_metadata:
        tqdm.write("")
        logger.info("🤗 Pushing analysed diffs to hub")
        push_analysed_metadata(repos)

    if main_args.push_raw_diffs_to_hub:
        tqdm.write("")
        logger.info("🤗 Pushing raw diffs to hub")
        push_raw_diffs_to_hub(repos)

    if main_args.push_to_hub:
        tqdm.write("")
        logger.info("🤗 Creating final dataset")
        push_final_dataset_to_hub()


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    set_start_method("spawn")

    logging.basicConfig(
        filename="status.log",
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s: %(message)s",
    )
    logging.getLogger().addHandler(logging.StreamHandler())
    with logging_redirect_tqdm():
        main()
