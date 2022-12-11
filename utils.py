from pathlib import Path

relative_root = Path(__file__).parent.resolve()

class Config:
    DATASET_ROOT = relative_root / 'dataset'
    METADATA_PATH = DATASET_ROOT / 'a-metadata'
    PRE_FILTERED_METADATA_PATH = DATASET_ROOT / 'b-prefiltered-metadata'
    RAW_DIFFS_PATH = DATASET_ROOT / 'c-raw-diffs'
    PREPROCESSED_DIFFS_PATH = DATASET_ROOT / 'd-preprocessed-diffs'
    MODEL_CHECKPOIN_BASE_PATH = relative_root / 'output-model'

    API_ENDPOINT = "https://api.github.com/repos"
    HEADERS = {
        "User-Agent": "mamiksik",
        "Accept": "application/vnd.github+json",
        "Authorization": "Bearer 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",
    }

    COMMITS_PER_PAGE = 90