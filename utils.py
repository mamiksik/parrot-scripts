from pathlib import Path

relative_root = Path(__file__).parent.resolve()


class Config:
    MODEL_CHECKPOINT_BASE_PATH = relative_root / 'output-model'
