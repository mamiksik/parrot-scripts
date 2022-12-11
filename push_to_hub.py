from transformers import RobertaForMaskedLM

from utils import Config


def main():
    path = Config.MODEL_CHECKPOINT_BASE_PATH / 'code-berta-large-experiment-4'
    print(f'▶️  Model name: {path}')

    local_model = RobertaForMaskedLM.from_pretrained(path)
    local_model.push_to_hub('mamiksik/CommitPredictor', private=True)


if __name__ == '__main__':
    main()