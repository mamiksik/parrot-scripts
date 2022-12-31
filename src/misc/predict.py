from copy import copy
from math import log

import torch

# from typing import Self

from transformers import RobertaTokenizer, RobertaForMaskedLM, pipeline


class PredictionCandidate:
    def __init__(self, mask=None):
        self.mask = [] if mask is None else mask

    def copy_with(self, score: float, token: str) -> "PredictionCandidate":
        return type(self)(mask=self.mask.copy() + [(score, token)])

    def score(self) -> int:
        return sum(map(lambda x: log(x[0]), self.mask))

    def masked(self):
        return str(self) + "".join(["<mask>" for _ in range(20 - len(self.mask))])

    def __str__(self) -> str:
        return " ".join(map(lambda x: x[1], self.mask))


def predict(tokenizer, model, text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    # outputs = model(**inputs)
    # predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    logits = model(**inputs).logits
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[0]

    mask_token_logits = logits[0, mask_token_index, :]
    top_3_tokens = torch.topk(mask_token_logits, 3, dim=1).indices[0].tolist()

    # values, predictions = predictions.topk(5)
    # for i, (_values, _predictions) in enumerate(zip(values.tolist(), predictions.tolist())):
    #     for v, p in zip(_values, _predictions):
    #         print(tokenizer.decode(p))

    return outputs


def main():
    model = RobertaForMaskedLM.from_pretrained("mamiksik/Testing")
    tokenizer = RobertaTokenizer.from_pretrained("mamiksik/CommitPredictor")

    pipe = pipeline("fill-mask", model=model, tokenizer=tokenizer)

    # encoding = tokenizer(patch, msg, padding="max_length", truncation='only_first', return_tensors='pt')
    # input_ids = encoding['input_ids']
    # attention_mask = encoding['attention_mask']
    # output = model(input_ids, attention_mask=attention_mask, output_attentions=False)
    # predictions = outputs.logits.argmax(-1)

    # predictions = pipe(patch + msg)
    # print(predictions)

    patch = (
        "<keep>def main():" '<keep>   print("Hello World!")' '<remove>   name = "John"'
    )

    candidates: list[PredictionCandidate] = [PredictionCandidate()]
    predict(tokenizer, model, f"{patch}<msg>{candidates[0].masked()}")

    # for _ in range(10):
    #     iter_candidates: list[PredictionCandidate] = []
    #     for candidate in candidates:
    #         predictions = pipe(f"<msg>{candidate.masked()}{patch}")[0]
    #         for prediction in predictions:
    #             iter_candidates.append(candidate.copy_with(prediction['score'], prediction['token_str']))
    #
    #     iter_candidates.sort(key=lambda x: x.score(), reverse=True)
    #     candidates = iter_candidates[:4]
    #
    # print([str(x) for x in candidates])


if __name__ == "__main__":
    main()
