from copy import copy
from math import log

from transformers import RobertaTokenizer, RobertaForMaskedLM, pipeline


def fmask(prefix):
    c = len(prefix.split())
    return prefix + ["<mask>" for _ in range(50 - c)]

class Mask:
    def __init__(self, mask=None):
        self.mask = [] if mask is None else mask

    def append(self, score, token):
        self.mask.append((score, token))

    def __copy__(self):
        return type(self)(self.mask.copy())

    def __call__(self):
        c = len(self.mask)
        return list(map(lambda x: x[1], self.mask)) + ["<mask>" for _ in range(50 - c)]

    def score(self):
        return sum(map(lambda x: log(x[0]), self.mask))

    def __str__(self):
        return " ".join(self())

def main():
    model = RobertaForMaskedLM.from_pretrained("mamiksik/CommitPredictor")
    tokenizer = RobertaTokenizer.from_pretrained("mamiksik/CommitPredictor")

    pipe = pipeline('fill-mask', model=model, tokenizer=tokenizer)

    # encoding = tokenizer(patch, msg, padding="max_length", truncation='only_first', return_tensors='pt')
    # input_ids = encoding['input_ids']
    # attention_mask = encoding['attention_mask']
    # output = model(input_ids, attention_mask=attention_mask, output_attentions=False)
    # predictions = outputs.logits.argmax(-1)

    # predictions = pipe(patch + msg)
    # print(predictions)

    patch = """<keep> def main():
<remove>  name = "John"
<keep>    print("Hello World!")</s></s>"""

    masks: list[Mask] = [Mask()]
    while True:
        candidates: list[Mask] = []
        for mask in masks:
            predictions = pipe(patch + ' '.join(mask()))[0]
            for prediction in predictions:
                sub_mask = copy(mask)
                sub_mask.append(prediction['score'], prediction['token_str'])
                candidates.append(sub_mask)

        candidates.sort(key=lambda x: x.score(), reverse=True)
        masks = candidates[:2]




    # input_ids = tokenizer.encode(patch, msg, return_tensors='pt')
    # model.generate(
    #     input_ids,
    #     max_length=50,
    #     num_beams=5,
    #     early_stopping=True
    # )



if __name__ == '__main__':
    main()
