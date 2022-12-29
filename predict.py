from transformers import RobertaTokenizer, RobertaForMaskedLM, pipeline


def main():
    model = RobertaForMaskedLM.from_pretrained("mamiksik/CommitPredictor")
    tokenizer = RobertaTokenizer.from_pretrained("mamiksik/CommitPredictor")

    pipe = pipeline('fill-mask', model=model, tokenizer=tokenizer)
    result = pipe.predict("""<msg> <mask> <mask> <mask> <>
<keep> def main():
<remove>  name = "John"
<keep>    print("Hello World!")""")

    print(result)


if __name__ == '__main__':
    main()
