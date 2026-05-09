import numpy as np


class CharTokenizer:
    def __init__(self, text: str):
        # text に含まれる文字だけを語彙にする
        chars = sorted(list(set(text)))

        # string to integer
        self.stoi = {ch: i for i, ch in enumerate(chars)}

        # integer to string
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join([self.itos[i] for i in ids])

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)


if __name__ == "__main__":
    print("Testing CharTokenizer...")