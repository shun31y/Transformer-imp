from pathlib import Path

import torch

from src.model import MiniGPT
from src.tokenizer import CharTokenizer


def load_tokenizer(tokenizer_state: dict) -> CharTokenizer:
    tokenizer = CharTokenizer("")
    tokenizer.stoi = tokenizer_state["stoi"]
    tokenizer.itos = tokenizer_state["itos"]
    return tokenizer


def get_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except RuntimeError:
            pass
    return torch.device("cpu")


def main() -> None:
    checkpoint_path = Path("checkpoint.pt")
    prompt = " "
    max_new_tokens = 128

    device = get_device()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    tokenizer = load_tokenizer(checkpoint["tokenizer_state"])
    if prompt and any(ch not in tokenizer.stoi for ch in prompt):
        prompt = tokenizer.itos[0]

    model = MiniGPT(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        block_size=config["block_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)

    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)

    generated_text = tokenizer.decode(output_ids[0].tolist())
    print(generated_text)


if __name__ == "__main__":
    main()
