from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataset import CharDataset
from src.model import MiniGPT
from src.tokenizer import CharTokenizer


def main() -> None:
    data_path = Path("data/input.txt")
    block_size = 1024
    batch_size = 32
    d_model = 128
    num_heads = 4
    num_layers = 4
    dropout = 0.1
    lr = 3e-4
    max_steps = 5000
    save_path = Path("checkpoint.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    text = data_path.read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)
    print(f"vocab size: {tokenizer.vocab_size}")
    token_ids = tokenizer.encode(text)

    dataset = CharDataset(token_ids=token_ids, block_size=block_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        block_size=block_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    data_iter = iter(dataloader)
    for step in range(1, max_steps + 1):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)

        x = x.to(device)
        y = y.to(device)

        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % 100 == 0:
            print(f"step {step}: loss {loss.item():.4f}")

    torch.save(
        {
            "model": model.state_dict(),
            "tokenizer_state": {
                "stoi": tokenizer.stoi,
                "itos": tokenizer.itos,
            },
            "config": {
                "vocab_size": tokenizer.vocab_size,
                "block_size": block_size,
                "d_model": d_model,
                "num_heads": num_heads,
                "num_layers": num_layers,
                "dropout": dropout,
            },
        },
        save_path,
    )
    print(f"saved checkpoint to {save_path}")


if __name__ == "__main__":
    main()
