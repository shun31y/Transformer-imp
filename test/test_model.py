# test/test_model.py

import torch
import pytest

from src.model import MiniGPT


def build_tiny_model():
    return MiniGPT(
        vocab_size=20,
        block_size=8,
        d_model=32,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
    )


def test_model_forward_without_targets_returns_logits_and_none_loss():
    torch.manual_seed(0)

    model = build_tiny_model()

    input_ids = torch.randint(0, 20, (2, 8))

    logits, loss = model(input_ids)

    assert logits.shape == (2, 8, 20)
    assert loss is None


def test_model_forward_with_targets_returns_logits_and_loss():
    torch.manual_seed(0)

    model = build_tiny_model()

    input_ids = torch.randint(0, 20, (2, 8))
    targets = torch.randint(0, 20, (2, 8))

    logits, loss = model(input_ids, targets)

    assert logits.shape == (2, 8, 20)
    assert loss is not None
    assert loss.ndim == 0


def test_model_loss_is_finite():
    torch.manual_seed(0)

    model = build_tiny_model()

    input_ids = torch.randint(0, 20, (2, 8))
    targets = torch.randint(0, 20, (2, 8))

    _, loss = model(input_ids, targets)

    assert torch.isfinite(loss)


def test_model_allows_backward():
    torch.manual_seed(0)

    model = build_tiny_model()

    input_ids = torch.randint(0, 20, (2, 8))
    targets = torch.randint(0, 20, (2, 8))

    _, loss = model(input_ids, targets)
    loss.backward()

    grads = [
        p.grad
        for p in model.parameters()
        if p.requires_grad and p.grad is not None
    ]

    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_model_raises_if_sequence_length_exceeds_block_size():
    torch.manual_seed(0)

    model = build_tiny_model()

    input_ids = torch.randint(0, 20, (2, 9))

    with pytest.raises(AssertionError):
        model(input_ids)


def test_generate_increases_sequence_length():
    torch.manual_seed(0)

    model = build_tiny_model()
    model.eval()

    input_ids = torch.randint(0, 20, (1, 4))

    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=6,
    )

    assert out.shape == (1, 10)


def test_generate_outputs_integer_token_ids_in_vocab_range():
    torch.manual_seed(0)

    model = build_tiny_model()
    model.eval()

    input_ids = torch.randint(0, 20, (1, 4))

    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=6,
    )

    assert out.dtype == torch.long
    assert out.min().item() >= 0
    assert out.max().item() < 20