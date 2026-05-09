# test/test_blocks.py

import torch

from src.blocks import FeedForward, TransformerBlock


def test_feedforward_output_shape_matches_input_shape():
    torch.manual_seed(0)

    x = torch.randn(2, 8, 32)

    mlp = FeedForward(
        d_model=32,
        dropout=0.0,
    )

    out = mlp(x)

    assert out.shape == x.shape


def test_feedforward_allows_backward():
    torch.manual_seed(0)

    x = torch.randn(2, 8, 32, requires_grad=True)

    mlp = FeedForward(
        d_model=32,
        dropout=0.0,
    )

    out = mlp(x)
    loss = out.mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_transformer_block_output_shape_matches_input_shape():
    torch.manual_seed(0)

    x = torch.randn(2, 8, 32)

    block = TransformerBlock(
        d_model=32,
        num_heads=4,
        block_size=8,
        dropout=0.0,
    )

    out = block(x)

    assert out.shape == x.shape


def test_transformer_block_output_is_finite():
    torch.manual_seed(0)

    x = torch.randn(2, 8, 32)

    block = TransformerBlock(
        d_model=32,
        num_heads=4,
        block_size=8,
        dropout=0.0,
    )

    out = block(x)

    assert torch.isfinite(out).all()


def test_transformer_block_allows_backward():
    torch.manual_seed(0)

    x = torch.randn(2, 8, 32, requires_grad=True)

    block = TransformerBlock(
        d_model=32,
        num_heads=4,
        block_size=8,
        dropout=0.0,
    )

    out = block(x)
    loss = out.mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()