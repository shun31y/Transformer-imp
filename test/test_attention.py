# test/test_attention.py

import torch
import pytest

from src.attention import CausalSelfAttention


def test_attention_output_shape_matches_input_shape():
    torch.manual_seed(0)

    x = torch.randn(2, 8, 32)

    attn = CausalSelfAttention(
        d_model=32,
        num_heads=4,
        block_size=8,
        dropout=0.0,
    )

    out = attn(x)

    assert out.shape == x.shape


def test_attention_raises_if_d_model_is_not_divisible_by_num_heads():
    with pytest.raises(AssertionError):
        CausalSelfAttention(
            d_model=30,
            num_heads=4,
            block_size=8,
            dropout=0.0,
        )


def test_attention_output_is_finite():
    torch.manual_seed(0)

    x = torch.randn(2, 8, 32)

    attn = CausalSelfAttention(
        d_model=32,
        num_heads=4,
        block_size=8,
        dropout=0.0,
    )

    out = attn(x)

    assert torch.isfinite(out).all()


def test_attention_allows_backward():
    torch.manual_seed(0)

    x = torch.randn(2, 8, 32, requires_grad=True)

    attn = CausalSelfAttention(
        d_model=32,
        num_heads=4,
        block_size=8,
        dropout=0.0,
    )

    out = attn(x)
    loss = out.mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_attention_does_not_use_future_tokens():
    """
    Causal attention の外部仕様だけを確認するテスト。

    入力系列の未来側だけを変更したとき、
    それより前の出力が変わらなければよい。

    これは内部実装名や mask の持ち方には依存しない。
    """
    torch.manual_seed(0)

    B = 1
    T = 8
    d_model = 32
    num_heads = 4

    attn = CausalSelfAttention(
        d_model=d_model,
        num_heads=num_heads,
        block_size=T,
        dropout=0.0,
    )
    attn.eval()

    x1 = torch.randn(B, T, d_model)
    x2 = x1.clone()

    # 位置4以降だけ変更する
    x2[:, 4:, :] = torch.randn_like(x2[:, 4:, :]) * 100.0

    out1 = attn(x1)
    out2 = attn(x2)

    # 位置0〜3は未来側の変更に影響されてはいけない
    assert torch.allclose(out1[:, :4, :], out2[:, :4, :], atol=1e-5)