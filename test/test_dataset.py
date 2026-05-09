# test/test_dataset.py

import torch

from src.dataset import CharDataset


def test_dataset_length():
    token_ids = list(range(10))
    block_size = 4

    dataset = CharDataset(token_ids, block_size)

    assert len(dataset) == len(token_ids) - block_size


def test_dataset_getitem_returns_x_and_y():
    token_ids = list(range(10))
    block_size = 4

    dataset = CharDataset(token_ids, block_size)
    x, y = dataset[0]

    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)


def test_dataset_x_y_have_block_size_shape():
    token_ids = list(range(10))
    block_size = 4

    dataset = CharDataset(token_ids, block_size)
    x, y = dataset[0]

    assert x.shape == (block_size,)
    assert y.shape == (block_size,)


def test_dataset_x_y_are_shifted_by_one_token():
    token_ids = list(range(10))
    block_size = 4

    dataset = CharDataset(token_ids, block_size)
    x, y = dataset[2]

    expected_x = torch.tensor([2, 3, 4, 5], dtype=torch.long)
    expected_y = torch.tensor([3, 4, 5, 6], dtype=torch.long)

    assert torch.equal(x, expected_x)
    assert torch.equal(y, expected_y)


def test_dataset_returns_long_tensors():
    token_ids = list(range(10))
    block_size = 4

    dataset = CharDataset(token_ids, block_size)
    x, y = dataset[0]

    assert x.dtype == torch.long
    assert y.dtype == torch.long