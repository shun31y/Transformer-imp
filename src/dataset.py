import torch

class CharDataset(torch.utils.data.Dataset):
    def __init__(self, token_ids: list[int], block_size: int):
        self.token_ids = token_ids
        self.block_size = block_size
            
    def __len__(self):
        return len(self.token_ids) - self.block_size
    def __getitem__(self, idx):
        x = self.token_ids[idx:idx+self.block_size]
        y = self.token_ids[idx+1:idx+1+self.block_size]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
            