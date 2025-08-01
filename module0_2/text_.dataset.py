import torch
from torch.utils.data import Dataset, DataLoader

pad_id=63

class TextDataset(Dataset):
    def __init__(self, token_ids: list, context_length: int,stride):
    
        super().__init__()

        self.inputs = []
        self.targets = []

        for i in range(0, len(token_ids) - context_length, stride):
            input_chunk = token_ids[i:i + context_length]
            target_chunk = token_ids[i + 1:i + context_length + 1]

            #truncate if the last chunk is shorter than context_length
            input_chunk = input_chunk[:context_length]
            target_chunk = target_chunk[:context_length]
        
        # Pad the input and target chunks to ensure they are of the same length
        input_chunk += [pad_id] * (context_length - len(input_chunk))
        target_chunk += [pad_id] * (context_length - len(target_chunk))
        
        self.inputs.append(torch.tensor(input_chunk))
        self.targets.append(torch.tensor(target_chunk))

    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]