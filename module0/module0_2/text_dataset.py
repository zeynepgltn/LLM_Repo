import torch
from torch.utils.data import Dataset, DataLoader

pad_id = 63

import torch
from torch.utils.data import Dataset

pad_id = 63

class TextDataset(Dataset):
    def __init__(self, token_ids: list, context_length: int, stride: int):
        super().__init__()
        self.inputs = []
        self.targets = []

        for i in range(0, len(token_ids) - context_length, stride):
            input_chunk = token_ids[i:i + context_length]
            target_chunk = token_ids[i + 1:i + context_length + 1]

            # pad if needed
            input_chunk += [pad_id] * (context_length - len(input_chunk))
            target_chunk += [pad_id] * (context_length - len(target_chunk))

            self.inputs.append(torch.tensor(input_chunk))
            self.targets.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# Düzeltilmiş DataLoader fonksiyonu
def create_data_loader(token_ids: list, context_length: int, stride: int,
                       batch_size: int, shuffle: bool = True, device: str = "cpu"):
    dataset = TextDataset(token_ids, context_length, stride)
    
    # Generator düzeltmesi
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(42)  # Reproducibility için
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        pin_memory=(device != "cpu"),  # GPU kullanıyorsanız performance için
        num_workers=0  # Multiprocessing sorunlarını önlemek için
    )
    return data_loader

# Test kodu
if __name__ == "__main__":
    # Örnek token_ids
    token_ids = list(range(100))  # 0-99 arası sayılar
    context_length = 8
    stride = 4
    batch_size = 2
    
    # DataLoader oluştur
    loader = create_data_loader(token_ids, context_length, stride, batch_size)
    
    # Test et
    for i, (inp, tgt) in enumerate(loader):
        print(f"Batch {i+1}:")
        print(f"Input: {inp}")
        print(f"Target: {tgt}")
        if i > 2:  # Sadece ilk 3 batch'i göster
            break