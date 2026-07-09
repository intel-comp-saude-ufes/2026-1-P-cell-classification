from torch.utils.data import Dataset

class CellClassificationDataset(Dataset):
    """
    Classe dataset do Pytorch para armazenar todos os nossos dados.

    Args:
        Dataset (Dataset): Classe base do PyTorch para datasets.
    """
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]