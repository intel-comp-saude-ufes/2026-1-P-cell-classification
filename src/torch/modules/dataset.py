"""
Arquivo: src/torch/modules/dataset.py
Descrição:
    Este arquivo contém a classe dataset do Pytorch para armazenar todos os nossos dados.
"""
from PIL import Image
from typing import Callable
from torch.utils.data import Dataset

from src.data.process_data import Cell


class CellClassificationDataset(Dataset):
    """
    Classe dataset do Pytorch para armazenar todos os nossos dados.

    Args:
        Dataset (Dataset): Classe base do PyTorch para datasets.
    """
    def __init__(self, data: list[Cell], width, height, transform: Callable | None = None):
        self.data = data
        self.transform = transform
        
        # Hiperparâmetros
        self.width = width
        self.height = height

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        cell_info = self.data[idx]
        
        image = Image.open(cell_info.image_path).convert("RGB")
        image = image.crop((cell_info.x, cell_info.y, cell_info.x + self.width, cell_info.y + self.height))
        
        if self.transform:
            image = self.transform(image)
        
        label = self.data_processer.label2index(cell_info.label)
        
        return image, label