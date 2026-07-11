"""
Arquivo: src/torch/modules/dataset.py
Descrição:
    Este arquivo contém a classe dataset do Pytorch para armazenar todos os nossos dados.
"""
from PIL import Image
from typing import Callable
from torch.utils.data import Dataset
import torchvision.transforms

from src.data.process_data import Cell, DataProcessing


class CellClassificationDataset(Dataset):
    """
    Classe dataset do Pytorch para armazenar todos os nossos dados.

    Args:
        Dataset (Dataset): Classe base do PyTorch para datasets.
    """
    def __init__(self, data: list[Cell], data_processor: DataProcessing, width, height, transform: Callable | None = None):
        self.data = data
        self.data_processor = data_processor
        
        if transform is not None:
            self.transform = transform
        else:
            self.transform = torchvision.transforms.ToTensor()


        # Hiperparâmetros
        self.width = width
        self.height = height

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        cell_info = self.data[idx]
        
        horizontal = self.width // 2
        vertical = self.height // 2
        
        image = Image.open(cell_info.image_path).convert("RGB")
        image = image.crop((cell_info.x - horizontal, 
                            cell_info.y - vertical, 
                            cell_info.x + horizontal, 
                            cell_info.y + vertical))
        
        if self.transform:
            image = self.transform(image)
        
        label = self.data_processor.label2index(cell_info.label)
        
        return image, label