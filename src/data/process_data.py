from dataclass import dataclass

@dataclass
class ImageCellDataset:
    image_paths: list
    label: list

class DataProcessing:
    """
    Classe responsável pelo processamento de dados.

    Args:
        None
    """
    def __init__(self):
        pass