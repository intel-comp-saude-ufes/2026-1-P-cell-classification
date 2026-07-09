"""
Arquivo: src/torch/modules/model.py
Descrição:
    Este arquivo contém a classe com a implementação do modelo de classificação de células.
"""
from torch import nn

class CellClassifier(nn.Module):
    """
    Classe com a implementação do modelo de classificação de células.

    Args:
        nn (Module): Classe base do PyTorch para modelos de redes neurais.
    """
    def __init__(self):
        super().__init__()
    
    # TODO
    def forward(self, x):
        pass