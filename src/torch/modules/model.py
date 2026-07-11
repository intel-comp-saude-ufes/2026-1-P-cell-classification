"""
Arquivo: src/torch/modules/model.py
Descrição:
    Este arquivo contém a classe com a implementação do modelo de classificação de células.
"""
from torch import nn
import torchvision.models as models

class CellClassifier(nn.Module):
    """
    Classe com a implementação do modelo de classificação de células.

    Args:
        nn (Module): Classe base do PyTorch para modelos de redes neurais.
    """
    def __init__(self):
        super().__init__()
        self.cnn_features = models.efficientnet_b3(weights='DEFAULT')
        
        num_features = self.cnn_features.classifier[1].in_features
        
        self.cnn_features.classifier = nn.Identity()

        self.fc = nn.Linear(num_features, 6)
    
    def forward(self, x):
        out = self.cnn_features(x)
        return self.fc(out)