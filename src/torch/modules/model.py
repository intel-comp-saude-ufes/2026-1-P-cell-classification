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
    def __init__(self, dropout, num_classes):
        super().__init__()
        self.cnn_features = models.efficientnet_b3(weights='DEFAULT')

        num_features = self.cnn_features.classifier[1].in_features

        self.cnn_features.classifier = nn.Identity()

        self.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, num_classes),
        )

        # Blocos da backbone com os pesos congelados. Preenchido por
        # set_backbone_trainable(); usado pelo train() abaixo.
        self._frozen_blocks = []

    def set_backbone_trainable(self, trainable_blocks=None):
        """
        Define quantos blocos FINAIS da backbone continuam treináveis. Os blocos
        iniciais aprendem bordas e texturas genéricas, que transferem bem de
        qualquer domínio; os finais são os semanticamente ligados à ImageNet, e
        são os que precisam se readaptar à citologia. Congelar os iniciais corta
        parâmetros treináveis sem sacrificar a adaptação de domínio.

        Args:
            trainable_blocks (int | None): nº de blocos finais treináveis.
                None treina a backbone inteira; 0 congela tudo (linear probe).
        """
        blocks = self.cnn_features.features

        if trainable_blocks is None:
            cut = 0  # nada congelado
        else:
            cut = max(0, min(len(blocks) - trainable_blocks, len(blocks)))

        # Blocos [0, cut) ficam congelados; [cut, fim) permanecem treináveis.
        self._frozen_blocks = [blocks[i] for i in range(cut)]

        for i, block in enumerate(blocks):
            for param in block.parameters():
                param.requires_grad = i >= cut

    def train(self, mode=True):
        """
        Sobrescreve o train() do PyTorch para manter os blocos congelados em
        eval(). Sem isso, requires_grad=False impediria a atualização dos pesos,
        mas o BatchNorm continuaria atualizando running_mean/running_var a cada
        batch — a backbone "congelada" ainda derivaria com os dados.
        """
        super().train(mode)

        if mode:
            for block in self._frozen_blocks:
                block.eval()

        return self

    def forward(self, x):
        out = self.cnn_features(x)
        return self.fc(out)