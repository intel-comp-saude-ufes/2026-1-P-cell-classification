"""
Arquivo: src/config/hyperparameters.py
Descrição:
    Este arquivo contém uma dataclass para armazenar hiperparâmetros que serão
    passados para o modelo a ser treinado.
"""
from dataclasses import dataclass

@dataclass
class Hyperparameters:
    """
    Classe para armazenar os hiperparâmetros do modelo.

    Args:
        learning_rate (float): Taxa de aprendizado.
        batch_size (int): Tamanho do lote.
        num_epochs (int): Número de épocas.
        optimizer (str): Otimizador a ser usado.
        loss_function (str): Função de perda a ser usada.
    """
    learning_rate: float
    batch_size: int
    num_epochs: int
    optimizer: str
    loss_function: str