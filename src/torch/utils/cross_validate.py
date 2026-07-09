"""
Arquivo: src/torch/utils/cross_validate.py
Descrição:
    Este arquivo contém a classe que irá realizar a validação cruzada nos dados.
    
    O objetivo é que a classe de tratamento dos dados entregue os dados e essa classe apenas
    utilize eles.
"""
from data.process_data import DataProcessing


class CrossValidation():
    """
    Classe para fazer validação cruzada nos dados.
    """
    def __init__(self, data_processer: DataProcessing, k_folds=5):
        self.data_processer = data_processer
        self.k_folds = k_folds