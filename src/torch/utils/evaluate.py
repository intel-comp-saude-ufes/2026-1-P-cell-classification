"""
Arquivo: src/torch/utils/evaluate.py
Descrição:
    Este arquivo contém a classe que irá avaliar o nosso modelo no conjunto de teste.
"""
from data.process_data import DataProcessing


class Evaluator():
    def __init__(self, data_processor: DataProcessing):
        self.data_processor = data_processor

    def evaluate(self):
        pass