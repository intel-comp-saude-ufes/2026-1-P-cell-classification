"""
Arquivo: src/torch/utils/cross_validate.py
Descrição:
    Este arquivo contém a classe que irá realizar a validação cruzada nos dados.
    
    O objetivo é que a classe de tratamento dos dados entregue os dados e essa classe apenas
    utilize eles.
"""
from tqdm import tqdm

from src.data.process_data import DataProcessing
from src.torch.utils.train_strategy import TrainingStrategy


class CrossValidation():
    """
    Classe para fazer validação cruzada nos dados.
    """
    def __init__(self, data_processor: DataProcessing, k_folds=5):
        self.data_processor = data_processor
        self.k_folds = k_folds
    
    def cross_validate(self, hyperparameters):
        """
        Roda validação cruzada para o nosso treinamento.

        Args:
            hyperparameters (Hyperparameters): Hiperparâmetros do treinamento.
        """
        with tqdm(total=self.k_folds, desc="Cross Validation") as pbar:
            for fold, (train_data, val_data) in enumerate(
                self.data_processor.iterfolds(),
                start=1,
            ):
                training_strategy = TrainingStrategy(
                    hyperparameters=hyperparameters,
                    data_processor=self.data_processor,
                )

                training_strategy.train(
                    train_data=train_data,
                    val_data=val_data,
                )
                
                # TODO: Resultados do treino ainda deverão ser tratados aqui

                pbar.update()