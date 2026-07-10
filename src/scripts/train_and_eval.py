"""
Arquivo: src/scripts/train_and_eval.py
Descrição:
    Este arquivo é responsável por treinar a avaliar o modelo de classificação de células.
    
    Ele deve utilizar todas as outras classes implementadas no projeto.
"""
import logging
import pandas as pd

from pathlib import Path
from src.config.logging import setup_logging
from src.data.process_data import DataProcessing
from src.config.hyperparameters import Hyperparameters
from src.torch.utils.cross_validate import CrossValidation


# ----- Configs
METADATA_PATH = Path("data/raw/classifications.csv")

setup_logging(level="INFO")
logger = logging.getLogger(__name__)
# -----


def main():
    logger.info("Iniciando script de treinamento e avaliação do modelo.")
    
    # Leitura e processamento dos dados
    logger.info(f"Lendo metadados do dataset em: {METADATA_PATH}")
    metadata_df = pd.read_csv(METADATA_PATH)
    data_processor = DataProcessing(metadata=metadata_df)
    
    logger.info(f"Total de amostras no dataset: {len(data_processor)}")
    logger.info(f"Labels: {data_processor.get_labels()}")
    
    # Inicializando hiperparâmetros
    h_params = Hyperparameters(
        width=150,
        height=150,
        batch_size=32,
        learning_rate=0.001,
        num_epochs=10
    )
    
    # Inicialização do Cross Validation
    cross_val = CrossValidation(data_processor=data_processor, k_folds=5)
    cross_val.cross_validate(hyperparameters=h_params)
    
    # TODO: Pegar o melhor modelo obtido na validação cruzada e testar ele
    # TODO: A ideia é que o conjunto de teste seja seja salvo em `data_processor`
    #       para que ele seja usado aqui.

if __name__ == "__main__":
    main()