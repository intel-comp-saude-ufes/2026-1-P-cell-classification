"""
Arquivo: src/scripts/train_and_eval.py
Descrição:
    Este arquivo é responsável por treinar a avaliar o modelo de classificação de células.
    
    Ele deve utilizar todas as outras classes implementadas no projeto.
"""
import logging
import pandas as pd

from pathlib import Path
from src.data.process_data import DataProcessing
from src.config.logging import setup_logging


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

if __name__ == "__main__":
    main()