"""
Arquivo: src/scripts/train_and_eval.py
Descrição:
    Este arquivo é responsável por treinar a avaliar o modelo de classificação de células.
    
    Ele deve utilizar todas as outras classes implementadas no projeto.
"""
import logging
import pandas as pd

from pathlib import Path
from datetime import datetime
from src.config.logging import setup_logging
from src.data.process_data import DataProcessing
from src.config.hyperparameters import Hyperparameters
from src.torch.utils.train_strategy import TrainingStrategy
from src.torch.utils.cross_validate import CrossValidation


# ----- Configs
METADATA_PATH = Path("data/raw/classifications.csv")
IMAGE_FOLDER_PATH = Path("data/raw/images")

setup_logging(level="INFO")
logger = logging.getLogger(__name__)
# -----


def main():
    logger.info("Iniciando script de treinamento e avaliação do modelo.")
    
    # Leitura e processamento dos dados
    logger.info(f"Lendo metadados do dataset em: {METADATA_PATH}")
    metadata_df = pd.read_csv(METADATA_PATH)
    data_processor = DataProcessing(metadata=metadata_df, image_folder_path=IMAGE_FOLDER_PATH, random_state=5)
    
    logger.info(f"Total de amostras no dataset: {len(data_processor)}")
    logger.info(f"Labels: {data_processor.get_labels()}")
    
    # Inicializando hiperparâmetros
    h_params = Hyperparameters(
        width=100,
        height=100,
        batch_size=32,
        learning_rate=0.0001,
        num_epochs=100,
        dropout=0.5,
        num_classes=6,
        num_workers=8
    )
    
    # ----- Treino único (fora do cross-validation)
    # iterfolds() é um gerador de splits (treino, validação); next() pega o
    # primeiro, dando um único split (~64% treino / ~16% validação, com o
    # restante reservado para teste). Assim treinamos uma vez, em vez das 5
    # execuções do cross_validate.
    train_data, val_data = next(data_processor.iterfolds())

    train_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs") / train_id

    training_strategy = TrainingStrategy(h_params, data_processor)
    result = training_strategy.train(
        train_data=train_data,
        val_data=val_data,
        output_dir=output_dir,
    )
    logger.info(
        f"Melhor modelo: F1={result['best_f1']:.4f} "
        f"na época {result['best_epoch']}. Artefatos salvos em {output_dir}"
    )
    # -----

    # # Inicialização do Cross Validation
    # cross_val = CrossValidation(data_processor=data_processor, k_folds=5)
    # cross_val.cross_validate(hyperparameters=h_params)
    
    # TODO: Pegar o melhor modelo obtido na validação cruzada e testar ele
    # TODO: A ideia é que o conjunto de teste seja seja salvo em `data_processor`
    #       para que ele seja usado aqui.

if __name__ == "__main__":
    main()