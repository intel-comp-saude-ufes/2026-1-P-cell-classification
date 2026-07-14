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
from src.torch.utils.evaluate import Evaluator
from src.torch.utils.cross_validate import CrossValidation


# ----- Configs
METADATA_PATH = Path("data/raw/classifications.csv")
IMAGE_FOLDER_PATH = Path("data/raw/images")

RANDOM_STATE = 5
TRAIN_SEED = 42
K_FOLDS = 5

setup_logging(level="INFO")
logger = logging.getLogger(__name__)
# -----


def main():
    logger.info("Iniciando script de treinamento e avaliação do modelo.")
    
    # Leitura e processamento dos dados
    logger.info(f"Lendo metadados do dataset em: {METADATA_PATH}")
    metadata_df = pd.read_csv(METADATA_PATH)
    data_processor = DataProcessing(
        metadata=metadata_df,
        image_folder_path=IMAGE_FOLDER_PATH,
        random_state=RANDOM_STATE,
    )
    
    logger.info(f"Total de amostras no dataset: {len(data_processor)}")
    logger.info(f"Labels: {data_processor.get_labels()}")
    
    # Inicializando hiperparâmetros
    h_params = Hyperparameters(
        width=90,
        height=90,
        batch_size=32,
        learning_rate=0.001,
        backbone_lr=0.0001,
        trainable_blocks=None,
        freeze_epochs=2,
        num_epochs=100,
        patience=10,
        dropout=0.5,
        num_workers=10,
        balance_strategy="sampler_sqrt",
        label_smoothing=0.1,
        weight_decay=0.05,
    )

    train_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Definindo as tarefas
    tarefas = {
        "6_classes": data_processor.flat_label_space(),
        "3_classes": data_processor.grade_label_space(),
        "2_classes": data_processor.binary_label_space(),
    }

    # O conjunto de teste é o mesmo para as três tarefas 
    test_data = data_processor.get_test_data()
    logger.info(f"Conjunto de teste (intocado): {len(test_data)} células")

    resultados = {}
    for nome, label_space in tarefas.items():
        logger.info(f"===== Tarefa '{nome}': {label_space.names}")

        output_dir = Path("outputs") / train_id / nome

        # 1) Validação cruzada
        cross_val = CrossValidation(
            data_processor=data_processor,
            k_folds=K_FOLDS,
            label_space=label_space,
        )
        cv = cross_val.cross_validate(
            hyperparameters=h_params,
            seed=TRAIN_SEED,
            output_dir=output_dir,
        )

        # 2) Avaliando no conjunto de teste
        logger.info(f"--- Avaliando '{nome}' no conjunto de teste")
        evaluator = Evaluator(data_processor=data_processor, label_space=label_space)
        teste = evaluator.evaluate(
            checkpoints=cv["checkpoints"],
            data=test_data,
            hyperparameters=h_params,
            output_dir=output_dir,
        )

        resultados[nome] = {"cv": cv, "test": teste}

    logger.info("===== Resumo final (F1-macro) =====")
    for nome, r in resultados.items():
        logger.info(
            f"  {nome:<10} "
            f"validação cruzada: {r['cv']['aggregate']['f1_macro']['mean']:.4f} "
            f"± {r['cv']['aggregate']['f1_macro']['std']:.4f}  |  "
            f"teste: {r['test']['aggregate']['f1_macro']['mean']:.4f} "
            f"± {r['test']['aggregate']['f1_macro']['std']:.4f}  |  "
            f"teste (ensemble): {r['test']['ensemble']['f1_macro']:.4f}"
        )


if __name__ == "__main__":
    main()