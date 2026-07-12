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
        width=90,
        height=90,
        batch_size=32,
        learning_rate=0.001,      # cabeça (fc): pesos aleatórios, aprende do zero
        backbone_lr=0.0001,       # backbone: pré-treinada, só ajuste fino
        # Backbone INTEIRA treinável. Congelar os blocos iniciais é a receita padrão
        # em imagens naturais, mas aqui ela se inverte: o que separa ASC-H de HSIL é
        # a granularidade da cromatina — textura de baixo/médio nível, justamente o
        # que esses blocos codificam. Congelá-los travou o F1 em 0,47 (contra 0,56
        # com tudo treinável), independentemente do learning rate.
        trainable_blocks=None,
        freeze_epochs=2,          # warm-up: 2 épocas treinando só a cabeça
        num_epochs=100,
        # Precisa ser folgada o bastante para o ReduceLROnPlateau (patience=3) ter
        # 2 ou 3 chances de destravar o platô antes do early stopping desistir.
        patience=10,
        dropout=0.5,
        num_workers=10,
        balance_strategy="sampler_sqrt",
        label_smoothing=0.1,      # contra o excesso de confiança (val_loss subindo)
        weight_decay=0.05,        # acima do default do AdamW (0.01)
    )

    # ----- Treino único (fora do cross-validation)
    # iterfolds() é um gerador de splits (treino, validação); next() pega o
    # primeiro, dando um único split (~64% treino / ~16% validação, com o
    # restante reservado para teste). Assim treinamos uma vez, em vez das 5
    # execuções do cross_validate.
    #
    # O split é calculado UMA vez e reaproveitado pelas três tarefas. Como
    # iterfolds() estratifica sempre por `bethesda_system`, ele não depende do
    # espaço de rótulos — então os três modelos veem exatamente as mesmas lâminas
    # em treino e em validação, e os três resultados são comparáveis entre si.
    train_data, val_data = next(data_processor.iterfolds())

    train_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # As três granularidades da mesma tarefa. Não são estágios encadeados: cada
    # modelo é independente, treinado sobre TODAS as células, mudando só o alvo.
    tarefas = {
        "6_classes": data_processor.flat_label_space(),
        "3_classes": data_processor.grade_label_space(),
        "2_classes": data_processor.binary_label_space(),
    }

    resultados = {}
    for nome, label_space in tarefas.items():
        logger.info(f"--- Treinando tarefa '{nome}': {label_space.names}")

        output_dir = Path("outputs") / train_id / nome

        training_strategy = TrainingStrategy(h_params, data_processor, label_space)
        result = training_strategy.train(
            train_data=train_data,
            val_data=val_data,
            output_dir=output_dir,
        )
        resultados[nome] = result
    # -----

    # # Inicialização do Cross Validation
    # cross_val = CrossValidation(data_processor=data_processor, k_folds=5)
    # cross_val.cross_validate(hyperparameters=h_params)
    
    # TODO: Pegar o melhor modelo obtido na validação cruzada e testar ele
    # TODO: A ideia é que o conjunto de teste seja seja salvo em `data_processor`
    #       para que ele seja usado aqui.

if __name__ == "__main__":
    main()