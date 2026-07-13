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

# Duas sementes, dois eixos independentes. RANDOM_STATE semeia o SPLIT (quais
# lâminas caem no teste e como os folds se formam); TRAIN_SEED semeia o TREINO
# (inicialização da cabeça, dropout, augmentation, sampler), e cada fold usa
# TRAIN_SEED + fold.
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

    train_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # As três granularidades da mesma tarefa. Não são estágios encadeados: cada
    # modelo é independente, treinado sobre TODAS as células, mudando só o alvo.
    #
    # Os splits são idênticos entre elas: iterfolds() estratifica sempre por
    # `bethesda_system`, independente do espaço de rótulos, então os três modelos
    # veem exatamente as mesmas lâminas em cada fold — e os três resultados são
    # comparáveis no mesmo conjunto de teste.
    tarefas = {
        "6_classes": data_processor.flat_label_space(),
        "3_classes": data_processor.grade_label_space(),
        "2_classes": data_processor.binary_label_space(),
    }

    # O conjunto de teste é o mesmo para as três tarefas — separado por iterfolds()
    # antes de qualquer fold, e nunca visto durante o treino nem a seleção.
    test_data = data_processor.get_test_data()
    logger.info(f"Conjunto de teste (intocado): {len(test_data)} células")

    resultados = {}
    for nome, label_space in tarefas.items():
        logger.info(f"===== Tarefa '{nome}': {label_space.names}")

        output_dir = Path("outputs") / train_id / nome

        # 1) Validação cruzada: estima o desempenho COM BARRA DE ERRO. Um treino
        #    único daria um número sem desvio, e não dá para saber se a diferença
        #    entre duas configurações é real ou é ruído — sobretudo aqui, onde o
        #    F1-macro pesa igual uma classe de ~28 amostras e uma de ~1143.
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

        # 2) Teste final. Medido UMA vez, no fim, com a configuração já decidida —
        #    se o teste for consultado a cada ajuste, vira um segundo conjunto de
        #    validação e a estimativa final fica otimista.
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