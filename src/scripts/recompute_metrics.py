"""
Arquivo: src/scripts/recompute_metrics.py
Descrição:
    Reconstrói a tabela completa de métricas (F1-macro, precisão, recall e
    acurácia) de uma run JÁ TREINADA, nas três colunas que se reporta: validação
    cruzada, modelos individuais no teste e ensemble.

    Nenhum modelo é retreinado. Os pesos da melhor época de cada fold estão em
    best_model.pt, e o split é reconstruído com a mesma semente — então tudo o que
    falta é inferência. O que era caro (o treino) já foi pago.

    Uso:
        uv run python -m src.scripts.recompute_metrics outputs/20260712_005136
"""
import argparse
import logging
from pathlib import Path

import pandas as pd
import torch

from src.config.hyperparameters import Hyperparameters
from src.config.logging import setup_logging
from src.data.process_data import DataProcessing
from src.torch.modules.model import load_from_checkpoint
from src.torch.utils.evaluate import METRICAS, Evaluator, aggregate_metrics
from src.scripts.train_and_eval import (
    IMAGE_FOLDER_PATH,
    K_FOLDS,
    METADATA_PATH,
    RANDOM_STATE,
)

setup_logging(level="INFO")
logger = logging.getLogger(__name__)

# Só afetam a velocidade da inferência, não o resultado.
BATCH_SIZE = 32
NUM_WORKERS = 10

# Tolerância ao conferir o F1 recalculado contra o que o checkpoint registrou.
# Não é zero porque a inferência roda em mixed precision, e a mesma soma em
# float16 pode diferir na última casa. Qualquer diferença acima disto não é ruído
# numérico: é o split ter sido reconstruído errado.
TOLERANCIA = 1e-3


def _inference_hyperparameters(checkpoint_path):
    """
    Hiperparâmetros de inferência, lidos do PRÓPRIO checkpoint.

    O recorte ao redor do núcleo (width x height) precisa ser o mesmo do treino,
    senão o modelo recebe uma imagem que não é a que ele aprendeu a ler. Em vez de
    duplicar esses valores aqui — onde eles poderiam divergir da run em silêncio —
    lemos do checkpoint, que os guardou. Os demais campos são exigidos pela
    dataclass mas só o treino usa.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    return Hyperparameters(
        width=checkpoint['width'],
        height=checkpoint['height'],
        dropout=checkpoint['dropout'],
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        learning_rate=0.0,
        num_epochs=0,
    )


def _checkpoints(task_dir):
    """Os k best_model.pt de uma tarefa, na ordem dos folds."""
    caminhos = [
        task_dir / f'fold_{fold}' / 'best_model.pt'
        for fold in range(1, K_FOLDS + 1)
    ]

    faltando = [c for c in caminhos if not c.exists()]
    if faltando:
        raise FileNotFoundError(
            f'Checkpoints ausentes em {task_dir}: '
            f'{", ".join(str(c) for c in faltando)}'
        )

    return caminhos


def _cross_validation(evaluator, data_processor, checkpoints, hyperparameters):
    """
    Refaz a validação cruzada por inferência: cada fold é medido pelo modelo que
    foi treinado nos outros quatro, exatamente como no treino original.

    A verificação no meio é o que torna isto confiável. O checkpoint guarda o
    `best_f1` — o F1-macro que o modelo obteve NESTE mesmo conjunto de validação,
    na época em que o early stopping o selecionou. Se o número que recalculamos
    agora não bate com aquele, o fold foi reconstruído errado (uma semente de
    split diferente, por exemplo), e estaríamos medindo o modelo em células que
    ele viu no treino — um resultado otimista e silenciosamente falso. Melhor
    quebrar aqui do que reportar isso numa tabela.
    """
    por_fold = []

    for fold, (_, val_data) in enumerate(
        data_processor.iterfolds(k_folds=K_FOLDS), start=1
    ):
        checkpoint = checkpoints[fold - 1]

        resultado = evaluator.evaluate(
            checkpoints=[checkpoint],
            data=val_data,
            hyperparameters=hyperparameters,
            # Sem output_dir: nada de sobrescrever os artefatos da run original.
            output_dir=None,
        )
        metricas = resultado['per_model'][0]

        registrado = torch.load(
            checkpoint, map_location='cpu', weights_only=False
        )['best_f1']

        if abs(metricas['f1_macro'] - registrado) > TOLERANCIA:
            raise RuntimeError(
                f'Fold {fold}: o F1-macro recalculado ({metricas["f1_macro"]:.4f}) '
                f'não bate com o do checkpoint ({registrado:.4f}). O conjunto de '
                f'validação reconstruído não é o mesmo do treino — confira se o '
                f'RANDOM_STATE ({RANDOM_STATE}) e o K_FOLDS ({K_FOLDS}) são os '
                f'mesmos da run.'
            )

        logger.info(
            f'  fold {fold}: F1-macro={metricas["f1_macro"]:.4f} '
            f'(confere com o checkpoint), acurácia={metricas["accuracy"]:.4f}'
        )
        por_fold.append(metricas)

    return por_fold, aggregate_metrics(por_fold)


def _write_table(task_dir, evaluator, cv_por_fold, cv_agregado, teste):
    """Escreve metrics_table.txt: a tabela pronta para o artigo, e o detalhe dela."""
    class_names = evaluator.label_space.names
    metricas = METRICAS

    def media_desvio(agregado, chave):
        return f'{agregado[chave]["mean"]:.4f} ± {agregado[chave]["std"]:.4f}'

    linhas = [
        f'Tabela de métricas — tarefa de {len(class_names)} classes {class_names}',
        '',
        'Reconstruída dos checkpoints da run: nenhum modelo foi retreinado, só',
        'reavaliado. Os números da validação cruzada conferem com o best_f1 que',
        'cada checkpoint registrou, o que confirma que o split é o mesmo do treino.',
        '',
        f'  {"Métrica":<12} {"Validação cruzada":<22} '
        f'{"Modelos individuais":<22} {"Ensemble":<12}',
        f'  {"":<12} {"(média ± desvio)":<22} '
        f'{"no teste (média ± σ)":<22} {"(teste final)":<12}',
    ]
    linhas += [
        f'  {rotulo:<12} {media_desvio(cv_agregado, chave):<22} '
        f'{media_desvio(teste["aggregate"], chave):<22} '
        f'{teste["ensemble"][chave]:<12.4f}'
        for chave, rotulo in metricas
    ]

    rotulos = [rotulo for _, rotulo in metricas]
    chaves = [chave for chave, _ in metricas]

    linhas += [
        '',
        'Validação cruzada, por fold (cada modelo no seu conjunto de validação):',
        f'  {"fold":<8} ' + ' '.join(f'{r:<10}' for r in rotulos),
    ]
    linhas += [
        f'  {fold:<8} ' + ' '.join(f'{m[c]:<10.4f}' for c in chaves)
        for fold, m in enumerate(cv_por_fold, start=1)
    ]

    linhas += [
        '',
        'Teste, por modelo (os k modelos no MESMO conjunto de teste — aqui o desvio',
        'mede só a instabilidade do treino, já que os dados não mudam):',
        f'  {"modelo":<8} ' + ' '.join(f'{r:<10}' for r in rotulos),
    ]
    linhas += [
        f'  {i:<8} ' + ' '.join(f'{m[c]:<10.4f}' for c in chaves)
        for i, m in enumerate(teste['per_model'], start=1)
    ]

    caminho = task_dir / 'metrics_table.txt'
    caminho.write_text('\n'.join(linhas) + '\n')

    return caminho


def main():
    parser = argparse.ArgumentParser(
        description='Recalcula a tabela de métricas de uma run já treinada, '
                    'a partir dos checkpoints salvos.',
    )
    parser.add_argument(
        'run_dir',
        type=Path,
        help='pasta da run, ex.: outputs/20260712_005136',
    )
    args = parser.parse_args()

    metadata_df = pd.read_csv(METADATA_PATH)
    data_processor = DataProcessing(
        metadata=metadata_df,
        image_folder_path=IMAGE_FOLDER_PATH,
        random_state=RANDOM_STATE,
    )
    test_data = data_processor.get_test_data()

    tarefas = sorted(
        d for d in args.run_dir.iterdir()
        if d.is_dir() and (d / 'fold_1' / 'best_model.pt').exists()
    )
    if not tarefas:
        raise SystemExit(f'Nenhuma tarefa com checkpoints em {args.run_dir}')

    logger.info(
        f'Run {args.run_dir}: {len(tarefas)} tarefa(s), '
        f'{len(test_data)} células de teste.'
    )

    for task_dir in tarefas:
        checkpoints = _checkpoints(task_dir)

        # A tarefa vem do checkpoint, não do nome da pasta: ele é a única fonte
        # que sabe o que o índice 2 de uma predição significa.
        _, label_space = load_from_checkpoint(checkpoints[0])
        evaluator = Evaluator(data_processor=data_processor, label_space=label_space)
        hyperparameters = _inference_hyperparameters(checkpoints[0])

        logger.info(f"===== Tarefa '{task_dir.name}': {label_space.names}")

        logger.info('--- Validação cruzada (reavaliando cada fold)')
        cv_por_fold, cv_agregado = _cross_validation(
            evaluator, data_processor, checkpoints, hyperparameters
        )

        logger.info('--- Conjunto de teste (modelos individuais + ensemble)')
        teste = evaluator.evaluate(
            checkpoints=checkpoints,
            data=test_data,
            hyperparameters=hyperparameters,
            output_dir=None,
        )

        caminho = _write_table(
            task_dir, evaluator, cv_por_fold, cv_agregado, teste
        )
        logger.info(f'Tabela salva em {caminho}')


if __name__ == '__main__':
    main()
