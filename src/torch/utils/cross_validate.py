"""
Arquivo: src/torch/utils/cross_validate.py
Descrição:
    Este arquivo contém a classe que irá realizar a validação cruzada nos dados.

    O objetivo é que a classe de tratamento dos dados entregue os dados e essa classe apenas
    utilize eles.
"""
import logging
import statistics
from pathlib import Path
from datetime import datetime

from tqdm import tqdm

from src.data.process_data import DataProcessing, LabelSpace
from src.torch.utils.train_strategy import TrainingStrategy

logger = logging.getLogger(__name__)


class CrossValidation():
    """
    Validação cruzada k-fold.

    Args:
        data_processor (DataProcessing): fonte dos dados e dos splits.
        k_folds (int): número de folds.
        label_space (LabelSpace | None): a tarefa (6, 3 ou 2 classes). Sem ele, as
            6 classes originais.
    """
    def __init__(self, data_processor: DataProcessing, k_folds=5,
                 label_space: LabelSpace | None = None):
        self.data_processor = data_processor
        self.k_folds = k_folds
        self.label_space = label_space or data_processor.flat_label_space()

    def cross_validate(self, hyperparameters, seed=42, output_dir=None):
        """
        Roda os k folds e agrega os resultados.

        Args:
            hyperparameters (Hyperparameters): hiperparâmetros do treinamento.
            seed (int): semente base. Cada fold usa `seed + fold`, então os folds
                são treinos distintos entre si, mas a execução inteira é reprodutível.
            output_dir (Path | None): pasta dos artefatos. Se None, cria uma nova
                em outputs/<timestamp>.

        Returns:
            dict: métricas por fold e o agregado (média e desvio).
        """
        if output_dir is None:
            output_dir = Path('outputs') / datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(output_dir)

        class_names = self.label_space.names
        folds = []
        checkpoints = []

        with tqdm(total=self.k_folds, desc='Cross Validation') as pbar:
            for fold, (train_data, val_data) in enumerate(
                self.data_processor.iterfolds(k_folds=self.k_folds),
                start=1,
            ):
                training_strategy = TrainingStrategy(
                    hyperparameters=hyperparameters,
                    data_processor=self.data_processor,
                    label_space=self.label_space,
                )

                fold_dir = output_dir / f'fold_{fold}'

                result = training_strategy.train(
                    train_data=train_data,
                    val_data=val_data,
                    output_dir=fold_dir,
                    seed=seed + fold,
                )

                checkpoints.append(fold_dir / 'best_model.pt')

                # As métricas da MELHOR época deste fold — as mesmas que o early
                # stopping usou para selecionar os pesos. best_epoch é 1-indexado.
                melhor = result['history'][result['best_epoch'] - 1]
                folds.append({
                    'fold': fold,
                    'best_epoch': result['best_epoch'],
                    'f1_macro': melhor['f1_score'],
                    'precision_macro': melhor['precision'],
                    'recall_macro': melhor['recall'],
                    'accuracy': melhor['accuracy'],
                    'per_class': melhor['per_class'],
                })

                logger.info(
                    f'Fold {fold}/{self.k_folds}: F1-macro={melhor["f1_score"]:.4f} '
                    f'(época {result["best_epoch"]})'
                )

                pbar.update()

        agregado = self._aggregate(folds, class_names)
        self._save_summary(folds, agregado, class_names, output_dir)

        logger.info(
            f'Validação cruzada ({self.k_folds} folds): '
            f'F1-macro = {agregado["f1_macro"]["mean"]:.4f} '
            f'± {agregado["f1_macro"]["std"]:.4f}. '
            f'Resumo em {output_dir / "cross_validation.txt"}'
        )

        return {
            'folds': folds,
            'aggregate': agregado,
            'checkpoints': checkpoints,
            'output_dir': str(output_dir),
        }

    def _aggregate(self, folds, class_names):
        """
        Média e desvio padrão entre os folds, no agregado e por classe.

        O desvio POR CLASSE é o mais informativo: ele mostra quais classes são
        instáveis. As raras terão desvio grande — e são elas que puxam a variância
        do F1-macro, já que ele as pesa igual às comuns.
        """
        def mean_std(valores):
            return {
                'mean': statistics.mean(valores),
                'std': statistics.stdev(valores) if len(valores) > 1 else 0.0,
            }

        agregado = {
            metrica: mean_std([f[metrica] for f in folds])
            for metrica in ('f1_macro', 'precision_macro', 'recall_macro', 'accuracy')
        }

        agregado['per_class'] = {
            nome: {
                **mean_std([f['per_class'][nome]['f1_score'] for f in folds]),
                # O suporte varia pouco entre folds; a média basta para dar a escala.
                'support': statistics.mean(
                    [f['per_class'][nome]['support'] for f in folds]
                ),
            }
            for nome in class_names
        }

        return agregado

    def _save_summary(self, folds, agregado, class_names, output_dir):
        """Escreve o resumo da validação cruzada em cross_validation.txt."""
        output_dir.mkdir(parents=True, exist_ok=True)

        linhas = [
            f'Validação cruzada — {len(folds)} folds',
            f'Tarefa: {len(class_names)} classes {class_names}',
            '',
            'Por fold:',
            f'  {"fold":<6} {"época":<7} {"F1-macro":<10} {"Precision":<10} '
            f'{"Recall":<10} {"Acurácia":<10}',
        ]
        linhas += [
            f'  {f["fold"]:<6} {f["best_epoch"]:<7} {f["f1_macro"]:<10.4f} '
            f'{f["precision_macro"]:<10.4f} {f["recall_macro"]:<10.4f} '
            f'{f["accuracy"]:<10.4f}'
            for f in folds
        ]

        linhas += [
            '',
            'Agregado (média ± desvio entre folds):',
            f'  F1-macro        : {agregado["f1_macro"]["mean"]:.4f} ± {agregado["f1_macro"]["std"]:.4f}',
            f'  Precision-macro : {agregado["precision_macro"]["mean"]:.4f} ± {agregado["precision_macro"]["std"]:.4f}',
            f'  Recall-macro    : {agregado["recall_macro"]["mean"]:.4f} ± {agregado["recall_macro"]["std"]:.4f}',
            f'  Acurácia        : {agregado["accuracy"]["mean"]:.4f} ± {agregado["accuracy"]["std"]:.4f}',
            '',
            'F1 por classe (média ± desvio). Um desvio grande numa classe de suporte',
            'pequeno significa que o F1-macro daquele fold foi decidido no cara-ou-coroa:',
            f'  {"classe":<38} {"F1":<8} {"±":<8} {"suporte":<8}',
        ]
        linhas += [
            f'  {nome:<38} '
            f'{agregado["per_class"][nome]["mean"]:<8.4f} '
            f'{agregado["per_class"][nome]["std"]:<8.4f} '
            f'{agregado["per_class"][nome]["support"]:<8.0f}'
            for nome in class_names
        ]

        (output_dir / 'cross_validation.txt').write_text('\n'.join(linhas) + '\n')
