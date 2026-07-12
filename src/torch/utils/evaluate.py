"""
Arquivo: src/torch/utils/evaluate.py
Descrição:
    Este arquivo contém a classe que irá avaliar o nosso modelo no conjunto de teste.
"""
import logging
import statistics
from pathlib import Path

import torch
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

from src.data.process_data import DataProcessing, LabelSpace
from src.torch.modules.dataset import CellClassificationDataset
from src.torch.modules.model import load_from_checkpoint
from src.torch.utils.train_strategy import build_eval_transform

logger = logging.getLogger(__name__)


class Evaluator():
    """
    Avalia modelos no conjunto de teste — o que ficou reservado por
    DataProcessing e nunca foi tocado durante o treino nem a seleção.

    O teste é medido UMA vez, no fim, depois que a configuração já está decidida.
    Se ele for consultado a cada ajuste de hiperparâmetro, deixa de ser teste e
    vira um segundo conjunto de validação, e a estimativa final fica otimista.

    Args:
        data_processor (DataProcessing): fonte dos dados.
        label_space (LabelSpace | None): a tarefa. Sem ele, as 6 classes originais.
    """
    def __init__(self, data_processor: DataProcessing,
                 label_space: LabelSpace | None = None):
        self.data_processor = data_processor
        self.label_space = label_space or data_processor.flat_label_space()

    def _probabilities(self, model, loader, device):
        """Probabilidades por classe (softmax) que `model` atribui a cada amostra."""
        model.to(device).eval()

        probs, labels = [], []
        with torch.no_grad():
            for images, y in loader:
                images = images.to(device)
                with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                    logits = model(images)
                # float() antes do softmax: em fp16 a exponencial satura fácil.
                probs.append(torch.softmax(logits.float(), dim=1).cpu())
                labels.append(y)

        return torch.cat(probs), torch.cat(labels)

    def _loader(self, data, hyperparameters, device):
        dataset = CellClassificationDataset(
            data,
            data_processor=self.data_processor,
            width=hyperparameters.width,
            height=hyperparameters.height,
            # Mesmo pré-processamento da validação: sem augmentation, determinístico.
            transform=build_eval_transform(),
            label_space=self.label_space,
        )
        return DataLoader(
            dataset,
            batch_size=hyperparameters.batch_size,
            shuffle=False,
            num_workers=hyperparameters.num_workers,
            pin_memory=device.type == 'cuda',
        )

    def evaluate(self, checkpoints, data, hyperparameters, output_dir=None):
        """
        Avalia no conjunto `data` cada checkpoint e, além disso, o ensemble deles
        (média das probabilidades).

        Recebe uma LISTA porque a validação cruzada produz k modelos, não um. As
        três leituras que isso permite:

        - Por modelo, agregado em média ± desvio: o desempenho esperado de UM
          modelo treinado com esta configuração, com barra de erro. É o número
          honesto a reportar.
        - O ensemble: costuma bater qualquer membro individual, porque os erros
          independentes dos k modelos se cancelam na média. É o número a usar se o
          que você quer é o melhor sistema possível, e não uma estimativa dele.
        - A diferença entre os dois: mede o quanto do erro é instabilidade do
          treino (cancelável) e o quanto é dificuldade real da tarefa (não).

        Args:
            checkpoints (list[Path]): caminhos dos best_model.pt (um por fold).
            data (Subset): conjunto a avaliar — normalmente get_test_data().
            hyperparameters (Hyperparameters): para batch_size, width, height...
            output_dir (Path | None): onde salvar relatório e matriz de confusão.

        Returns:
            dict: métricas por modelo, o agregado e as do ensemble.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        loader = self._loader(data, hyperparameters, device)
        class_names = self.label_space.names

        todas_probs, labels = [], None
        por_modelo = []

        for i, checkpoint in enumerate(checkpoints, start=1):
            # Um modelo por vez na GPU: os k não precisam coexistir.
            model, label_space = load_from_checkpoint(checkpoint)

            # O checkpoint sabe a que tarefa pertence. Se ele não bater com a do
            # Evaluator, as predições seriam interpretadas na chave errada — um
            # índice 2 significaria "alto grau" num e "LSIL" noutro.
            if label_space.names != class_names:
                raise ValueError(
                    f'Checkpoint {checkpoint} é da tarefa {label_space.names}, '
                    f'mas o Evaluator está avaliando {class_names}.'
                )

            probs, labels = self._probabilities(model, loader, device)
            todas_probs.append(probs)

            metricas = self._metrics(labels, probs.argmax(dim=1))
            por_modelo.append(metricas)
            logger.info(f'  modelo {i}/{len(checkpoints)}: F1-macro={metricas["f1_macro"]:.4f}')

            del model
            torch.cuda.empty_cache()

        # Ensemble: média das PROBABILIDADES, não dos votos. Um modelo inseguro
        # (0,4/0,35/0,25) contribui menos do que um confiante (0,9/0,05/0,05), o
        # que não aconteceria numa votação simples.
        probs_ensemble = torch.stack(todas_probs).mean(dim=0)
        preds_ensemble = probs_ensemble.argmax(dim=1)
        metricas_ensemble = self._metrics(labels, preds_ensemble)

        f1s = [m['f1_macro'] for m in por_modelo]
        agregado = {
            'mean': statistics.mean(f1s),
            'std': statistics.stdev(f1s) if len(f1s) > 1 else 0.0,
        }

        logger.info(
            f'Teste: F1-macro individual = {agregado["mean"]:.4f} ± {agregado["std"]:.4f} '
            f'| ensemble = {metricas_ensemble["f1_macro"]:.4f}'
        )

        if output_dir is not None:
            self._save_artifacts(
                labels=labels,
                preds=preds_ensemble,
                class_names=class_names,
                output_dir=Path(output_dir),
                por_modelo=por_modelo,
                agregado=agregado,
                ensemble=metricas_ensemble,
            )

        return {
            'per_model': por_modelo,
            'aggregate': agregado,
            'ensemble': metricas_ensemble,
        }

    def _metrics(self, labels, preds):
        num_classes = len(self.label_space)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, labels=range(num_classes), average=None, zero_division=0,
        )
        return {
            'f1_macro': float(f1.mean()),
            'precision_macro': float(precision.mean()),
            'recall_macro': float(recall.mean()),
            'accuracy': float((torch.as_tensor(preds) == torch.as_tensor(labels)).float().mean()),
        }

    def _save_artifacts(self, labels, preds, class_names, output_dir,
                        por_modelo, agregado, ensemble):
        """Salva o relatório e a matriz de confusão do ensemble no conjunto de teste."""
        output_dir.mkdir(parents=True, exist_ok=True)
        num_classes = len(class_names)

        cm = confusion_matrix(labels, preds, labels=range(num_classes))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        fig, ax = plt.subplots(figsize=(8, 8))
        disp.plot(ax=ax, cmap='Blues', xticks_rotation=45, colorbar=False)
        ax.set_title('Matriz de Confusão — TESTE (ensemble)')
        fig.tight_layout()
        fig.savefig(output_dir / 'test_confusion_matrix.png', dpi=150)
        plt.close(fig)

        linhas = [
            f'Conjunto de TESTE — {len(labels)} células, {num_classes} classes',
            '',
            'F1-macro de cada modelo (um por fold da validação cruzada):',
        ]
        linhas += [
            f'  modelo {i}: {m["f1_macro"]:.4f}'
            for i, m in enumerate(por_modelo, start=1)
        ]
        linhas += [
            '',
            f'  média ± desvio : {agregado["mean"]:.4f} ± {agregado["std"]:.4f}',
            '',
            'Ensemble dos modelos (média das probabilidades):',
            f'  F1-macro  : {ensemble["f1_macro"]:.4f}',
            f'  Precision : {ensemble["precision_macro"]:.4f}',
            f'  Recall    : {ensemble["recall_macro"]:.4f}',
            f'  Accuracy  : {ensemble["accuracy"]:.4f}',
            '',
            'Relatório por classe (ensemble):',
            '',
            classification_report(
                labels, preds,
                labels=range(num_classes),
                target_names=class_names,
                zero_division=0,
            ),
        ]

        (output_dir / 'test_report.txt').write_text('\n'.join(linhas) + '\n')
