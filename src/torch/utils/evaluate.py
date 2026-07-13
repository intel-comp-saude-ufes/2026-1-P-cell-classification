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

# As quatro métricas reportadas, na ordem em que aparecem nas tabelas. Uma única
# lista para não haver duas ordens diferentes entre o cálculo e a escrita.
METRICAS = (
    ('f1_macro', 'F1-macro'),
    ('precision_macro', 'Precisão'),
    ('recall_macro', 'Recall'),
    ('accuracy', 'Acurácia'),
)


def aggregate_metrics(por_modelo):
    """
    Média e desvio de cada métrica ENTRE os k modelos.

    É o desempenho esperado de UM modelo treinado com esta configuração, com barra
    de erro — o número honesto a reportar. Quando os k modelos são medidos no mesmo
    conjunto (o teste), o desvio mede só a instabilidade do treino: a variação não
    vem da amostra, vem da semente.

    É função de módulo, e não método do Evaluator, porque a conta não depende de
    nada do avaliador — e o recompute_metrics.py agrega com ela os folds da
    validação cruzada reavaliados, sem precisar instanciar coisa nenhuma.

    Args:
        por_modelo (list[dict]): as métricas de cada modelo, como saem de evaluate().

    Returns:
        dict: métrica -> {'mean', 'std'}.
    """
    def mean_std(valores):
        return {
            'mean': statistics.mean(valores),
            'std': statistics.stdev(valores) if len(valores) > 1 else 0.0,
        }

    return {
        metrica: mean_std([m[metrica] for m in por_modelo])
        for metrica, _ in METRICAS
    }


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

        agregado = aggregate_metrics(por_modelo)

        logger.info(
            f'Teste: F1-macro individual = {agregado["f1_macro"]["mean"]:.4f} '
            f'± {agregado["f1_macro"]["std"]:.4f} '
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

        rotulos = [rotulo for _, rotulo in METRICAS]
        chaves = [chave for chave, _ in METRICAS]

        linhas = [
            f'Conjunto de TESTE — {len(labels)} células, {num_classes} classes',
            '',
            'Cada modelo (um por fold da validação cruzada):',
            f'  {"modelo":<8} ' + ' '.join(f'{r:<10}' for r in rotulos),
        ]
        linhas += [
            f'  {i:<8} ' + ' '.join(f'{m[c]:<10.4f}' for c in chaves)
            for i, m in enumerate(por_modelo, start=1)
        ]
        linhas += [
            '',
            'Modelo individual (média ± desvio entre os modelos) — o desempenho',
            'esperado de UM modelo treinado com esta configuração:',
        ]
        linhas += [
            f'  {rotulo:<10}: {agregado[chave]["mean"]:.4f} ± {agregado[chave]["std"]:.4f}'
            for chave, rotulo in METRICAS
        ]
        linhas += [
            '',
            'Ensemble dos modelos (média das probabilidades):',
        ]
        linhas += [
            f'  {rotulo:<10}: {ensemble[chave]:.4f}'
            for chave, rotulo in METRICAS
        ]
        linhas += [
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
