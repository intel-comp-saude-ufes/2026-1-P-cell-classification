"""
Arquivo: src/torch/utils/train_strategy.py
Descrição:
    Este arquivo contém a classe que irá definir 
    a estratégia de treinamento para os modelos do PyTorch.
"""
import copy
import torch
import logging
import matplotlib

matplotlib.use('Agg')  # backend sem interface gráfica, para salvar em arquivo

from pathlib import Path
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

from src.data.process_data import DataProcessing
from src.torch.modules.dataset import CellClassificationDataset
from src.torch.modules.model import CellClassifier

logger = logging.getLogger(__name__)

# Estatísticas do ImageNet, usadas na normalização. O backbone (EfficientNet-B3)
# é pré-treinado no ImageNet e espera receber entradas nessa distribuição.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TrainingStrategy():
    """
    Estratégia de treinamento para modelos do PyTorch.

    Args:
        ABC (ABC): Modelo abstrato base
    """
    def __init__(self, hyperparameters, data_processor: DataProcessing):
        self.hyperparameters = hyperparameters
        self.data_processor = data_processor

    # TODO: Treinamento deve retornar algumas informações para o cross validation
    #       para que elas sejam tratadas lá
    def train(self, train_data, val_data, output_dir=None):
        # Extraindo hiperparâmetros
        width = self.hyperparameters.width
        height = self.hyperparameters.height
        batch_size = self.hyperparameters.batch_size
        lr = self.hyperparameters.learning_rate
        num_epochs = self.hyperparameters.num_epochs
        dropout = self.hyperparameters.dropout
        num_classes = self.hyperparameters.num_classes
        patience = self.hyperparameters.patience
        min_delta = self.hyperparameters.min_delta
        num_workers = self.hyperparameters.num_workers

        # Nomes das classes na ordem dos índices (labels[i] == classe i)
        class_names = self.data_processor.get_labels()

        # Verificando a utilização do cuda
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Transformações nos dados.
        # - Normalize (ImageNet): obrigatório, pois o backbone pré-treinado
        #   espera entradas nessa distribuição.
        # - Augmentation (flips/rotação/jitter): aplicado SÓ no treino, como
        #   regularizador contra overfitting.
        # A validação recebe só ToTensor + Normalize, para ser determinística.
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        # Criando datasets do PyTorch
        train_dataset = CellClassificationDataset(
            train_data,
            data_processor=self.data_processor,
            width=width, height=height,
            transform=train_transform
        )

        val_dataset = CellClassificationDataset(
            val_data,
            data_processor=self.data_processor,
            width=width, height=height,
            transform=val_transform
        )
        
        # Criando dataloaders do PyTorch.
        # num_workers > 0 paraleliza a decodificação das imagens em vários
        # processos, evitando que a GPU fique ociosa esperando os dados.
        # pin_memory só ajuda quando há GPU (acelera a cópia CPU -> GPU) e
        # persistent_workers só faz sentido com workers (mantém os processos
        # vivos entre épocas, evitando recriá-los toda vez).
        pin_memory = device.type == 'cuda'
        persistent_workers = num_workers > 0

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        
        # Instânciando modelo, otimizador e função de custo
        model = CellClassifier(dropout, num_classes)
        model.to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        loss_func = nn.CrossEntropyLoss()
        
        # Estado para early stopping e para guardar o melhor modelo
        best_val_loss = float('inf')
        best_epoch = 0
        best_model_state = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
        # Predições/rótulos da melhor época, para a matriz de confusão e o relatório
        best_preds = []
        best_labels = []

        history = []
        for epoch in tqdm(range(num_epochs), desc='Train Progress: '):
            # Treinando pesos da rede
            model.train()
            
            train_loss = 0
            loop_interno = tqdm(train_loader, leave=False, desc=' Batch Progress: ')
            for images, labels in loop_interno:
                # Forward pass na rede
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = loss_func(outputs, labels)
                
                train_loss += loss.item()
                
                # Zerando os gradientes antes da atualização dos pesos
                optimizer.zero_grad()
                
                # Atualizando os pesos e aplicando passo do backpropagation
                loss.backward()
                optimizer.step()
            
            # Avaliação no conjunto de validação
            model.eval()
            
            val_loss = 0
            all_preds = []
            all_labels = []
            with torch.no_grad():
                loop_interno_val = tqdm(val_loader, leave=False, desc=' Validation Progress: ')
                for images, labels in loop_interno_val:
                    images, labels = images.to(device), labels.to(device)
                    # Forward pass na rede
                    outputs = model(images)
                    loss = loss_func(outputs, labels)

                    # Obtendo valor da loss de validação
                    val_loss += loss.item()

                    # Classe prevista = índice do maior logit; acumula para as métricas
                    preds = outputs.argmax(dim=1)
                    all_preds.extend(preds.cpu().tolist())
                    all_labels.extend(labels.cpu().tolist())

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            # Métricas de validação por classe. average=None retorna um array
            # com o valor de cada classe; labels=range(num_classes) garante que
            # todas as 6 classes apareçam sempre, na mesma ordem, mesmo que
            # alguma não tenha amostras nesta época. zero_division=0 evita
            # warning quando uma classe não recebe nenhuma predição.
            precision_pc, recall_pc, f1_pc, support_pc = precision_recall_fscore_support(
                all_labels, all_preds,
                labels=range(num_classes),
                average=None,
                zero_division=0,
            )

            # Métricas por classe, indexadas pelo nome da classe
            per_class = {
                (class_names[i] if i < len(class_names) else str(i)): {
                    "precision": float(precision_pc[i]),
                    "recall": float(recall_pc[i]),
                    "f1_score": float(f1_pc[i]),
                    "support": int(support_pc[i]),
                }
                for i in range(num_classes)
            }

            # Métricas agregadas 'macro' = média simples entre as classes,
            # tratando todas com o mesmo peso (bom quando são desbalanceadas).
            precision = float(precision_pc.mean())
            recall = float(recall_pc.mean())
            f1 = float(f1_pc.mean())

            history.append({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "per_class": per_class,
            })
            
            # Early stopping: guarda os melhores pesos e para se a val_loss
            # não melhorar por `patience` épocas seguidas.
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                best_model_state = copy.deepcopy(model.state_dict())
                best_preds = all_preds
                best_labels = all_labels
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info(
                        f'Early stopping na época {epoch+1}: sem melhora na '
                        f'val_loss há {patience} épocas (melhor: {best_val_loss:.4f} '
                        f'na época {best_epoch}).'
                    )
                    break
            
            
            if output_dir is not None:
                # Salva a matriz de confusão e o relatório de classificação do melhor
                # modelo, avaliado no conjunto de validação.
                self._save_evaluation_artifacts(
                    output_dir=Path(output_dir),
                    labels=best_labels,
                    preds=best_preds,
                    class_names=class_names,
                    num_classes=num_classes,
                    best_epoch=best_epoch,
                )
                
                # Salva a curva de aprendizado (loss) e a evolução das métricas macro.
                self._save_learning_curves(
                    history=history,
                    output_dir=Path(output_dir),
                    best_epoch=best_epoch,
                )
                
        # Restaura os pesos da melhor época (não os da última, que podem já
        # estar em overfitting).
        model.load_state_dict(best_model_state)

        return {
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "history": history,
            "output_dir": str(output_dir) if output_dir is not None else None,
        }

    def _save_evaluation_artifacts(
        self, output_dir, labels, preds, class_names, num_classes, best_epoch
    ):
        """
        Salva a matriz de confusão (PNG) e o relatório de classificação (TXT)
        do melhor modelo no diretório informado.

        Args:
            output_dir (Path): pasta onde os arquivos serão salvos.
            labels (list): rótulos verdadeiros da melhor época.
            preds (list): predições da melhor época.
            class_names (list): nomes das classes na ordem dos índices.
            num_classes (int): número total de classes.
            best_epoch (int): época que gerou esses resultados (para o título).
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Nomes na ordem dos índices, com fallback caso faltem nomes
        target_names = [
            class_names[i] if i < len(class_names) else str(i)
            for i in range(num_classes)
        ]

        # Matriz de confusão -> PNG
        cm = confusion_matrix(labels, preds, labels=range(num_classes))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=target_names
        )
        fig, ax = plt.subplots(figsize=(8, 8))
        disp.plot(ax=ax, cmap='Blues', xticks_rotation=45, colorbar=False)
        ax.set_title(f'Matriz de Confusão (melhor época: {best_epoch})')
        fig.tight_layout()
        fig.savefig(output_dir / 'confusion_matrix.png', dpi=150)
        plt.close(fig)

        # Relatório de classificação -> TXT
        report = classification_report(
            labels, preds,
            labels=range(num_classes),
            target_names=target_names,
            zero_division=0,
        )
        (output_dir / 'report.txt').write_text(report)

    def _save_learning_curves(self, history, output_dir, best_epoch=None):
        """
        Salva, em um único PNG, a curva de aprendizado (train/val loss) e a
        evolução das métricas macro (precision, recall, f1) ao longo das épocas.

        Args:
            history (list[dict]): métricas registradas a cada época.
            output_dir (Path): pasta onde o arquivo será salvo.
            best_epoch (int | None): época do melhor modelo; se informada, é
                marcada como referência nos dois gráficos.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        epochs = [h["epoch"] for h in history]

        # Cores categóricas colorblind-safe (validadas): uma por série.
        COR_TREINO, COR_VAL, COR_F1 = '#0072B2', '#D55E00', '#009E73'

        fig, (ax_loss, ax_metrics) = plt.subplots(1, 2, figsize=(14, 5))

        # --- Curva de aprendizado (loss) ---
        ax_loss.plot(
            epochs, [h["train_loss"] for h in history],
            color=COR_TREINO, linewidth=2, marker='o', label='Treino'
        )
        ax_loss.plot(
            epochs, [h["val_loss"] for h in history],
            color=COR_VAL, linewidth=2, marker='o', label='Validação'
        )
        ax_loss.set_title('Curva de Aprendizado')
        ax_loss.set_xlabel('Época')
        ax_loss.set_ylabel('Loss')
        ax_loss.grid(True, alpha=0.3)

        # --- Evolução das métricas macro (validação) ---
        ax_metrics.plot(
            epochs, [h["precision"] for h in history],
            color=COR_TREINO, linewidth=2, marker='o', label='Precision'
        )
        ax_metrics.plot(
            epochs, [h["recall"] for h in history],
            color=COR_VAL, linewidth=2, marker='o', label='Recall'
        )
        ax_metrics.plot(
            epochs, [h["f1_score"] for h in history],
            color=COR_F1, linewidth=2, marker='o', label='F1-score'
        )
        ax_metrics.set_title('Métricas Macro (validação)')
        ax_metrics.set_xlabel('Época')
        ax_metrics.set_ylabel('Valor')
        ax_metrics.set_ylim(0, 1)
        ax_metrics.grid(True, alpha=0.3)

        # Marca a época do melhor modelo (onde o early stopping selecionou).
        if best_epoch is not None:
            for ax in (ax_loss, ax_metrics):
                ax.axvline(
                    best_epoch, color='gray', linestyle='--',
                    linewidth=1, alpha=0.7, label=f'Melhor época ({best_epoch})'
                )

        ax_loss.legend()
        ax_metrics.legend()

        fig.tight_layout()
        fig.savefig(output_dir / 'learning_curves.png', dpi=150)
        plt.close(fig)