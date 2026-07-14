"""
Arquivo: src/torch/utils/train_strategy.py
Descrição:
    Este arquivo contém a classe que irá definir 
    a estratégia de treinamento para os modelos do PyTorch.
"""
import copy
import math
import random
import torch
import logging
import matplotlib
import numpy as np

matplotlib.use('Agg')

from pathlib import Path
from collections import Counter
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

from src.data.process_data import DataProcessing, LabelSpace
from src.torch.modules.dataset import CellClassificationDataset
from src.torch.modules.model import CellClassifier

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Resolução de entrada da EfficientNet-B3.
INPUT_SIZE = 300


def build_eval_transform():
    """
    Pré-processamento determinístico, sem augmentation: só Resize + Normalize.

    É função de módulo, e não código solto dentro do train(), porque a VALIDAÇÃO e
    o TESTE precisam receber exatamente o mesmo tratamento. Duplicar essa lista em
    dois lugares é o jeito clássico de introduzir um desvio silencioso entre o que
    o modelo viu ao ser selecionado e o que ele vê ao ser medido.
    """
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def set_seed(seed: int):
    """
    Semeia todos os geradores que afetam um treino.

    O `random_state` do DataProcessing semeia apenas o SPLIT (o StratifiedGroupKFold
    do sklearn). Tudo o mais é estocástico e vinha do gerador global do PyTorch, sem
    semente: a inicialização da cabeça, as máscaras de dropout, a augmentation, a
    ordem do DataLoader e o WeightedRandomSampler (que sorteia com reposição).

    Sem isto, dois treinos com a mesma configuração dão resultados diferentes, e
    comparar duas configurações vira leitura de folha de chá.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TrainingStrategy():
    """
    Estratégia de treinamento para modelos do PyTorch.

    Args:
        ABC (ABC): Modelo abstrato base
    """
    def __init__(self, hyperparameters, data_processor: DataProcessing,
                 label_space: LabelSpace | None = None):
        self.hyperparameters = hyperparameters
        self.data_processor = data_processor
        
        # Define a tarefa: 6, 3 ou 2 classes. Sem ele, as 6 originais.
        self.label_space = label_space or data_processor.flat_label_space()

    def train(self, train_data, val_data, output_dir=None, seed=None):
        if seed is not None:
            set_seed(seed)

        # Extraindo hiperparâmetros
        width = self.hyperparameters.width
        height = self.hyperparameters.height
        batch_size = self.hyperparameters.batch_size
        lr = self.hyperparameters.learning_rate
        num_epochs = self.hyperparameters.num_epochs
        dropout = self.hyperparameters.dropout
        patience = self.hyperparameters.patience
        min_delta = self.hyperparameters.min_delta
        num_workers = self.hyperparameters.num_workers
        balance_strategy = self.hyperparameters.balance_strategy
        trainable_blocks = self.hyperparameters.trainable_blocks
        freeze_epochs = self.hyperparameters.freeze_epochs
        label_smoothing = self.hyperparameters.label_smoothing
        weight_decay = self.hyperparameters.weight_decay
        backbone_lr = self.hyperparameters.backbone_lr
        if backbone_lr is None:
            backbone_lr = lr

        label_space = self.label_space
        class_names = label_space.names
        num_classes = len(label_space)

        # Verificando a utilização do cuda
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Transformações nos dados.
        train_transform = transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            
            # Rotação total
            transforms.RandomRotation(180),
            
            # Impedem o modelo de assumir que o núcleo está sempre exatamente no centro do recorte
            transforms.RandomResizedCrop(
                INPUT_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1), antialias=True
            ),
            
            # Sem isto, a rede aprende a coloração da lâmina em vez da morfologia da célula
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        val_transform = build_eval_transform()

        # Criando datasets do PyTorch
        train_dataset = CellClassificationDataset(
            train_data,
            data_processor=self.data_processor,
            width=width, height=height,
            transform=train_transform,
            label_space=label_space,
        )

        val_dataset = CellClassificationDataset(
            val_data,
            data_processor=self.data_processor,
            width=width, height=height,
            transform=val_transform,
            label_space=label_space,
        )

        # Balanceamento de classes (só no treino)
        sampler = None
        class_weight_tensor = None
        if balance_strategy != "none":
            train_labels = [
                label_space.index(train_data[i].label)
                for i in range(len(train_data))
            ]
            class_counts = Counter(train_labels)

            if balance_strategy in ("sampler", "sampler_sqrt"):
                if balance_strategy == "sampler_sqrt":
                    class_weights = {c: 1.0 / math.sqrt(n) for c, n in class_counts.items()}
                else:
                    class_weights = {c: 1.0 / n for c, n in class_counts.items()}

                sample_weights = [class_weights[label] for label in train_labels]
                sampler = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(sample_weights),
                    replacement=True,
                )
            elif balance_strategy == "weighted_loss":
                total = len(train_labels)
                weights = [
                    total / (num_classes * class_counts[c]) if class_counts.get(c, 0) > 0 else 0.0
                    for c in range(num_classes)
                ]
                class_weight_tensor = torch.tensor(weights, dtype=torch.float, device=device)
            else:
                raise ValueError(
                    f"balance_strategy inválido: {balance_strategy!r}. Use "
                    f"'none', 'sampler', 'sampler_sqrt' ou 'weighted_loss'."
                )

        # Criando dataloaders do PyTorch.
        pin_memory = device.type == 'cuda'
        persistent_workers = num_workers > 0

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
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

        # Havendo warm-up, a fase 1 congela. Caso contrário, já entra na configuração definitiva.
        model.set_backbone_trainable(0 if freeze_epochs > 0 else trainable_blocks)

        # Dois grupos de parâmetros: a backbone e a cabeça
        optimizer = torch.optim.AdamW(
            [
                {"params": model.cnn_features.parameters(), "lr": backbone_lr},
                {"params": model.fc.parameters(), "lr": lr},
            ],
            weight_decay=weight_decay,
        )

        # Agendamento do LR reativo.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=3,
        )
        # weight=None (padrão) quando não é 'weighted_loss'.
        loss_func = nn.CrossEntropyLoss(
            weight=class_weight_tensor, label_smoothing=label_smoothing
        )

        # Mixed precision (AMP): roda as convoluções em float16 e mantém em
        # float32 só o que precisa de precisão
        use_amp = device.type == 'cuda'
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        
        # Estado para early stopping e para guardar o melhor modelo.
        best_f1 = float('-inf')
        best_epoch = 0
        best_model_state = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
        
        best_preds = []
        best_labels = []

        def count_trainable():
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(
            f'Parâmetros treináveis no início: {count_trainable()/1e6:.2f}M '
            f'(freeze_epochs={freeze_epochs}, trainable_blocks={trainable_blocks}).'
        )

        history = []
        for epoch in tqdm(range(num_epochs), desc='Train Progress: '):
            if freeze_epochs > 0 and epoch == freeze_epochs:
                model.set_backbone_trainable(trainable_blocks)
                epochs_no_improve = 0
                logger.info(
                    f'Época {epoch+1}: backbone descongelada '
                    f'({count_trainable()/1e6:.2f}M params treináveis, lr={backbone_lr}).'
                )

            model.train()

            train_loss = 0
            loop_interno = tqdm(train_loader, leave=False, desc=' Batch Progress: ')
            for images, labels in loop_interno:
                # Forward pass na rede
                images, labels = images.to(device), labels.to(device)

                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs = model(images)
                    loss = loss_func(outputs, labels)

                train_loss += loss.item()

                # Zerando os gradientes antes da atualização dos pesos
                optimizer.zero_grad()

                # Atualizando os pesos e aplicando passo do backpropagation.
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            
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
                    with torch.amp.autocast('cuda', enabled=use_amp):
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

            # Métricas de validação por classe
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

            # Métricas agregadas 'macro'
            precision = float(precision_pc.mean())
            recall = float(recall_pc.mean())
            f1 = float(f1_pc.mean())

            accuracy = float(np.mean(np.array(all_preds) == np.array(all_labels)))

            lr_antes = optimizer.param_groups[0]['lr']
            scheduler.step(f1)
            if optimizer.param_groups[0]['lr'] < lr_antes:
                logger.info(
                    f'Época {epoch+1}: F1 em platô, LR cortado pela metade '
                    f'(backbone: {optimizer.param_groups[0]["lr"]:.2e}, '
                    f'cabeça: {optimizer.param_groups[1]["lr"]:.2e}).'
                )

            history.append({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "accuracy": accuracy,
                "per_class": per_class,
            })
            
            # Early stopping
            if f1 > best_f1 + min_delta:
                best_f1 = f1
                best_epoch = epoch + 1
                best_model_state = copy.deepcopy(model.state_dict())
                best_preds = all_preds
                best_labels = all_labels
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info(
                        f'Early stopping na época {epoch+1}: sem melhora no '
                        f'F1 há {patience} épocas (melhor: {best_f1:.4f} '
                        f'na época {best_epoch}).'
                    )
                    break
            
            
            if output_dir is not None:
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
                
        # Restaura os pesos da melhor época
        model.load_state_dict(best_model_state)

        if output_dir is not None:
            self._save_learning_curves(
                history=history,
                output_dir=Path(output_dir),
                best_epoch=best_epoch,
            )

            # Persiste o modelo da melhor época
            checkpoint_path = Path(output_dir) / 'best_model.pt'
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'label_space': label_space,
                    'dropout': dropout,
                    'num_classes': num_classes,
                    'input_size': INPUT_SIZE,
                    'width': width,
                    'height': height,
                    'best_f1': best_f1,
                    'best_epoch': best_epoch,
                },
                checkpoint_path,
            )
            logger.info(f'Melhor modelo salvo em {checkpoint_path}')

        return {
            "model": model,
            "label_space": label_space,
            "best_f1": best_f1,
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