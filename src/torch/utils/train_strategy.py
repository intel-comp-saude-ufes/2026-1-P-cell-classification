"""
Arquivo: src/torch/utils/train_strategy.py
Descrição:
    Este arquivo contém a classe que irá definir 
    a estratégia de treinamento para os modelos do PyTorch.
"""
import copy
import torch
import logging

from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.data.process_data import DataProcessing
from src.torch.modules.dataset import CellClassificationDataset
from src.torch.modules.model import CellClassifier

logger = logging.getLogger(__name__)


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
    def train(self, train_data, val_data):
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

        # Verificando a utilização do cuda
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # TODO: Adicionar transformações nos dados
        
        # Criando datasets do PyTorch
        train_dataset = CellClassificationDataset(
            train_data,
            data_processor=self.data_processor,
            width=width, height=height,
            transform=None
        )
        
        val_dataset = CellClassificationDataset(
            val_data,
            data_processor=self.data_processor,
            width=width, height=height,
            transform=None
        )
        
        # Criando dataloaders do PyTorch
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
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
            with torch.no_grad():
                loop_interno_val = tqdm(val_loader, leave=False, desc=' Validation Progress: ')
                for images, labels in loop_interno_val:
                    images, labels = images.to(device), labels.to(device)
                    # Forward pass na rede
                    outputs = model(images)
                    loss = loss_func(outputs, labels)
                    
                    # Obtendo valor da loss de validação
                    val_loss += loss.item()
                    
                    # TODO: Adicionar métricas de avaliação (ex: acurácia, f1-score, etc.)
                    # TODO: Adicionar matriz de confusão e relatório de classificação
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            history.append({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss
            })

            logger.info(f'[{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

            # Early stopping: guarda os melhores pesos e para se a val_loss
            # não melhorar por `patience` épocas seguidas.
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                best_model_state = copy.deepcopy(model.state_dict())
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

        # Restaura os pesos da melhor época (não os da última, que podem já
        # estar em overfitting).
        model.load_state_dict(best_model_state)

        # TODO: Adicionar curva de aprendizado

        return {
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "history": history,
        }