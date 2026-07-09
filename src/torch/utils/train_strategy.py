class TrainingStrategy():
    """
    Estratégia de treinamento para modelos do PyTorch.

    Args:
        ABC (ABC): Modelo abstrato base
    """
    def __init__(self):
        pass
    
    def train(self, model, hyperparameters, train_loader, val_loader):
        raise NotImplementedError("O método train() deve ser implementado.")