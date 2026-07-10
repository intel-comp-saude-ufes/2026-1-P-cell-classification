"""
Arquivo: src/data/process_data.py
Descrição:
    Este arquivo contém a classe que irá armazenar o dataset. Ele irá conter toda a
    lógica de processamento para entregar os dados a todas as outras interfaces do projeto.
"""
import numpy as np
import pandas as pd

from dataclasses import dataclass
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedGroupKFold

# ----- Data classes
@dataclass
class Cell:
    id: int
    x: int 
    y: int
    label: str
    image_path: str
# ------

class DataProcessing:
    """
    Classe responsável pelo processamento de dados.

    Args:
        None
    """
    def __init__(self, metadata: pd.DataFrame, random_state=None):
        self.metadata = metadata
        self.random_state = random_state
        
        self.__process_data()
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        return self.processed_data[idx]

    def label2index(self, label: str):
        return self.labels.index(label)

    def index2label(self, index: int):
        return self.labels[index]
    
    def get_labels(self):
        return self.labels

    def __process_data(self):
        """
        Constrói um objeto de dados processado a partir dos metadados do dataset.
        
        Também constrói uma lista de labels únicas presentes no dataset.
        """
        self.processed_data = [
            Cell(
                id=row["cell_id"],
                x=row["nucleus_x"],
                y=row["nucleus_y"],
                label=row["bethesda_system"],
                image_path=row["image_filename"]
            ) for _, row in self.metadata.iterrows()
        ]
        
        self.labels = list(self.metadata["bethesda_system"].unique())
    
    def __split_test_data(self, test_size=0.2):
        """
        Separa os dados em treino e teste

        Args:
            test_size (float): Proporção de dados a serem separados para teste.
        """
        outer_folds = round(1 / test_size)
        
        y = self.metadata["bethesda_system"]
        groups = self.metadata["image_filename"]
        
        splitter = StratifiedGroupKFold(
            n_splits=outer_folds,
            shuffle=True,
            random_state=self.random_state,
        )
        
        train_val_idx, test_idx = next(
            splitter.split(
                X=np.zeros(len(self.metadata)),
                y=y,
                groups=groups,
            )
        )
        
        self._train_val_indices = train_val_idx
        self._test_indices = test_idx
    
    def iterfolds(self, train_size=0.7, val_size=0.1, test_size=0.2, k_folds=5):
        """_summary_

        Args:
            train_size (float, optional): _description_. Defaults to 0.7.
            val_size (float, optional): _description_. Defaults to 0.1.
            test_size (float, optional): _description_. Defaults to 0.2.
            k_folds (int, optional): _description_. Defaults to 5.

        Yields:
            train: Conjunto de treino
            val: Conjunto de validação
        """
        self.__split_test_data(test_size=test_size)
        
        train_val_metadata = self.metadata.iloc[self._train_val_indices]
        
        y = train_val_metadata["bethesda_system"]
        groups = train_val_metadata["image_filename"]
        
        splitter = StratifiedGroupKFold(
            n_splits=k_folds,
            shuffle=True,
            random_state=self.random_state,
        )
        
        for train_local_idx, val_local_idx in splitter.split(
            X=np.zeros(len(train_val_metadata)),
            y=y,
            groups=groups,
        ):
            train_idx = self._train_val_indices[train_local_idx]
            val_idx = self._train_val_indices[val_local_idx]

            # Subset cria uma versão indexada do dataset, portanto se eu faço
            # train_data[0], ele irá fazer self[train_idx[0]]
            yield (
                Subset(self, train_idx.tolist()),
                Subset(self, val_idx.tolist()),
            )
    
    def get_test_data(self) -> Subset:
        """
        Utilizar somente após finalizar a validação cruzada.
        """
        return Subset(self, self._test_indices.tolist())
