"""
Arquivo: src/data/process_data.py
Descrição:
    Este arquivo contém a classe que irá armazenar o dataset. Ele irá conter toda a
    lógica de processamento para entregar os dados a todas as outras interfaces do projeto.
"""
import pandas as pd

from dataclasses import dataclass

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
    def __init__(self, metadata: pd.DataFrame):
        self.metadata = metadata
        
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
        
        Returns:
            test: Conjunto de teste
        """
        # TODO: Implementar a função de validação cruzada
        pass
