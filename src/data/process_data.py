"""
Arquivo: src/data/process_data.py
Descrição:
    Este arquivo contém a classe que irá armazenar o dataset. Ele irá conter toda a
    lógica de processamento para entregar os dados a todas as outras interfaces do projeto.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from dataclasses import dataclass
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedGroupKFold

# Agrupamento clínico dos rótulos do sistema Bethesda. O corte entre baixo e alto
# grau é o que decide conduta (repetir a citologia vs. encaminhar para colposcopia),
# e é o agrupamento de 3 classes usado na literatura.
NEGATIVE_LABEL = "Negative for intraepithelial lesion"
LOW_GRADE_LABELS = ["ASC-US", "LSIL"]
HIGH_GRADE_LABELS = ["ASC-H", "HSIL", "SCC"]


# ----- Data classes
@dataclass
class Cell:
    id: int
    x: int
    y: int
    label: str
    image_path: str


@dataclass
class LabelSpace:
    """
    Define o espaço de rótulos de UMA tarefa: quais classes existem e como cada
    rótulo original do dataset mapeia para um índice.

    Existe porque as mesmas células alimentam tarefas de granularidades diferentes.
    A célula rotulada "HSIL" é a classe 3 na tarefa de 6 classes, a classe 2 ("alto
    grau") na de 3, e a classe 1 ("lesão") na binária. Sem esta indireção, o
    mapeamento rótulo -> índice ficaria preso às 6 classes originais.

    Nenhuma célula é removida em nenhuma das tarefas — só re-rotulada. É por isso
    que as três compartilham exatamente o mesmo split: iterfolds() estratifica
    sempre por `bethesda_system`, independente do espaço de rótulos escolhido, e
    portanto os três resultados são comparáveis no mesmo conjunto de teste. Filtrar
    o metadata e refazer o split por tarefa (um CSV por caso) daria splits
    incompatíveis, e a comparação entre as tarefas seria ilusória.

    Args:
        names (list[str]): nomes das classes, na ordem dos índices.
        mapping (dict[str, int]): rótulo original do dataset -> índice nesta tarefa.
    """
    names: list[str]
    mapping: dict[str, int]

    def __len__(self):
        return len(self.names)

    def index(self, label: str) -> int:
        return self.mapping[label]
# ------

class DataProcessing:
    """
    Classe responsável pelo processamento de dados.

    Args:
        None
    """
    def __init__(self, metadata: pd.DataFrame, image_folder_path: Path,
                 random_state=None, test_size=0.2):
        self.metadata = metadata
        self.random_state = random_state
        self.image_folder_path = image_folder_path

        self.__process_data()

        # O conjunto de teste é decidido AQUI, uma única vez. Antes ele nascia
        # dentro do iterfolds(), o que fazia get_test_data() quebrar se ninguém
        # tivesse iterado os folds primeiro — e refazia o split do teste a cada
        # nova iteração. Ele não é efeito colateral de iterar folds: é uma
        # propriedade da partição dos dados.
        self.__split_test_data(test_size=test_size)
    
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

    def flat_label_space(self) -> LabelSpace:
        """As 6 classes originais do Bethesda."""
        return LabelSpace(
            names=list(self.labels),
            mapping={label: i for i, label in enumerate(self.labels)},
        )

    def grade_label_space(self) -> LabelSpace:
        """
        3 classes: negativo / baixo grau / alto grau.

        Funde os pares que o Bethesda define de forma ambígua — ASC-US com LSIL, e
        ASC-H com HSIL ("atípico, não dá para excluir HSIL"). Como nem os
        patologistas concordam nessas fronteiras, elas são ruído de rótulo: o
        agrupamento remove uma distinção que o modelo não tem como aprender.
        """
        mapping = {NEGATIVE_LABEL: 0}
        mapping.update({label: 1 for label in LOW_GRADE_LABELS})
        mapping.update({label: 2 for label in HIGH_GRADE_LABELS})
        return LabelSpace(
            names=["Negative", "Low grade", "High grade"],
            mapping=mapping,
        )

    def binary_label_space(self) -> LabelSpace:
        """2 classes: há lesão ou não. Tudo que não é o negativo vira 'lesão'."""
        return LabelSpace(
            names=["Negative", "Lesion"],
            mapping={
                label: (0 if label == NEGATIVE_LABEL else 1)
                for label in self.labels
            },
        )

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
                image_path= self.image_folder_path / row["image_filename"]
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
    
    def iterfolds(self, k_folds=5):
        """
        Gera os k splits (treino, validação) sobre os dados que NÃO são de teste.

        O conjunto de teste já foi separado no __init__ e não é tocado aqui.

        A estratificação é sempre por `bethesda_system` (as 6 classes originais),
        independente do espaço de rótulos da tarefa. É por isso que os modelos de
        6, 3 e 2 classes veem exatamente as mesmas lâminas em cada fold, e seus
        resultados são comparáveis entre si.

        Args:
            k_folds (int): número de folds. A fração de validação é 1/k_folds dos
                dados de treino+validação — não há parâmetro separado para ela.

        Yields:
            tuple[Subset, Subset]: conjunto de treino e conjunto de validação.
        """
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
        O conjunto de teste, separado no __init__ e agrupado por lâmina.

        Só deve ser MEDIDO uma vez, no fim, com a configuração já decidida. Se for
        consultado a cada ajuste de hiperparâmetro, deixa de ser teste e vira um
        segundo conjunto de validação — e a estimativa final fica otimista.
        """
        return Subset(self, self._test_indices.tolist())
