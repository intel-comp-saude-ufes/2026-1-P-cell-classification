"""
Arquivo: src/config/hyperparameters.py
Descrição:
    Este arquivo contém uma dataclass para armazenar hiperparâmetros que serão
    passados para o modelo a ser treinado.
"""
from dataclasses import dataclass

@dataclass
class Hyperparameters:
    """
    Classe para armazenar os hiperparâmetros do modelo.

    Args:
        learning_rate (float): Taxa de aprendizado da cabeça de classificação (fc).
        backbone_lr (float | None): Taxa de aprendizado da backbone. Se None, usa
            a mesma de `learning_rate`. A backbone já vem pré-treinada e só precisa
            de ajuste fino, então costuma pedir um LR uma ou duas ordens de grandeza
            menor que o da cabeça, que parte de pesos aleatórios.
        trainable_blocks (int | None): nº de blocos FINAIS da backbone que ficam
            treináveis. Controla a CAPACIDADE do modelo, e portanto o overfitting.
            - None: backbone inteira treinável (10,70 M de params).
            - 3: 8,51 M (80%)   - 2: 3,88 M (36%)   - 1: 0,59 M (6%)
            - 0: backbone toda congelada — linear probe, só a fc treina.
        freeze_epochs (int): nº de épocas iniciais com a backbone INTEIRA congelada,
            treinando só a cabeça. Resolve um problema diferente de `trainable_blocks`:
            não é sobre capacidade, é sobre a ORDEM do treino. A fc começa com pesos
            aleatórios e, nas primeiras iterações, produz gradientes que são ruído;
            propagá-los distorce as features pré-treinadas antes que elas rendessem
            algo. Esta fase deixa a cabeça alcançar a backbone antes de mexer nela.
            Terminada, aplica-se `trainable_blocks`. 0 desliga o warm-up.
        batch_size (int): Tamanho do lote.
        num_epochs (int): Número de épocas.
        optimizer (str): Otimizador a ser usado.
        loss_function (str): Função de perda a ser usada.
        width (int): Largura da imagem.
        height (int): Altura da imagem.
        dropout (float): dropout utilizado na parte linear.
        patience (int): nº de épocas sem melhora no F1-macro antes de parar (early stopping).
        min_delta (float): melhora mínima no F1-macro para contar como progresso.
        num_workers (int): nº de processos paralelos para carregar os dados (DataLoader).
        balance_strategy (str): estratégia contra o desbalanceamento de classes.
            - "none": sem balanceamento (padrão).
            - "sampler": WeightedRandomSampler com peso 1/frequência.
            - "sampler_sqrt": WeightedRandomSampler com peso 1/sqrt(frequência).
            - "weighted_loss": CrossEntropyLoss ponderada pelo inverso da frequência.
        label_smoothing (float): suaviza os rótulos, distribuindo esta fração da
            probabilidade entre as demais classes em vez de exigir 1.0 na correta.
            Ataca o excesso de confiança (val_loss subindo com o F1 parado). Aqui é
            também conceitualmente honesto: "ASC-H = não dá pra excluir HSIL" é um
            rótulo que de fato não é 100% certo, nem entre patologistas. 0.1 é o usual.
        weight_decay (float): regularização L2 do AdamW. O default do PyTorch é 0.01.
    """
    learning_rate: float
    batch_size: int
    num_epochs: int
    width: int
    height: int
    dropout: float
    patience: int = 5
    min_delta: float = 0.0
    num_workers: int = 8
    balance_strategy: str = "none"
    backbone_lr: float | None = None
    trainable_blocks: int | None = None
    freeze_epochs: int = 0
    label_smoothing: float = 0.0
    weight_decay: float = 0.01