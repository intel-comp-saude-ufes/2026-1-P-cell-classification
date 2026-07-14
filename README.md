# 2026-1-P-cell-classification

## Sobre

Clique no [link](https://youtu.be/l_bRUnQDzzY) para ver a o vídeo de apresentação deste trabalho.

## Instalação

Utilize o gerenciador de pacotes `uv` para instalar e gerir as dependênicas deste projeto.

Abra seu terminal na raiz do projeto e rode o comando:
```bash
uv sync
```

## Rodando o projeto

### Adição de dados

Adicione o seu conjunto de dados na pasta `data/raw/`. Essa pasta deve conter duas coisas principais:

- Um arquivo chamado `classifications.csv`.
- Um diretório chamado `images` contendo todas as imagens.

### Rodando o código

Para rodar o código abra o terminal na pasta raiz do projeto e rode o comando:
```bash
uv run python -m src.scripts.train_and_eval
```

## Arquitetura do projeto

A arquitetura do projeto consiste em desacoplar algumas partes do código, de modo que cada parte se comunique uma com a outra através de uma interface.

![figura](docs/figs/modelagem_treinamento.drawio.svg)