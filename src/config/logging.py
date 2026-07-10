"""
Arquivo: src/config/logging.py
Descrição:
    Este arquivo contém a função para configurar o logging do projeto.
"""
import sys
import logging


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="[%(asctime)s - %(levelname)s] %(message)s",
        stream=sys.stdout,
    )