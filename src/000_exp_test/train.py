import os
from logging import INFO, FileHandler, Formatter, StreamHandler, getLogger
from pathlib import Path

USE_TPU = False
EXP_DIR = Path("./")
if "KAGGLE_URL_BASE" in set(os.environ.keys()):
    INPUT_DIR = Path("../input")
else:
    INPUT_DIR = Path("../../input")


def init_logger(log_file: Path = "./train.log"):
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(asctime)s %(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


logger = init_logger(EXP_DIR / "run.log")

logger.info("Test Run")
