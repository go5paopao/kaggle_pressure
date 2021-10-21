import os
import random
from logging import INFO, FileHandler, Formatter, StreamHandler, getLogger
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error

USE_TPU = False
EXP_DIR = Path("./")
if "KAGGLE_URL_BASE" in set(os.environ.keys()):
    INPUT_DIR = Path("../input")
    ENV = "kaggle"
elif 'COLAB_GPU' in set(os.environ.keys()):

    from google.colab import drive
    drive.mount('/content/drive')

    gpu_info = ""  # !nvidia-smi
    gpu_info = '\n'.join(gpu_info)
    if gpu_info.find('failed') >= 0:
        print('Not connected to a GPU')
    else:
        print(gpu_info)

    INPUT_DIR = Path("./drive/MyDrive/Colab Notebooks/VP/input/")
    from requests import get
    exp_name = get('http://172.28.0.2:9000/api/sessions').json()[0]['name'][:-6]  # remove .ipynb
    EXP_DIR = Path(f'./drive/MyDrive/Colab Notebooks/VP/{exp_name}/')
    EXP_DIR.mkdir(exist_ok=True)
    ENV = "colab"
else:
    INPUT_DIR = Path("../../input")
    ENV = "other"


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


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def run():
    train_df = pd.read_csv(INPUT_DIR / "ventilator-pressure-prediction/train.csv")
    test_df = pd.read_csv(INPUT_DIR / "ventilator-pressure-prediction/test.csv")

    pred_data_dir = Path("./pao_exp097_15fold")
    n_fold = 15
    oof_files = [pred_data_dir / f"_oof_fold{ix}.csv" for ix in range(n_fold)]
    test_files = [list(pred_data_dir.glob(f"_submission_fold{ix}_*.csv"))[0] for ix in range(n_fold)]

    # create oof pred
    oof_df_list = []
    for oof_path in oof_files:
        logger.info(f"loading {oof_path}")
        _df = pd.read_csv(oof_path)
        oof_df_list.append(_df)
    oof_df = pd.concat(oof_df_list, axis=0)
    oof_df = train_df[["id", "breath_id", "u_out"]].merge(
        oof_df, on=["breath_id", "id"], how="left"
    )

    # check score
    calc_indices = oof_df["u_out"] == 0
    val_score = mean_absolute_error(
        oof_df.loc[calc_indices, "pressure"],
        oof_df.loc[calc_indices, "preds"]
    )
    logger.info(f"Val Score: {val_score}")

    # create test pred
    pred_cols = []
    for fold_ix, test_path in enumerate(test_files):
        logger.info(f"loading {test_path}")
        _df = pd.read_csv(test_path)
        test_df[f"fold{fold_ix}"] = _df["pressure"]
        pred_cols.append(f"fold{fold_ix}")

    test_df["pressure"] = test_df[pred_cols].median(axis=1)
    test_df[["id", "pressure"]].to_csv(EXP_DIR / f"submission_{val_score:.5f}.csv", index=False)

    # raw prediction
    test_df.to_csv(EXP_DIR / "raw_submission.csv", index=False)

    # oof prediction
    save_cols = [
        "id", "breath_id", "time_step", "pressure", "preds"
    ]
    oof_df[save_cols].to_csv(EXP_DIR / "oof_df.csv", index=False)


logger = init_logger(EXP_DIR / "run.log")

if __name__ == "__main__":
    run()
