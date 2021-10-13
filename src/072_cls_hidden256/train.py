import os
import random
import time
from logging import INFO, FileHandler, Formatter, StreamHandler, getLogger
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

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


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def make_feature(train_df, test_df):
    def _make_feature_per_dataset(df):
        u_out_change_time = df.loc[
            df.groupby("breath_id")["u_out"].diff() == 1, ["breath_id", "time_step"]
        ]
        u_out_change_time = u_out_change_time.rename(
            columns={"time_step": "u_out_change_time_step"}
        )
        df = df.merge(u_out_change_time, on="breath_id", how="left")
        df["time_from_u_out_change"] = df["time_step"] - df["u_out_change_time_step"]
        df.drop(["u_out_change_time_step"], axis=1, inplace=True)

        df["u_in_cumsum_per_time"] = df.groupby("breath_id")["u_in"].cumsum() / df["time_step"]
        df.loc[df["time_step"] == 0, "u_in_cumsum_per_time"] = 0
        df["u_in_diff"] = (
            df.groupby("breath_id")["u_in"].diff()
            / df.groupby("breath_id")["time_step"].diff()
        ).fillna(0)
        df["u_in_diff2"] = df.groupby("breath_id")["u_in_diff"].shift(1).fillna(0)
        df["u_in_diff_diff"] = df.groupby("breath_id")["u_in_diff"].diff().fillna(0)
        df["area"] = df["u_in"] * df["time_step"]
        df["area"] = df.groupby("breath_id")["area"].cumsum()
        df['cross2'] = df['time_step'] * df['u_out']

        df["u_in_sqrt"] = np.sqrt(df["u_in"] / 100)
        return df

    train_df = _make_feature_per_dataset(train_df)
    test_df = _make_feature_per_dataset(test_df)

    return train_df, test_df


def normalize_feature(train_df, valid_df, test_df):

    cols = [
        "time_from_u_out_change",
        "u_in",
        "u_in_cumsum_per_time",
        "u_in_diff",
        "u_in_diff2",
        "u_in_diff_diff",
        "area",
        "cross2",
        "u_in_sqrt"
    ]

    scaler = StandardScaler()
    train_df[cols] = scaler.fit_transform(train_df[cols])
    valid_df[cols] = scaler.transform(valid_df[cols])
    test_df[cols] = scaler.transform(test_df[cols])

    return train_df, valid_df, test_df


class PressureDataset(Dataset):
    def __init__(self, df, seq_features, target_dic, is_train=True):

        self.ids = df["id"].values
        self.breath_ids = df["breath_id"].unique()
        self.seq_features = seq_features

        self.target_dic = target_dic

        self.r_dict = {
            5: 0,
            20: 1,
            50: 2,
        }
        self.c_dict = {
            10: 0,
            20: 1,
            50: 2,
        }

        R_dict = df.groupby("breath_id")["R"].first()
        self.R_dict = R_dict.map(self.r_dict).to_dict()
        C_dict = df.groupby("breath_id")["C"].first()
        self.C_dict = C_dict.map(self.c_dict).to_dict()

        self.seq_features_arr_dict = {}
        for feat in self.seq_features:
            self.seq_features_arr_dict[feat] = (
                df.groupby("breath_id")[feat].apply(lambda x: x.values).to_dict()
            )

        self.is_train = is_train
        if is_train:
            self.target_arr_dict = (
                df.groupby("breath_id")["pressure"].apply(lambda x: x.values).to_dict()
            )
            df["pressure_cls"] = df["pressure"].map(self.target_dic)
            self.target_cls_arr_dict = (
                df.groupby("breath_id")["pressure_cls"].apply(lambda x: x.values).to_dict()
            )

    def __len__(self):
        return len(self.breath_ids)

    def __getitem__(self, idx):
        breath_id = self.breath_ids[idx]

        r_value = self.R_dict[breath_id]
        c_value = self.C_dict[breath_id]

        global_features = torch.tensor([r_value, c_value], dtype=torch.long)
        features = {"global": global_features}
        for feat in self.seq_features:
            features[feat] = torch.tensor(
                self.seq_features_arr_dict[feat][breath_id], dtype=torch.float
            )

        if self.is_train:
            target = torch.tensor(self.target_arr_dict[breath_id], dtype=torch.float)
            target_diff = torch.zeros(target.shape, dtype=torch.float)
            target_diff[1:] = target[1:] - target[0:-1]

            target_cls = torch.tensor(self.target_cls_arr_dict[breath_id], dtype=torch.long)
            return features, target, target_diff, target_cls
        else:
            return features


class RNNModel(nn.Module):
    def __init__(
        self, seq_features, pred_len=80, seq_len=80, device="cpu", n_hidden=256
    ):
        super(RNNModel, self).__init__()

        self.seq_features = seq_features
        self.seq_feature_len = len(seq_features)
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.device = device

        self.seq_linear = nn.Sequential(
            nn.Linear(self.seq_feature_len + 8 * 2, n_hidden),
            nn.LayerNorm(n_hidden),
            nn.ReLU(),
            # nn.Linear(n_hidden*2, n_hidden),
            # nn.ReLU()
        )

        self.r_emb = nn.Embedding(3, 8)
        self.c_emb = nn.Embedding(3, 8)

        self.encoder_rnn = nn.LSTM(
            num_layers=4,
            input_size=n_hidden,
            hidden_size=n_hidden,
            batch_first=True,
            bidirectional=True,
        )
        # self.decoder_out = nn.Linear(n_hidden*2 + 8*2, 1)  # lstm_hidden + id_embedding

        self.decoder_out = nn.Sequential(
            nn.Linear(n_hidden * 2, n_hidden),
            nn.LayerNorm(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1),
        )
        self.decoder_diff_out = nn.Sequential(
            nn.Linear(n_hidden * 2, n_hidden),
            nn.LayerNorm(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(n_hidden * 2, n_hidden),
            nn.LayerNorm(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 950),
        )

    def __call__(self, features):

        # global
        global_hidden = torch.cat(
            [
                self.r_emb(features["global"][:, 0]),
                self.c_emb(features["global"][:, 1]),
            ],
            dim=-1,
        )

        # sequence
        seq_input = torch.cat(
            [features[f].unsqueeze(-1) for f in self.seq_features], dim=-1
        )  # (batchsize, seq_len, feature_size)
        seq_input = torch.cat(
            [seq_input, global_hidden.unsqueeze(1).repeat([1, self.pred_len, 1])], dim=-1
        )

        seq_hidden = self.seq_linear(seq_input)  # (batchsize, seq_len, 32)

        hidden, (h_n, c_n) = self.encoder_rnn(seq_hidden)
        # pred = self.decoder_out(hidden)
        cls_pred = self.classifier(hidden)
        diff_pred = self.decoder_diff_out(hidden)
        # pred = torch.cat([pred, diff_pred], dim=-1)
        return cls_pred, diff_pred


class CustomLoss(nn.Module):
    """
    Directly optimizes the competition metric
    https://www.kaggle.com/theoviel/deep-learning-starter-simple-lstm
    """
    def __init__(self, max_epoch=100):
        self.max_epoch = max_epoch

    def __call__(self, preds, y, u_out, epoch):
        w = (1 - u_out)
        u_out_mae = w * (y - preds).abs()
        u_out_mae = u_out_mae.sum(-1) / (w.sum(-1))
        normal_mae = nn.L1Loss()(preds, y)
        epoch_w = min(epoch / self.max_epoch, 1.0)
        loss = u_out_mae * epoch_w + normal_mae * (1 - epoch_w)
        return loss


class MultitaskLoss(nn.Module):
    """
    Directly optimizes the competition metric
    https://www.kaggle.com/theoviel/deep-learning-starter-simple-lstm
    """

    def __init__(self, p_loss_weight_init, p_loss_weight_final, p_loss_weight_decay_period):
        super(MultitaskLoss, self).__init__()
        self.p_weight_init = p_loss_weight_init
        self.p_weight_final = p_loss_weight_final
        self.p_weight_period = p_loss_weight_decay_period
        self.pressure_loss = CustomLoss()
        self.pressure_diff_loss = CustomLoss()

    def __call__(self, preds, y, preds_diff, y_diff, u_out, epoch, iter_first=False):
        epoch_weight = min(epoch / self.p_weight_period, 1.0)
        p_weight = self.p_weight_init * (1 - epoch_weight) + self.p_weight_final * epoch_weight

        p_loss = self.pressure_loss(preds, y, u_out, epoch)
        p_diff_loss = self.pressure_diff_loss(preds_diff, y_diff, u_out, epoch)

        total_loss = p_loss * (p_weight) + p_diff_loss * (1 - p_weight)

        if iter_first:
            logger.info(f"multi task weight : {p_weight}")
        return total_loss


class MultitaskClsLoss(nn.Module):
    """
    Directly optimizes the competition metric
    https://www.kaggle.com/theoviel/deep-learning-starter-simple-lstm
    """

    def __init__(self, p_loss_weight_init, p_loss_weight_final, p_loss_weight_decay_period):
        super(MultitaskClsLoss, self).__init__()
        self.p_weight_init = p_loss_weight_init
        self.p_weight_final = p_loss_weight_final
        self.p_weight_period = p_loss_weight_decay_period
        self.pressure_cls_loss = nn.CrossEntropyLoss()
        self.pressure_diff_loss = CustomLoss()

    def __call__(self, cls_preds, cls_y, preds_diff, y_diff, u_out, epoch, iter_first=False):
        epoch_weight = min(epoch / self.p_weight_period, 1.0)
        p_weight = self.p_weight_init * (1 - epoch_weight) + self.p_weight_final * epoch_weight

        p_cls_loss = self.pressure_cls_loss(cls_preds, cls_y)
        p_diff_loss = self.pressure_diff_loss(preds_diff, y_diff, u_out, epoch)

        total_loss = p_cls_loss * (p_weight) + p_diff_loss * (1 - p_weight)

        if iter_first:
            logger.info(f"multi task weight : {p_weight}")
        return total_loss


def train_one_epoch(
    model, loss_fn, data_loader, optimizer, config, device, epoch=0, scaler=None
):
    # get batch data loop
    epoch_loss = 0
    epoch_data_num = len(data_loader.dataset)

    model.train()

    for iter_i, (features, targets, targets_diff, cls_targets) in enumerate(data_loader):
        # input
        features = {k: v.to(device) for k, v in features.items()}
        cls_targets = cls_targets.to(device)
        targets = targets.to(device)
        targets_diff = targets_diff.to(device)
        u_out = features["u_out"].to(device).view(-1)

        # zero grad
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            p_cls_preds, p_diff_preds = model(features)

            p_cls_preds = p_cls_preds.view(-1, 950)
            p_diff_preds = p_diff_preds.view(-1)
            # targets = targets.view(-1)
            cls_targets = cls_targets.view(-1)
            targets_diff = targets_diff.view(-1)

            loss = loss_fn(
                p_cls_preds, cls_targets, p_diff_preds, targets_diff, u_out, epoch,
                iter_first=(iter_i == 0)
            )
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    epoch_loss_per_data = epoch_loss / epoch_data_num
    return epoch_loss_per_data


def valid_one_epoch(model, loss_fn, data_loader, config, device):
    # get batch data loop
    epoch_loss = 0
    epoch_data_num = len(data_loader.dataset)
    pred_list = []
    target_list = []

    model.eval()
    for iter_i, (features, targets, targets_diff, cls_targets) in enumerate(data_loader):
        # input
        features = {k: v.to(device) for k, v in features.items()}
        targets = targets.to(device)

        with torch.no_grad():

            p_cls_preds, p_diff_preds = model(features)
            p_cls_preds = p_cls_preds.view(-1, 950)

            targets = targets.view(-1)

        pred_list.append(
            np.argmax(p_cls_preds.detach().cpu().numpy(), axis=1)
        )
        target_list.append(targets.detach().cpu().numpy())

    epoch_loss_per_data = epoch_loss / epoch_data_num
    val_preds = np.concatenate(pred_list, axis=0)
    val_targets = np.concatenate(target_list, axis=0)
    return epoch_loss_per_data, val_preds, val_targets


def train_run(
    train_df, valid_df, config, model_prefix="", save_best_model=True, save_model=True,
    target_dic=None, target_dic_inv=None
):

    set_seed(config.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info(f"train run device : {device}")

    ###################################
    # Model
    ###################################
    model = RNNModel(
        seq_features=config.seq_features, n_hidden=config.n_hidden, device=device
    )
    model.to(device)

    ###################################
    # Make data
    ###################################
    train_dataset = PressureDataset(
        train_df, seq_features=config.seq_features, is_train=True,
        target_dic=target_dic
    )
    valid_dataset = PressureDataset(
        valid_df, seq_features=config.seq_features, is_train=True,
        target_dic=target_dic
    )

    # data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
    )

    ##################
    # Optimiizer
    ##################
    lr = config.lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    ##################
    # lr scheduler
    ##################
    scheduler = config.SchedulerClass(optimizer, **config.scheduler_params)

    ##################
    # loss function
    ##################
    # loss_fn = nn.L1Loss()
    # loss_fn = CustomLoss()
    loss_fn = MultitaskClsLoss(
        config.p_loss_weight_init,
        config.p_loss_weight_final,
        config.p_loss_weight_decay_period
    )

    ###############################
    # train epoch loop
    ###############################
    # iteration and loss count
    val_score = 0
    best_model_path = None
    best_val_score = 10000

    results_list = []
    old_model_save_path = None

    val_calc_indices = valid_df["u_out"] == 0

    for epoch in range(config.n_epoch):

        t_epoch_start = time.time()

        # train loop
        train_epoch_loss = train_one_epoch(
            model, loss_fn, train_loader, optimizer, config, device,
            epoch=epoch
        )

        # valid loop
        valid_epoch_loss, val_cls_preds, val_targets = valid_one_epoch(
            model, loss_fn, valid_loader, config, device
        )
        # change cls to numerical preds
        val_preds = np.vectorize(target_dic_inv.get)(val_cls_preds)
        # calc metric
        val_score = mean_absolute_error(
            val_targets[val_calc_indices], val_preds[val_calc_indices]
        )

        t_epoch_finish = time.time()
        elapsed_time = t_epoch_finish - t_epoch_start

        # learning rate step
        lr = optimizer.param_groups[0]["lr"]

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(valid_epoch_loss)
        else:
            scheduler.step()
        # save results
        results = {
            "epoch": epoch + 1,
            "lr": lr,
            "train_loss": train_epoch_loss,
            "mae": val_score,
            "elapsed_time": elapsed_time,
        }

        results_list.append(results)

        logger.info(results)

        if val_score < best_val_score:
            best_val_score = val_score

            if save_best_model:

                if old_model_save_path is not None and old_model_save_path.exists():
                    os.remove(old_model_save_path)

                model_save_path = (
                    EXP_DIR
                    / f"{model_prefix}best-checkpoint-{str(epoch).zfill(3)}epoch.bin"
                )
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch,
                    },
                    model_save_path,
                )

                old_model_save_path = model_save_path
                best_model_path = model_save_path

        if epoch == config.n_epoch - 1 and save_model:
            model_save_path = EXP_DIR / f"{model_prefix}last-checkpoint.bin"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                },
                model_save_path,
            )

    return best_val_score, results_list, best_model_path


def test_predict(test_df, model_path, config, target_dic, target_dic_inv):

    set_seed(config.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ###################################
    # Model
    ###################################
    model = RNNModel(
        seq_features=config.seq_features, n_hidden=config.n_hidden, device=device
    )
    model.to(device)

    model.load_state_dict(
        torch.load(model_path, map_location=lambda storage, loc: storage)[
            "model_state_dict"
        ]
    )

    ###################################
    # Dataset & Dataloader
    ###################################
    test_dataset = PressureDataset(
        test_df, seq_features=config.seq_features, is_train=False, target_dic=target_dic
    )

    # data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
    )

    ###################################
    # Inference
    ###################################

    pred_list = []

    model.eval()
    for iter_i, features in enumerate(test_loader):
        # input
        features = {k: v.to(device) for k, v in features.items()}

        with torch.no_grad():
            p_cls_preds, p_diff_preds = model(features)
            p_cls_preds = p_cls_preds.view(-1, 950)

        pred_list.append(
            np.argmax(p_cls_preds.detach().cpu().numpy(), axis=1)
        )

    test_preds = np.concatenate(pred_list, axis=0)
    test_preds = np.vectorize(target_dic_inv.get)(test_preds)
    return test_preds


class Config:
    seed = 2021

    num_workers = 4
    batch_size = 512
    n_epoch = 50
    lr = 5e-3

    SchedulerClass = CosineAnnealingLR
    scheduler_params = dict(T_max=n_epoch, eta_min=1e-5)
    n_cv_fold = 5
    use_fp16 = False

    p_loss_weight_init = 1.0  # 0.7
    p_loss_weight_final = 1.0  # 0.9
    p_loss_weight_decay_period = 150

    n_hidden = 256
    seq_features = [
        "time_from_u_out_change",
        "u_in",
        "u_out",
        "u_in_cumsum_per_time",
        "u_in_diff",
        "u_in_diff2",
        "u_in_diff_diff",
        "area",
        "cross2",
        "u_in_sqrt"
    ]
    train_folds = [0]


def run():
    train_df = pd.read_csv(INPUT_DIR / "ventilator-pressure-prediction/train.csv")
    test_df = pd.read_csv(INPUT_DIR / "ventilator-pressure-prediction/test.csv")

    config = Config()

    folds = GroupKFold(n_splits=config.n_cv_fold)
    oof_preds = np.zeros(len(train_df))
    test_preds_list = []

    train_df, test_df = make_feature(train_df, test_df)

    target_dic = {
        v: i for i, v in
        enumerate(sorted(train_df['pressure'].unique().tolist()))
    }
    target_dic_inv = {v: k for k, v in target_dic.items()}

    model_num = 0
    val_scores = []
    for fold_ix, (trn_idx, val_idx) in enumerate(
        folds.split(train_df, groups=train_df["breath_id"])
    ):
        if fold_ix not in config.train_folds:
            continue
        logger.info(f"Fold {fold_ix}")

        _train_df = train_df.iloc[trn_idx].reset_index(drop=True)
        _valid_df = train_df.iloc[val_idx].reset_index(drop=True)

        _train_df, _valid_df, _test_df = normalize_feature(
            _train_df, _valid_df, test_df.copy()
        )

        best_val_score, results_list, best_model_path = train_run(
            _train_df,
            _valid_df,
            config,
            model_prefix=f"fold{fold_ix}",
            save_best_model=True,
            save_model=True,
            target_dic=target_dic,
            target_dic_inv=target_dic_inv
        )
        model_num += 1

        val_scores.append(best_val_score)
        oof_preds[val_idx] = test_predict(_valid_df, best_model_path, config, target_dic, target_dic_inv)
        test_preds_list.append(
            test_predict(_test_df, best_model_path, config, target_dic, target_dic_inv)
        )

    test_preds = np.mean(test_preds_list, axis=0)
    val_score = np.mean(val_scores)
    logger.info(f"Total CV: {val_score}")
    sub_df = pd.read_csv(
        INPUT_DIR / "ventilator-pressure-prediction/sample_submission.csv"
    )
    sub_df["pressure"] = test_preds
    sub_df.to_csv(f"submission_{val_score:.5f}.csv", index=False)

    # raw prediction
    for fold_ix, test_pred in enumerate(test_preds_list):
        sub_df[f"fold{fold_ix}"] = test_pred
    sub_df.to_csv("raw_submission.csv", index=False)

    # oof prediction
    train_df["preds"] = oof_preds
    save_cols = [
        "id", "breath_id", "time_step", "pressure", "preds"
    ]
    train_df[save_cols].to_csv("oof_df.csv", index=False)


logger = init_logger(EXP_DIR / "run.log")

if __name__ == "__main__":
    run()
