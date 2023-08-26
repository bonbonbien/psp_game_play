# %%
import pandas as pd
import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader

from config import event2int, name2int, fqid2int, room2int

# pd.set_option("display.max_columns", None)
# pd.set_option("display.max_rows", 200)

# %%
class PSPDataset(Dataset):
    def __init__(self, data_xy, cfg):
        self.X_base, self.y_base = data_xy
        self.t_window = cfg.T_WINDOW
        
        # Specify features to use
        self.num_feats = [feat for feat in cfg.FEATS if feat not in cfg.CAT_FEATS]
        self.cat_feats = [feat for feat in cfg.FEATS if feat in cfg.CAT_FEATS]

        # Setup level-related metadata
        # self.lv_gp = self.X_base["level_group"].unique()[0]

        # Generate data samples
        self._chunk_X_y()

    def __len__(self):
        return len(self.X_num)

    def __getitem__(self, idx):
        data_sample = {
            "X_num": torch.tensor(self.X_num[idx], dtype=torch.float32),
            "X_cat": torch.tensor(self.X_cat[idx], dtype=torch.int32),
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
            "mask": torch.tensor(self.mask[idx], dtype=torch.bool)
        }

        return data_sample

    def _chunk_X_y(self):
        """Chunk data samples."""
        X_num, X_cat, y = [], [], []
        # debug: list(self.X_base.groupby('session_id'))
        for sess_id, X_sess in self.X_base.groupby('session_id'):
            pad_len = max(self.t_window - X_sess.shape[0], 0)
            x_num = X_sess[self.num_feats].values[-self.t_window:] # keep the most recent
            x_cat = X_sess[self.cat_feats].values[-self.t_window:]
            if pad_len != 0:
                # https://blog.csdn.net/qq_34650787/article/details/80500407
                # pad at the beginning
                x_num = np.pad(x_num, ((pad_len, 0), (0, 0)), "constant", constant_values=(0,0))
                x_cat = np.pad(x_cat, ((pad_len, 0), (0, 0)), "constant", constant_values=-1)

            X_num.append(x_num)
            X_cat.append(x_cat)

            y_sess = self.y_base.loc[sess_id].values
            y.append(y_sess)

        self.X_num = np.stack(X_num)  # (N, P, C)
        self.X_cat = np.stack(X_cat)  # (N, P, M)
        self.y = np.vstack(y)  # (N, Q)
        self.mask = np.all(self.X_cat==-1, axis=-1)
        # mask = torch.all(x_cat==-1, dim=-1)


# %%
def build_dataloaders(data_tr, data_val, cfg):
    if data_tr is not None:
        train_loader = DataLoader(
            PSPDataset(data_tr, cfg),
            batch_size=cfg.BATCH_SIZE,
            shuffle=True,
            num_workers=cfg.NUM_WORKERS,
            collate_fn=None,
        )
    else:
        train_loader = None

    if data_val is not None:
        val_loader = DataLoader(
            PSPDataset(data_val, cfg),
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,  # Hard-coded
            num_workers=cfg.NUM_WORKERS,
            collate_fn=None,
        )
    else:
        val_loader = None

    return train_loader, val_loader


# %%
# if __name__ == '__main__':
# from config import event2int, name2int, fqid2int, room2int
#     class CFG:
#         # Question, one model per question
#         Q = 1
#         LEVEL_GRP = '0-4'

#         # ==Mode==
#         # Specify True to enable model training
#         train = True
        
#         # ==Data===
#         # FEATS = ["et_diff", "event_comb_code", "room_fqid_code", "page_code", "text_fqid_code", "level_code"]
#         # CAT_FEATS = ["event_comb_code", "room_fqid_code", "page_code","text_fqid_code", "level_code"]
#         # COLS_TO_USE = ["session_id", "index", "level", "level_group", "elapsed_time", 
#                         # "event_name", "name", "room_fqid", "page", "text_fqid"]

#         NUM_FEATS = ["elapsed_time_log1p", "elapsed_time_diff_log1p", "room_coor_x", "room_coor_y"]
#         CAT_FEATS = ["event_name", "name", "fqid", "room_fqid", "level"]
#         FEATS = NUM_FEATS + CAT_FEATS

#         # T_WINDOW = 512
#         T_WINDOW = 256
        
#         # ==Training==
#         SEED = 42
#         DEVICE = "cuda:0"
#         EPOCH = 70
#         CKPT_METRIC = "f1"
#         # CKPT_METRIC = "f1@0.63"

#         # ==DataLoader==
#         # BATCH_SIZE = 128
#         BATCH_SIZE = 3
#         # NUM_WORKERS = 4
#         NUM_WORKERS = 2

#         # ==Solver==
#         LR = 1e-3
#         WEIGHT_DECAY = 1e-2

#         # ==Early Stopping==
#         ES_PATIENCE = 20

#         # ==Evaluator==
#         EVAL_METRICS = ["auroc", "f1"]


#     cfg = CFG()
#     # seed_all(cfg.SEED)

#     # Read debug data
#     INPUT_PATH = "../input/how-to-get-32gb-ram/train.parquet"
#     TARGET_PATH = "../input/how-to-get-32gb-ram/train_labels.parquet"
#     df = pl.read_parquet(INPUT_PATH, n_rows=10000).to_pandas()
#     df = df.set_index('session_id')
#     df["elapsed_time_log1p"] = df["elapsed_time"].apply(lambda x: np.log1p(x))
#     df["elapsed_time_diff_log1p"] = df.groupby(["session_id", "level_group"], sort=False)\
#                                       .apply(lambda x: np.log1p(x["elapsed_time"].diff().fillna(0).clip(0, 3.6e6))).values

#     targets = pd.read_parquet(TARGET_PATH)
#     targets['session'] = targets.session_id.apply(lambda x: int(x.split('_')[0]))
#     targets['q'] = targets.session_id.apply(lambda x: int(x.split('_')[-1][1:]))

#     # One model per question
#     X_lvl = df.query("level_group=='0-4'")
#     X_lvl['fqid'] = X_lvl['fqid'].fillna('fqid_None')
#     X_lvl['room_coor_x'] = X_lvl['room_coor_x'].fillna(0.0)
#     X_lvl['room_coor_y'] = X_lvl['room_coor_y'].fillna(0.0)

#     X_lvl['event_name'] = X_lvl['event_name'].apply(lambda x: event2int[x])
#     X_lvl['name'] = X_lvl['name'].apply(lambda x: name2int[x])
#     X_lvl['fqid'] = X_lvl['fqid'].apply(lambda x: fqid2int[x])
#     X_lvl['room_fqid'] = X_lvl['room_fqid'].apply(lambda x: room2int[x])
#     y_q = targets.loc[targets.q==1].set_index('session')[['correct']]  # Suppose build model for question 1
#     display(X_lvl.index.value_counts())
#     print(y_q.shape)

#     sess_tr = [20090314221187252, 20090313571836404]  # len: 210, 112
#     sess_val = [20090313091715820, 20090314441803444] # len: 176, 107
#     X_tr, X_val = X_lvl.loc[sess_tr], X_lvl.loc[sess_val]
#     y_tr, y_val = y_q.loc[sess_tr], y_q.loc[sess_val]

#     X_tr, X_val = X_tr.reset_index(), X_val.reset_index()

#     # Test data loader
#     train_loader, val_loader = build_dataloaders((X_tr, y_tr), (X_val, y_val), cfg)

#     # list(val_loader)
#     for vv in val_loader:
#         print(vv)
