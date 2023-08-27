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
        _X_cat = np.stack(X_cat)  # (N, P, M)
        self.y = np.vstack(y)  # (N, Q)
        self.mask = np.all(_X_cat==-1, axis=-1)
        self.X_cat = _X_cat + 1


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

