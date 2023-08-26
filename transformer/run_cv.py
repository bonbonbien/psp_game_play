# %%
import gc
import pandas as pd
import numpy as np
import polars as pl
from sklearn.model_selection import KFold, GroupKFold
import torch.nn as nn
from torch import optim

from config import CFG
from data_prep import feature_engineering
from data_loader import PSPDataset, build_dataloaders
from model import PspTransformerModel
from cust_trainer import CustTrainer
from evaluator import Evaluator

cfg = CFG()

# %%
df, targets = feature_engineering(debug=True)
print(df.shape, targets.shape)

X = df.query(f" level_group=='{cfg.LVL_GRP}' ")
ALL_USERS = X.index.unique()
# check seq_len of each session_id
# display(X.groupby('session_id').agg({'elapsed_time': 'count'}).describe())

# %%
oof_pred = pd.DataFrame(
    data=np.zeros((len(ALL_USERS), cfg.N_QNS)), 
    index=ALL_USERS, columns=[f'meta_{i}' for i in range(cfg.Q_ST, cfg.Q_ED+1)]
)

gkf = GroupKFold(n_splits=cfg.N_FOLD)
for i, (train_index, test_index) in enumerate(gkf.split(X=X, groups=X.index)):
    break
    print(f"Training and evaluation process of fold{i} starts...")
    # TRAIN DATA
    train_x = X.iloc[train_index]
    train_users = train_x.index.unique().values
    train_y = targets.loc[train_users, range(cfg.Q_ST, cfg.Q_ED+1)]
    
    # VALID DATA
    valid_x = X.iloc[test_index]
    valid_users = valid_x.index.unique().values
    valid_y = targets.loc[valid_users, range(cfg.Q_ST, cfg.Q_ED+1)]
    
     # Build dataloader
    train_loader, val_loader = build_dataloaders((train_x, train_y), (valid_x, valid_y), cfg)
    # tt = PSPDataset((train_x, train_y), cfg)
    # ? why val_load load from the beginning ?
    # for vv in val_loader:
    #     print(vv)

    # Build model
    model = PspTransformerModel(out_size=cfg.N_QNS)
    model.to(cfg.DEVICE)
    
    # Build criterion
    loss_fn = nn.BCEWithLogitsLoss()

    # Build solvers
    optimizer = optim.AdamW(list(model.parameters()), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    lr_skd = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=10, eta_min=1e-5, T_mult=1)

    # Build evaluator
    evaluator = Evaluator(cfg.EVAL_METRICS, cfg.N_QNS)

    # Build trainer
    trainer_cfg = {
        "cfg": cfg,
        "model": model,
        "loss_fn": loss_fn,
        "optimizer": optimizer,
        "lr_skd": lr_skd,
        "evaluator": evaluator,
        "train_loader": train_loader,
        "eval_loader": val_loader,
    }
    trainer = CustTrainer(**trainer_cfg)

    # Run training and evaluation processes for one fold
    best_model, best_preds = trainer.train_eval(proc_id=i)

    # Dump output objects of the current fold
    torch.save(best_model.state_dict(), f"model_fold{i}.pt")

    # Free mem.
    del (train_x, train_y, valid_x, valid_y, train_loader, val_loader,
         model, loss_fn, optimizer, lr_skd, evaluator, trainer)
    _ = gc.collect()


    oof_pred.loc[valid_users, :] = best_preds["val"].numpy()


