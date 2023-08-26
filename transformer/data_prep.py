# %%
import pandas as pd
import polars as pl
import numpy as np

from config import event2int, name2int, fqid2int, room2int

INPUT_PATH = "../input/how-to-get-32gb-ram/train.parquet"
TARGET_PATH = "../input/how-to-get-32gb-ram/train_labels.parquet"

feats = [
    (
        (pl.col("elapsed_time") - pl.col("elapsed_time").shift(1))
         .fill_null(0)
         .clip(0, 3.6e6)  # fix data issue
         .over(["session_id", "level_group"])
         .alias("elapsed_time_diff")
    ),
    pl.col("room_coor_x").fill_null(0.0),
    pl.col("room_coor_y").fill_null(0.0),
    pl.col("fqid").fill_null("fqid_None"),
    pl.col("text_fqid").fill_null("text_fqid_None"),
]

# %%
def feature_engineering(feats_path=None, debug=True):
    if feats_path:
        X = pd.read_parquet(feats_path)
    else:
        df = pl.read_parquet(INPUT_PATH, n_rows=10000 if debug else None)
        df = df.drop(["fullscreen", "hq", "music"]).with_columns(feats)
        df = df.to_pandas()
        df = df.set_index('session_id')

        df["elapsed_time_log1p"] = df["elapsed_time"].apply(lambda x: np.log1p(x))
        df["elapsed_time_diff_log1p"] = df["elapsed_time_diff"].apply(lambda x: np.log1p(x))
        # df["elapsed_time_diff_log1p"] = df.groupby(["session_id", "level_group"], sort=False)\
                                        #   .apply(lambda x: np.log1p(x["elapsed_time"].diff().fillna(0).clip(0, 3.6e6))).values
        # Categorical feats
        df['event_name'] = df['event_name'].apply(lambda x: event2int[x])
        df['name'] = df['name'].apply(lambda x: name2int[x])
        df['fqid'] = df['fqid'].apply(lambda x: fqid2int[x])
        df['room_fqid'] = df['room_fqid'].apply(lambda x: room2int[x])
        X = df

    # Labels
    targets = pd.read_parquet(TARGET_PATH).rename({'session_id': 'session_id_q'}, axis=1)
    targets['session_id'] = targets.session_id_q.apply(lambda x: int(x.split('_')[0]))
    targets['q'] = targets.session_id_q.apply(lambda x: int(x.split('_')[-1][1:]))
    y = targets.pivot(index='session_id', columns='q', values='correct')

    # display(y.isna().sum())

    return X, y

