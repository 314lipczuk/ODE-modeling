import pandas as pd
from pathlib import Path, PosixPath
import numpy as np
from utils.utils import DATA_PATH

def ramp_light_fn_linear(t, rest=None):
    if t < 10 or t > 150: return 0
    r = (t-10) / 140 * 6
    # TODO: parametrize this, make it take values from context, not a magic number.
    return  r / 5.98

light_fn = ramp_light_fn_linear

def read_parquet_and_clean(file, save_as=None):
    assert str(file).endswith('.parquet')
    assert (save_as is None) or (type(save_as) == str and save_as.endswith('.csv') or (type(save_as)==PosixPath and save_as.name.endswith('.csv')))
    ex = pd.read_parquet(file, )
    ex = ex[ex['cell_line'] == "EGFR"]

    # unique cell id
    ex["uid"] = ex["fov"].astype(str) + "_" + ex["particle"].astype(str)

    # time
    ex["time"] = ex["timestep"]

    # completeness filter: keep UIDs with near-full trajectories
    frame_counts = ex["uid"].value_counts()
    if len(frame_counts) > 0:
        threshold = int(np.floor(0.9 * frame_counts.max()))
        valid_uids = frame_counts[frame_counts >= threshold].index
        ex = ex[ex["uid"].isin(valid_uids)].copy()

    # compute fraction (recompute ring/nuc AFTER filtering to keep index aligned)
    eps = 1e-12
    ring = ex["median_intensity_C1_ring"].astype(float)
    nuc  = ex["median_intensity_C1_nuc"].astype(float)
    ex["fraction"] = ring / (ring + nuc + eps)

    # compute baseline mean and std per uid using frames strictly before 10s
    baseline = (
        ex[ex["time"] < 10]
          .groupby("uid")["fraction"]
          .agg(["mean", "std"]).reset_index()
    )
    baseline.rename(columns={"mean": "baseline_mean", "std": "baseline_std"}, inplace=True)
    baseline["baseline_std"] = baseline["baseline_std"].fillna(0.0) + eps

    # TODO: IDEA: remove cells that have too high variability in the first X frames?

    # merge baseline stats back and DROP cells without a pre-10s baseline
    ex = ex.merge(baseline, on="uid", how="left")
    ex = ex[ex["baseline_mean"].notna()].copy()

    # normalized variants
    ex["frac_sub"] = ex["fraction"] - ex["baseline_mean"]                      # delta

    ex["group"] = 1
    ex["stim_exposure"] = ex["stim_exposure"].astype(int)

    # robust normalization within (group, stim_exposure) and NaN handling
    denom = ex.groupby(['group', 'stim_exposure'])["frac_sub"].transform('max')
    denom = denom.replace([0, np.inf, -np.inf], np.nan)
    ex['y'] = ex['frac_sub'] / denom
    ex = ex.dropna(subset=['y'])

    if save_as is not None: ex.to_csv(DATA_PATH / save_as, index=False)

    return ex

    
