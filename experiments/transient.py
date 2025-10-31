import numpy as np
from pathlib import PosixPath
from utils.utils import DATA_PATH
import pandas as pd

def light_func(t, rest=None):
  # Smooth transitions to avoid solver issues
  delta_t = 0.2
  assert type(rest) == None or type(rest) == dict, f"rest is not what it should be: {rest}"
  if (10-delta_t) < t < (10+delta_t): 
    modifier = float(rest['group'])
    if modifier == 0: return 0
    log_modifier = np.log(modifier)
    return log_modifier - log_modifier * (np.abs(10-t)/delta_t)
  else:
    return 0

# if light func so good, why no light func 2?
def light_func2(t, rest=None):
  # Smooth transitions to avoid solver issues
  delta_t = 0.5
  assert type(rest) == None or type(rest) == dict, f"rest is not what it should be: {rest}"
  if (10-delta_t) < t < (10+delta_t): 
    modifier = float(rest['group'])
    if modifier == 0: return 0
    log_modifier = np.log(modifier)
    return (log_modifier - log_modifier * (np.abs(10-t)/delta_t)) / np.log(1000)  # normalizing by biggest possible value
  else:
    return 0

light_fn = light_func2

def read_parquet_and_clean(file, save_as=None):
    assert str(file).endswith('.parquet')
    assert (save_as is None) or (type(save_as) == str and save_as.endswith('.csv') or (type(save_as)==PosixPath and save_as.name.endswith('.csv')))
    ex = pd.read_parquet(file, )
    ex = ex[ex['cell_line'] == "EGFR"]

    ring = ex["median_intensity_C1_ring"].astype(float)
    nuc  = ex["median_intensity_C1_nuc"].astype(float)

    # unique cell id
    ex["uid"] = ex["fov"].astype(str) + "_" + ex["particle"].astype(str)

    # time
    ex["time"] = ex["timestep"]

    # compute fraction
    eps = 1e-12
    ex["fraction"] = ring / (ring + nuc + eps)

    # compute baseline mean and std per uid (frames < 10s)
    baseline = ex[ex["time"] < 10].groupby("uid")["fraction"].agg(["mean","std"]).reset_index()
    baseline.rename(columns={"mean":"baseline_mean","std":"baseline_std"}, inplace=True)
    baseline["baseline_std"] = baseline["baseline_std"].fillna(0.0) + eps

    # TODO: IDEA: remove cells that have too high variability in the first X frames?

    # merge baseline stats back
    ex = ex.merge(baseline, on="uid", how="left")

    # normalized variants
    ex["frac_sub"] = ex["fraction"] - ex["baseline_mean"]                      # delta

    ex["group"] = ex["stim_exposure"].astype(int)

    ex['y'] = ex.groupby('group')['frac_sub'].transform(lambda x: x / x.max())

    ex.drop(axis=1, columns=ex.columns.difference(['time','group', 'y' ]), inplace=True)

    if save_as is not None: ex.to_csv(DATA_PATH / save_as, index=False)

    return ex