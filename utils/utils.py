from pathlib import Path, PosixPath
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_PATH = PROJECT_ROOT / 'data'
UTILS_PATH = PROJECT_ROOT / 'utils'
MODELS_PATH = PROJECT_ROOT / 'models'  
RESULTS_PATH = PROJECT_ROOT / 'results'

# Additional paths that might be useful
VENV_PATH = PROJECT_ROOT / 'venv'

def ensure_dirs_exist():
    """Create directories if they don't exist"""
    for path in [DATA_PATH, MODELS_PATH, RESULTS_PATH ]:
        path.mkdir(exist_ok=True, parents=True)

def get_data_file(filename):
    """Get path to a file in the data directory"""
    return DATA_PATH / filename

def get_results_file(filename):
    """Get path to a file in the results directory"""
    return RESULTS_PATH / filename

def get_model_file(filename):
    """Get path to a file in the models directory"""
    return MODELS_PATH / filename

def read_parquet_and_clean(file, save_as=None):
    assert str(file).endswith('.parquet')
    assert (save_as is None) or (type(save_as) == str and save_as.endswith('.csv') or (type(save_as)==PosixPath and save_as.name.endswith('.csv')))
    ex = pd.read_parquet(file, )
    ring = ex["mean_intensity_C1_ring"].astype(float)
    nuc  = ex["mean_intensity_C1_nuc"].astype(float)

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


# Print paths for debugging (can be removed later)
if __name__ == "__main__":
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"DATA_PATH: {DATA_PATH}")
    print(f"UTILS_PATH: {UTILS_PATH}")
    print(f"MODELS_PATH: {MODELS_PATH}")
    print(f"RESULTS_PATH: {RESULTS_PATH}")
    print()
    print("Path existence check:")
    for name, path in [("DATA", DATA_PATH), ("UTILS", UTILS_PATH), 
                       ("MODELS", MODELS_PATH), ("RESULTS", RESULTS_PATH),
                       ]:
        print(f"  {name}: {path.exists()}")

