import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from scipy.integrate import solve_ivp
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from utils.utils import DATA_PATH, MODELS_PATH, RESULTS_PATH
    mo.md("# EGFR Pathway Simulation")
    return DATA_PATH, mo, np, pd, plt


@app.cell(disabled=True)
def _():
    #df = pd.read_csv(DATA_PATH / 'data_transient.csv', index_col=False)
    #df['y'] = df['cnr_norm']
    #df['time'] = df['frame']
    #df['group'] = df['uid'].astype('str') + df['stim_exposure'].astype('str')
    #df.drop(axis=1, columns=df.columns.difference(['y','time','group']), inplace=True)
    #df.to_csv(DATA_PATH / 'data_transient_v2.csv', index=False)
    return


@app.cell
def _(DATA_PATH, pd):
    pqt =  pd.read_parquet(DATA_PATH / 'exp_data.parquet')
    return (pqt,)


@app.cell(hide_code=True)
def _(mo, plt, pqt):
    df_og = pqt.copy()
    df_og["cnr"] = df_og["mean_intensity_C1_ring"] / df_og["mean_intensity_C1_nuc"]
    df_og["uid"] = df_og["fov"].astype("string") + "_" + df_og["particle"].astype("string")
    df_og['cell_id'] = df_og['cell_line'].astype(str) + '_' + df_og['stim_exposure'].astype(str) + 'ms_' + df_og['uid'].astype(str)
    df_og["frame"] = df_og["timestep"]

    _frame_counts = df_og["uid"].value_counts()
    _threshold = 0.9 * _frame_counts.max()

    _valid_uids = _frame_counts[_frame_counts >= _threshold].index
    df_og = df_og[df_og["uid"].isin(_valid_uids)]

    _NORM_UNTIL_TIMEPOINT = 10
    _mean_cnr_first_four_frames = df_og[df_og['frame'] < _NORM_UNTIL_TIMEPOINT].groupby('uid')['cnr'].mean()
    df_og['cnr_norm'] = df_og.apply(lambda row: row['cnr'] / _mean_cnr_first_four_frames[row['uid']], axis=1)
    df_og["stim_timestep_str"] = df_og["stim_timestep"].apply(str) 
    # Compute frame-to-frame differences
    df_og['diff'] = df_og.groupby('uid')['cnr'].diff().abs()
    # Drop first frame per UID (NaN in diff)
    df_og = df_og.dropna(subset=['diff'])
    # Compute mean absolute difference per UID
    df_og = df_og[df_og['cell_line'] == 'EGFR']
    df_og['mean_diff'] = df_og.groupby('uid')['diff'].transform('mean')
    # Define a _threshold (e.g., remove top 0.02% fluctuating cells)
    _threshold = df_og['mean_diff'].quantile(0.998)
    df_og = df_og[df_og['mean_diff'] < _threshold]
    df_og = df_og[df_og['cell_line'] == 'EGFR']
    # get fraction from ratio, normalize, and assign to y
    frac = df_og['cnr_norm'] / (df_og['cnr_norm'] + 1)
    df_og['y'] = (frac - frac.min()) / (frac.max() - frac.min())

    df_og['time'] = df_og['frame']
    df_og['group'] = df_og['stim_exposure'].astype('int')
    df_og.drop(axis=1, columns=df_og.columns.difference(['y','time','group']), inplace=True)
    df_og_toplot = df_og[df_og['group'] == 200].groupby('time', as_index=False, sort=True).median('y')
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    ax.plot('time','y' , data=df_og_toplot )
    mo.hstack([df_og,fig], widths=[0.5,0.5], gap="1rem")
    #df_og_toplot
    #df_og
    return


@app.cell
def _(mo, plt, pqt):
    df = pqt.copy()
    df = df[df['cell_line'] == 'EGFR']
    #df = df[df['stim_exposure'] == 200]


    df["cnr"] = df["mean_intensity_C1_ring"] / df["mean_intensity_C1_nuc"]

    df['fraction'] = df["mean_intensity_C1_ring"] / (df["mean_intensity_C1_nuc"] + df["mean_intensity_C1_ring"])
    df["uid"] = df["fov"].astype("string") + "_" + df["particle"].astype("string")
    df['cell_id'] = df['cell_line'].astype(str) + '_' + df['stim_exposure'].astype(str) + 'ms_' + df['uid'].astype(str)
    df["frame"] = df["timestep"]

    _frame_counts = df["uid"].value_counts()
    _threshold = 0.9 * _frame_counts.max()

    _valid_uids = _frame_counts[_frame_counts >= _threshold].index
    df = df[df["uid"].isin(_valid_uids)]
    _NORM_UNTIL_TIMEPOINT = 10
    _mean_cnr_first_four_frames = df[df['frame'] < _NORM_UNTIL_TIMEPOINT].groupby('uid')['fraction'].mean()

    df['frac_norm'] = df.apply(lambda row: row['fraction'] / _mean_cnr_first_four_frames[row['uid']], axis=1)
    df['frac_norm_sub'] = df.apply(lambda row: row['fraction'] - _mean_cnr_first_four_frames[row['uid']], axis=1)
    df["stim_timestep_str"] = df["stim_timestep"].apply(str) 
    # Compute frame-to-frame differences
    df['diff'] = df.groupby('uid')['fraction'].diff().abs()
    # Drop first frame per UID (NaN in diff)
    df = df.dropna(subset=['diff'])
    # Compute mean absolute difference per UID
    df['mean_diff'] = df.groupby('uid')['diff'].transform('mean')
    # Define a _threshold (e.g., remove top 0.02% fluctuating cells)
    _threshold = df['mean_diff'].quantile(0.998)
    df = df[df['mean_diff'] < _threshold]

    # get fraction from ratio, normalize, and assign to y
    _frac = df['frac_norm'] 
    df['y'] = (_frac - _frac.min()) / (_frac.max() - _frac.min())
    df['group'] = df['stim_exposure'].astype('int')

    df['time'] = df['frame']
    #df.drop(axis=1, columns=df.columns.difference(['y','time','group', 'fraction', 'frac_norm', 'frac_norm_sub']), inplace=True)

    _df_toplot = df.groupby('time', as_index=False, sort=True).median('y')
    _fig, _ax = plt.subplots(figsize=(8, 5), dpi=120)

    _ax.plot('time','y' , data=_df_toplot )
    mo.vstack([_fig, df],  gap="1rem")
    return


@app.cell
def _(mo, plt, pqt):
    ex = pqt.copy()

    # pick the right columns (adjust names if needed)
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

    # merge baseline stats back
    ex = ex.merge(baseline, on="uid", how="left")

    # normalized variants
    ex["frac_div"] = ex["fraction"] / (ex["baseline_mean"] + eps)              # fold-change
    ex["frac_sub"] = ex["fraction"] - ex["baseline_mean"]                      # delta
    ex["frac_pct"] = (ex["fraction"] - ex["baseline_mean"]) / (ex["baseline_mean"] + eps) * 100
    ex["frac_z"]   = (ex["fraction"] - ex["baseline_mean"]) / ex["baseline_std"]

    # group column from stim exposure
    ex["group"] = ex["stim_exposure"].astype(int)

    ex.drop(axis=1, columns=ex.columns.difference(['time','group', 'fraction', 'frac_div', 'frac_sub', 'frac_pct', 'frac_z']), inplace=True)

    fullnames = ['frac_'+_i for _i in ['div','sub','pct', 'z']]

    ex['final_norm'] = ex['frac_sub'] / max(ex['frac_sub'])

    _ex_toplot = ex[ex['group'] == 0].groupby('time', as_index=False, sort=True).mean(fullnames)

    _fig, _ax = plt.subplots(figsize=(8, 5), dpi=120)
    _ax.plot('time','final_norm', data=_ex_toplot)
    mo.vstack([_fig, ex ], gap="1rem")



    return


@app.cell
def _(np):
    from models.simple_EGFR_transient import light_func
    x= np.linspace(1,60,120)
    y = np.array([light_func(xx, {'group':50}) for xx in x])
    return


if __name__ == "__main__":
    app.run()
