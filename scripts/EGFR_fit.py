import marimo

__generated_with = "0.14.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy

    df = pd.read_parquet('exp_data.parquet')
    df["cnr"] = df["mean_intensity_C1_ring"] / df["mean_intensity_C1_nuc"]
    df["uid"] = df["fov"].astype("string") + "_" + df["particle"].astype("string")
    df['cell_id'] = df['cell_line'].astype(str) + '_' + df['stim_exposure'].astype(str) + 'ms_' + df['uid'].astype(str)
    df["frame"] = df["timestep"]
    frame_counts = df["uid"].value_counts()
    threshold = 0.9 * frame_counts.max()

    valid_uids = frame_counts[frame_counts >= threshold].index
    df = df[df["uid"].isin(valid_uids)]
    NORM_UNTIL_TIMEPOINT = 10
    mean_cnr_first_four_frames = df[df['frame'] < NORM_UNTIL_TIMEPOINT].groupby('uid')['cnr'].mean()
    df['cnr_norm'] = df.apply(lambda row: row['cnr'] / mean_cnr_first_four_frames[row['uid']], axis=1)
    df["stim_timestep_str"] = df["stim_timestep"].apply(str) 
    # Compute frame-to-frame differences
    df['diff'] = df.groupby('uid')['cnr'].diff().abs()
    # Drop first frame per UID (NaN in diff)
    df = df.dropna(subset=['diff'])
    # Compute mean absolute difference per UID
    df['mean_diff'] = df.groupby('uid')['diff'].transform('mean')
    # Define a threshold (e.g., remove top 0.02% fluctuating cells)
    threshold = df['mean_diff'].quantile(0.998)
    df_filtered = df[df['mean_diff'] < threshold]
    # Flag cells that will be deleted
    df['is_deleted'] = df['mean_diff'] >= threshold
    # Display the rows that are deleted
    deleted_rows = df[df['is_deleted']]
    df
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Fitting
    At first I want to fit my model to the average for each stimulation time.
    So, I'm interested in normalized cnr for EGFR as my y. Let's extract that first?

    """
    )
    return


if __name__ == "__main__":
    app.run()
