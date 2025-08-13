import marimo

__generated_with = "0.14.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy
    import seaborn as sns

    # mmmm spghetti 
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
    return df, np, plt, sns


@app.cell
def _(df):
    df
    return


@app.cell
def _(df):
    df['cell_line'].unique()
    return


@app.cell
def _(df, plt, sns):
    STIM_TIMESTEP_TO_PLOT = "[10]"
    df_plot = df.query("stim_timestep_str == @STIM_TIMESTEP_TO_PLOT")
    cell_lines = df_plot["cell_line"].unique()
    df_plot.loc[:, "stim_exposure"] = df_plot.loc[:, "stim_exposure"].astype("int")

    cell_line = 'EGFR'
    plt.figure(figsize=(6, 3),dpi = 250)
    num_cells = df_plot.query("cell_line == @cell_line")["uid"].nunique()

    sns.lineplot(data=df_plot.query("cell_line == @cell_line"), x='frame', y='cnr_norm', hue="stim_exposure", palette="tab10", estimator="median", errorbar=('ci',90))
    handles, labels = plt.gca().get_legend_handles_labels()
    new_labels = []
    for label in labels:
        exposure_time = int(float(label))
        num_cells_exposure = df_plot.query("cell_line == @cell_line and stim_exposure == @exposure_time and stim_timestep_str == @STIM_TIMESTEP_TO_PLOT")["uid"].nunique()
        new_labels.append(f'{int(float(label))} ms (n={num_cells_exposure})')
    plt.legend(handles, new_labels, loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=3, frameon=False)
    plt.ylabel('norm. ERK-KTR c/n ratio')
    plt.xlabel('time [min]')
    plt.axvline(x=10, color='black', linestyle='--')
    plt.title(f"{cell_line} (n={num_cells})")
    plt.savefig(f"{cell_line}_single_erkktr_norm.svg", bbox_inches='tight')
    plt.show()
    return (df_plot,)


@app.cell
def _(df):
    df[df['stim']]['frame'].unique()
    return


@app.cell
def _(df_plot):
    to_model = df_plot[df_plot['cell_line'] == 'EGFR'].groupby(['frame', 'stim_exposure'])['cnr_norm'].median().reset_index()

    md1 = to_model[to_model['stim_exposure']==200]
    md1
    return (md1,)


@app.cell
def _(df_plot):
    df_plot[df_plot['cell_line'] == 'EGFR']['stim_exposure'].unique()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from symfit import Fit
    #from EGFR import to_model
    from simulation import eqs, t, initial_cond_defaults, nodes
    from fit import build_symfit_odemodel
    import sympy as sp
    return Fit, build_symfit_odemodel, eqs, initial_cond_defaults, sp, t


@app.cell
def _(sp, t):
    def box_pulse(t, start, end):
        H = sp.Heaviside
        return H(t - start) - H(t - end)  # 1 on [start, end), else 0

    frame_width = 1.0       # or your dt per frame
    t_start = 10.0 * frame_width
    t_end   = t_start + frame_width

    light_expr = box_pulse(t, t_start, t_end)

    return (light_expr,)


@app.cell
def _(build_symfit_odemodel, eqs, initial_cond_defaults, light_expr, sp, t):
    init = {k:v for (k,v) in initial_cond_defaults.items() if k.endswith('_s') }
    param_bounds = {
    }
    print('initial cond', init )
    ode, sv, po = build_symfit_odemodel(eqs, t, init,known_subs={sp.Function("light")(t):light_expr} )

    return (ode,)


@app.cell
def _(np):
    tn = np.linspace(start=1,stop=59, num=59)
    tn
    return (tn,)


@app.cell
def _(Fit, initial_cond_defaults, md1, ode, tn):

    known_data = {
        't':md1['frame'],
        'KTR_s' : md1['cnr_norm']
    }
    data = { k:known_data.get(k) for k in initial_cond_defaults.keys() if k.endswith('_s')}

    print(data)
    fit = Fit(ode, **data, t=tn)
    fit

    return (fit,)


@app.cell
def _(fit):

    e = fit.execute()
    return


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
