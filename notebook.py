import marimo

__generated_with = "0.14.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from scipy.integrate import solve_ivp
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from utils.utils import DATA_PATH, MODELS_PATH
    mo.md("# EGFR Pathway Simulation")
    return DATA_PATH, mo, np, pd, plt, solve_ivp


@app.cell
def _(DATA_PATH, pd):
    from models.simple_EGFR_transient import m, y0, light_func
    df = pd.read_csv(DATA_PATH / 'data_transient_v2.csv', index_col=False)
    return df, light_func, m, y0


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
def _(df):
    df
    return


@app.cell
def _(mo):

    plot_original_widget = mo.ui.switch(label="Plot original data?")
    return (plot_original_widget,)


@app.cell
def _(DATA_PATH, mo):
    #def param_reader():
    import os
    import pathlib
    candidate_params = list(reversed([c for c in os.listdir(DATA_PATH) if c.endswith('.json')]))
    param_widget = mo.ui.radio(options=candidate_params, value=candidate_params[0])
    param_widget
    return (param_widget,)


@app.cell
def _(DATA_PATH, param_widget):
    #from simulation import param_defaults
    import json


    #dp = param_defaults.copy()
    with open(DATA_PATH / param_widget.value, 'r') as f:
        dp = json.load(f)
        test = dp.get('K12')
        if test is None:
            dp = dp.get("params")
    dp
    return (dp,)


@app.cell
def _(m):
    system = m.make_numerical()
    m.eqs
    return (system,)


@app.cell
def make_slider_node_widget(dp, m, mo):
    sliders = mo.ui.dictionary({
        p: mo.ui.slider(start=0.0, stop=5.0, step=0.01, value=dp[p], label=p)
        for p in m.parameters
    })
    plot_nodes = mo.ui.multiselect(
        options=m.active_states,
        value=["ERK_s"],
        label="Select state variables to plot"
    )

    return plot_nodes, sliders


@app.cell(hide_code=True)
def merge_params(m, mo, param_defaults, sliders):
    # cell: merge parameter values
    merged_param_values = [
        sliders[p].value if p in sliders else param_defaults[p]
        for p in m.parameters
    ]
    p_tbl = mo.ui.table(data=dict(zip(m.parameters, merged_param_values)))
    p_tbl


    return


@app.cell
def input(mo):
    mo.md("""# Inputs""")
    return


@app.cell(hide_code=True)
def input_plot(light_func, mo, np, plt):
    fig0, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    t_vals = np.linspace(1, 59, 100)
    ax1.plot(t_vals, [light_func(t, {'group':50}) for t in t_vals])
    ax1.set_title("light input")
    ax1.set_xlabel("time")
    ax1.set_ylabel("light conc")
    ax1.grid(True)

    plt.show()

    plot_input = False
    input_plot_widget = mo.ui.switch(label="Plot input?")

    return input_plot_widget, t_vals


@app.cell
def _():
    return


@app.cell
def _(m, np, sliders):

    p_full = np.zeros(len(m.parameters))
    param_names = m.parameters
    # Start with provided p0 values
    for iP, param_name in enumerate(m.parameters):
        p_full[iP] = sliders[param_name].value

    return (p_full,)


@app.cell
def _(m):
    m.active_states
    return


@app.cell
def simulation(
    df,
    input_plot_widget,
    light_func,
    m,
    mo,
    p_full,
    plot_nodes,
    plot_original_widget,
    plt,
    sliders,
    solve_ivp,
    system,
    t_vals,
    y0,
):
    import seaborn as sns

    # def wrapped_system(t, y):
    #     return egfr_system(t, y, merged_param_values, light_input)


    sol = solve_ivp(
        lambda t, y: system(t, y, p_full, light_func, {'group':50}),
        (1, 59), y0, rtol=1e-4, atol=1e-7, t_eval=t_vals )

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)

    if plot_original_widget.value:
        sns.lineplot(data=df, x='frame', y='cnr_norm', estimator="median")

    for i, name in enumerate(m.active_states):
        if name in plot_nodes.value:
            ax.plot(sol.t, sol.y[i], label=name)

    if input_plot_widget.value:
        ax.plot(t_vals, [light_func(t) for t in t_vals], label="Input")

    ax.set_title("EGFR Pathway Simulation")
    ax.set_xlabel("Time")
    ax.set_ylabel("Concentration")
    ax.legend(loc=1)
    ax.grid(True)

    ctrls = mo.accordion({
        "State Variables to plot":plot_nodes,
        "Parameter values": sliders,
        "Others":[input_plot_widget, plot_original_widget]
    })
    mo.hstack([ctrls,fig], widths=[0.3,0.7], gap="1rem")

    return


if __name__ == "__main__":
    app.run()
