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
    return DATA_PATH, RESULTS_PATH, mo, np, pd, plt, solve_ivp


@app.cell
def _(DATA_PATH, mo):

    data_options = [file.name for file in DATA_PATH.iterdir() if file.name.endswith('csv')]


    data_option_widget = mo.ui.radio(options=data_options, label="Pick which data version you use")
    data_option_widget
    return (data_option_widget,)


@app.cell
def _(data_option_widget):

    data_p =  dow if (dow := data_option_widget.value) is not None else 'data_transient_v3.csv'


    return (data_p,)


@app.cell
def _(DATA_PATH, data_p, pd):
    from models.simple_EGFR_transient import m, y0 as models_y0, light_func, nodes as param_list, nodes as states 
    df = pd.read_csv(DATA_PATH / data_p,  index_col=False)
    return df, light_func, m


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
    dfy = df.groupby(['group','time']).mean('y')
    dfy
    return


@app.cell
def _(mo):

    plot_original_widget = mo.ui.switch(label="Plot original data?")
    return (plot_original_widget,)


@app.cell
def _(RESULTS_PATH, mo):
    #def param_reader():
    import os
    import pathlib
    candidate_params = list(reversed([c for c in os.listdir(RESULTS_PATH) if c.endswith('.json')]))
    param_widget = mo.ui.radio(options=candidate_params, value=candidate_params[0])
    param_widget
    return (param_widget,)


@app.cell
def _(RESULTS_PATH, m, param_widget):
    #from simulation import param_defaults
    import json


    #dp = param_defaults.copy()
    with open(RESULTS_PATH/ param_widget.value, 'r') as f:
        dp = json.load(f)
        test = dp.get('K12')
        test2 = dp.get('fitted_params')
        if test is not None:
            p = dp
            dp = {k: p[k] for k in m.parameters} 
        if test is None and test2 is None:
            dp = dp.get("params")
        if test is None and test2 is not None:
            p = dp.get('fitted_params')
            dp = {k: p[k] for k in m.parameters} 
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
    mo.md(
        """
    # Inputs
    # $$\\frac{dy}{dx} = ...$$
    """
    )
    return


@app.cell(hide_code=True)
def input_plot(df, mo, np, plt):
    fig0, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    t_vals = np.linspace(1, 59, 100)
    #ax1.plot(t_vals, [light_func(t, {'group':light_intensity}) for t in t_vals])
    #ax1.set_title("light input")
    #ax1.set_xlabel("time")
    #ax1.set_ylabel("light conc")
    #ax1.grid(True)
    #
    #plt.show()

    plot_input = False
    input_plot_widget = mo.ui.switch(label="Plot input?")

    pick_light_intensity_to_plot = mo.ui.dropdown(options=df['group'].unique())
    return input_plot_widget, pick_light_intensity_to_plot, t_vals


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
def _():
    y0 = [0.01, 0.01, 0.01, 0.01, 0.01]
    return (y0,)


@app.cell
def _(df, mo, plt):
    toplot = df[df['group']==0].groupby('time', as_index=False,sort=True).median('y')
    toplot
    plt.ylim(0, 0.6)
    plt.plot('time', 'y', data=toplot )

    ylim_w = mo.ui.text(label='ylab thresh')

    return (ylim_w,)


@app.cell
def simulation(
    df,
    input_plot_widget,
    light_func,
    m,
    mo,
    p_full,
    pick_light_intensity_to_plot,
    plot_nodes,
    plot_original_widget,
    plt,
    sliders,
    solve_ivp,
    system,
    t_vals,
    y0,
    ylim_w,
):
    import seaborn as sns
    # def wrapped_system(t, y):
    #     return egfr_system(t, y, merged_param_values, light_input)
    light_intensity =  litp if (litp := pick_light_intensity_to_plot.value) is not None else 200

    sol = solve_ivp(
        lambda t, y: system(t, y, p_full, light_func, {'group':light_intensity}),
        (1, 59), y0, rtol=1e-4, atol=1e-7, t_eval=t_vals )
    print(sol)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)

    if (ylim := ylim_w.value) != "" and float(ylim):
        ax.set_ylim(top=float(ylim))
    


    for i, name in enumerate(m.active_states):
        if name in plot_nodes.value:
            ax.plot(sol.t, sol.y[i], label=name)

    if input_plot_widget.value:
        ax.plot(t_vals, [light_func(t, {'group':light_intensity}) for t in t_vals], label="Input")

    if plot_original_widget.value:
        grouped = df[df['group'] == light_intensity]
        sns.lineplot(data=grouped, x='time', y='y', estimator="median", label='Original data')

    print('light intensity:', light_intensity)
    ax.set_title(f"EGFR Pathway Simulation (param={light_intensity})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Concentration")
    ax.legend(loc=1)
    ax.grid(True)

    ctrls = mo.accordion({
        "State Variables to plot":plot_nodes,
        "Others":[input_plot_widget, plot_original_widget, pick_light_intensity_to_plot, ylim_w],
        "Parameter values": sliders,
    })
    mo.hstack([ctrls,fig], widths=[0.3,0.7], gap="1rem")
    return (sol,)


@app.cell
def _():
    return


@app.cell
def _(sol):
    sol.t.shape
    return


@app.cell
def _(pd, sol):
    def sol_to_df(sol, state_names):
        """Convert solve_ivp solution to wide dataframe"""
        # Create dataframe with time column
        df = pd.DataFrame({'time': sol.t})
        print(df.shape) 
        # Add each state as a column
        for i, state in enumerate(state_names):
            df[state] = sol.y[i,:]

        return df


    # Usage:
    state_names = ['RAS_s', 'RAF_s', 'MEK_s',  'NFB_s', 'ERK_s']
    df_sol = sol_to_df(sol, state_names)
    df_sol
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
