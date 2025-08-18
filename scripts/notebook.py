import marimo

__generated_with = "0.14.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    from simulation import (
        egfr_system,
        param_list,
        state_vars,
        initial_cond_defaults,
        active_variants as nodes,
        eqs 
        #param_values
    )
    from scipy.integrate import solve_ivp
    import numpy as np
    import matplotlib.pyplot as plt
    mo.md("# EGFR Pathway Simulation")

    return (
        egfr_system,
        eqs,
        initial_cond_defaults,
        mo,
        nodes,
        np,
        param_list,
        plt,
        solve_ivp,
        state_vars,
    )


@app.cell
def _():
    from simulation import param_defaults
    import json


    #dp = param_defaults.copy()
    PARAM_FILE = 'egfr_fit_transient_1_params.json'
    with open(PARAM_FILE, 'r') as f:
        dp = json.load(f)
    dp
    return dp, param_defaults


@app.cell
def make_slider_node_widget(dp, mo, nodes, param_list):
    sliders = mo.ui.dictionary({
        p: mo.ui.slider(start=0.0, stop=4.0, step=0.01, value=dp[p], label=p)
        for p in param_list
    })
    plot_nodes = mo.ui.multiselect(
        options=nodes,
        value=["ERK_s"],
        label="Select state variables to plot"
    )

    return plot_nodes, sliders


@app.cell(hide_code=True)
def merge_params(mo, param_defaults, param_list, sliders):
    # cell: merge parameter values
    merged_param_values = [
        sliders[p].value if p in sliders else param_defaults[p]
        for p in param_list
    ]
    p_tbl = mo.ui.table(data=dict(zip(param_list, merged_param_values)))
    p_tbl


    return (merged_param_values,)


@app.cell(hide_code=True)
def _(initial_cond_defaults, np, state_vars):
    from math import sin
    y0 = [initial_cond_defaults[v[0:-3]] for v in state_vars]
    t_vals = np.linspace(0, 30, 300)

    return t_vals, y0


@app.cell
def input(mo):
    mo.md("# Inputs")
    #light_input = lambda t: 10 if 13 < t < 15 else 0
    light_input = lambda t: 300 if 10<t<11 else 0  
    return (light_input,)


@app.cell(hide_code=True)
def input_plot(light_input, mo, plt, t_vals):
    fig0, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))


    ax1.plot(t_vals, [light_input(t) for t in t_vals])
    ax1.set_title("light input")
    ax1.set_xlabel("time")
    ax1.set_ylabel("light conc")
    ax1.grid(True)

    plt.show()

    plot_input = False
    input_plot_widget = mo.ui.switch(label="Plot input?")

    return (input_plot_widget,)


@app.cell
def _(eqs):
    eqs
    return


@app.cell
def simulation(
    egfr_system,
    input_plot_widget,
    light_input,
    merged_param_values,
    mo,
    plot_nodes,
    plt,
    sliders,
    solve_ivp,
    state_vars,
    t_vals,
    y0,
):
    def wrapped_system(t, y):
        return egfr_system(t, y, merged_param_values, light_input)

    sol = solve_ivp(wrapped_system, (0, 30), y0, t_eval=t_vals)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    for i, name in enumerate([sv[:-3] for sv in state_vars]):
        if name in plot_nodes.value:
            ax.plot(sol.t, sol.y[i], label=name)

    if input_plot_widget.value:
        ax.plot(t_vals, [light_input(t) for t in t_vals], label="Input")

    ax.set_title("EGFR Pathway Simulation")
    ax.set_xlabel("Time")
    ax.set_ylabel("Concentration")
    ax.legend(loc=1)
    ax.grid(True)

    ctrls = mo.accordion({
        "State Variables to plot":plot_nodes,
        "Parameter values": sliders,
        "Plot input":input_plot_widget
    })
    mo.hstack([ctrls,fig], widths=[0.3,0.7], gap="1rem")

    return


if __name__ == "__main__":
    app.run()
