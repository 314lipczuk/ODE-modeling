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
def make_param_widget(mo, param_list):
    # cell: param selector
    selected_params = mo.ui.multiselect(
        options=param_list,
        label="Select parameters to adjust",
        value=[]
    )


    return (selected_params,)


@app.cell
def make_slider_node_widget(mo, nodes, selected_params):
    sliders = mo.ui.dictionary({
        p: mo.ui.slider(start=0.0, stop=2.0, step=0.01, value=1.0, label=p)
        for p in selected_params.value
    })
    plot_nodes = mo.ui.multiselect(
        options=nodes,
        value=["ERK_s"],
        label="Select state variables to plot"
    )

    return plot_nodes, sliders


@app.cell(hide_code=True)
def merge_params(mo, param_list, sliders):
    # cell: merge parameter values
    from simulation import param_defaults

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
    light_input = lambda t: 1 if 3 < t <5 else 0.2  
    return (light_input,)


@app.cell(hide_code=True)
def input_plot(light_input, plt, t_vals):
    fig0, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))


    ax1.plot(t_vals, [light_input(t) for t in t_vals])
    ax1.set_title("light input")
    ax1.set_xlabel("time")
    ax1.set_ylabel("light conc")
    ax1.grid(True)

    plt.show()

    return


@app.cell
def show_widgets(mo, plot_nodes, selected_params, sliders):
    mo.vstack([selected_params,sliders,plot_nodes])
    return


@app.cell
def _(eqs):
    eqs
    return


@app.cell
def _(y0):
    y0
    return


@app.cell(hide_code=True)
def simulation(
    egfr_system,
    light_input,
    merged_param_values,
    plot_nodes,
    plt,
    solve_ivp,
    state_vars,
    t_vals,
    y0,
):
    def wrapped_system(t, y):
        return egfr_system(t, y, merged_param_values, light_input)


    sol = solve_ivp(wrapped_system, (0, 30), y0, t_eval=t_vals)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, name in enumerate([sv[:-3] for sv in state_vars]):
        if name in plot_nodes.value:
            ax.plot(sol.t, sol.y[i], label=name)

    ax.set_title("EGFR Pathway Simulation")
    ax.set_xlabel("Time")
    ax.set_ylabel("Concentration")
    ax.legend()
    ax.grid(True)
    #mo.ui.plotly(fig)
    #fig.show()
    plt.show()


    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # General things that need doing:
    ### Parameter estimation
    Current values are semi-random, with little thought put into them being instructive. It would be nice if i could assign units to them as sanity check.
    I need to do also find a library that will do fitting of these coefficients based on data I have.

    ### Weird behavior of plot
    The plot right now exhibits one problematic behavior: each subsequent node in the network shows weaker activation, where I'd expect it to exhibit a stronger one (RAS has a very steep activation, whereas RAF, MEK, ERK are subsequently weaker). This one requires checking with literature, as I'm not sure if this is really how it should behave.

    ### Resolution idea
    It's possible that two of the above are connected in a way. If I fit actual data from real cells, maybe the real parameters will also fix the diminishing steepness. 
    """
    )
    return


if __name__ == "__main__":
    app.run()
