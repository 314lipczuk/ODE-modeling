import marimo

__generated_with = "0.15.2"
app = marimo.App(
    width="medium",
    layout_file="layouts/presentation.slides.json",
)


@app.cell
def _(mo):
    mo.md(
        r"""
    # ODE modelling of MAPK/ERK

    Przemysław Pilipczuk
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    from scipy.integrate import solve_ivp
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from utils.utils import DATA_PATH, MODELS_PATH, RESULTS_PATH
    from models.simple_EGFR_transient import m, y0 as models_y0, light_func, nodes as param_list, nodes as states 
    import os
    import pathlib
    mo.md(
        r"""
    ### Introduction: My goals for this project

    - Learn the practicalities of math modeling in general
    - And ODE modelling in specific
    - Build the understanding of the process of building, validating, and iterating on a model.
    """
    )
    return (
        DATA_PATH,
        RESULTS_PATH,
        light_func,
        m,
        mo,
        np,
        os,
        pd,
        plt,
        solve_ivp,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    ### What is modeling?

    - When trying to understand a system, there's only so much we can do by passive observation.
    - The path of understanding is to generate testable predictions, and validating them on real world data
    - In here, the model is a computational scheme and its parameters, that we want to tune such that it can reproduce experimental data.
    - "What I can't create, I don't understand" - R.F
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Modeling: Constraints and Assumptions

    - So what can we do?
        - We try our best to isolate and account for things that we can
        - And for what we cannot, we acknowledge a limitation and include a assumption that we have when using this model.
    - The goal then becomes not to have a perfect model that represents reality (we waved that one goodbye, since it's not feasible), but to gain some insight into the phenomenon we're studying.
    - "All models are wrong, but some are useful" - George Box
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Modeling: Constraints and Assumptions

    - Researching an entire complex system at once is often impossible to do
    - Why? Physical limitations of technology we have (both on the side of accurate collection of data and computational resources)
    - So we narrow down, and investigate components
    - PROBLEM: there are no neat abstraction layers / modules / interfaces in biology
        - Everything is connected to everything else
        - Doesn't it make it impossible to 'Narrow down and investigate components?'
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Key Modeling Assumptions
    ## Making the Complex Tractable

    **Spatial Homogeneity:**
    - Perfect mixing assumption (well-mixed system)
    - No spatial gradients considered
    - Representative of bulk cellular measurements

    **Conserved Moieties:**
    - Total enzyme concentrations constant
    - Mass conservation constraints applied
    - Reduced parameter space through transformation
    """
    )
    return


@app.cell
def _(m, mo):
    mo.hstack([mo.md(
        r"""
    ### Ordinary Differential Equations
    ###### What are they? How are they useful to us?
    - Mathematical framework describing how state variables change over time
    - Relate derivatives (rates of change) to current system states
    - Perfect for modeling biochemical reaction networks

    **Why ODEs for EGFR Pathway?**

    - Capture temporal dynamics of protein concentrations
    - Model reaction kinetics using mass action principles
    - Enable parameter estimation from time-series data

    """
    ), mo.vstack([m.eqs ])])
    return


@app.cell
def _(mo):
    mo.vstack([mo.md(
        r"""
    ### Solving ODE Systems: Analytical vs Numerical Approaches


    """
    ), mo.hstack([
        mo.md(r"""
    **Analytical Solutions** (when possible):

    - Exact mathematical solutions
    - Limited to simple, linear systems
    - Rarely feasible for complex biological<br>networks
            """),
        mo.md(r"""
    **Numerical Integration** (our approach):

    - Approximate solutions using computational methods
    - `scipy.integrate.solve_ivp()` for robust solving
    - Handles non-linear, coupled equations effectively
        """),

    ], justify='space-between', gap=5),
              ], gap=1)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Ordinary Differential Equations
    ## Solving: Analytical vs Numerical
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Ordinary Differential Equations
    ## Fitting
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Ordinary Differential Equations
    ## Why ODEs and not other differential equations?
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # My approach
    ## Basics

    • Equations
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # My approach
    ## Assumptions in practice

    • Spatial aspect (assumption of perfect mixing)
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # My approach
    ## Assumptions in practice

    • Spatial aspect
    • Baseline values
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### My approach: Assumptions in practice

    • Conserved moieties
    • Fits over summary statistic (losing variance in the process)
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # My approach
    ## Transformed model showcase

    • Conserved moieties
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # My approach
    ## Simulation presentation
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # My approach
    ## Light function
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""### My approach to fitting""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # My approach
    ## General API, ?????
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Results
    ## Single experiment fits

    • 2 images : one of original data, one of the simulation learned from it, or 4/6/8, depending on the amount of experiments I end up working with.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Results
    ## Cross-validation

    • Trained on n-1, tested on 1 -> for each experiment
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""# EGFR Pathway Simulation""")
    return


@app.cell
def _(DATA_PATH, pd):
    df = pd.read_csv(DATA_PATH / 'data_transient_v2.csv', index_col=False)
    return (df,)


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
def _(RESULTS_PATH, mo, os):
    #def param_reader():
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
def _():
    y0 = [0.01, 0.01, 0.01, 0.01, 0.01]
    return (y0,)


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
    print(sol)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)

    if plot_original_widget.value:
        sns.lineplot(data=df, x='frame', y='cnr_norm', estimator="median")

    for i, name in enumerate(m.active_states):
        if name in plot_nodes.value:
            ax.plot(sol.t, sol.y[i], label=name)

    if input_plot_widget.value:
        ax.plot(t_vals, [light_func(t, {'group':50}) for t in t_vals], label="Input")

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


if __name__ == "__main__":
    app.run()
