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
    from datetime import datetime
    from utils.utils import DATA_PATH, MODELS_PATH, RESULTS_PATH
    mo.md("# EGFR Pathway Simulation")
    return DATA_PATH, RESULTS_PATH, datetime, mo, np, pd, plt, solve_ivp


@app.cell
def _(DATA_PATH, mo):

    data_options = [file.name for file in DATA_PATH.iterdir() if file.name.endswith('csv')]


    data_option_widget = mo.ui.radio(options=data_options, label="Pick which data version you use")
    data_option_widget
    return (data_option_widget,)


@app.cell
def _(data_option_widget):

    data_p =  dow if (dow := data_option_widget.value) is not None else 'data_transient_v5.csv'
    return


@app.cell
def _():
    from models.simple_EGFR_transient import m, y0 as models_y0, light_func, nodes as param_list, nodes as states , df
    #df = pd.read_csv(DATA_PATH / data_p,  index_col=False)
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
def _():
    #dfy = df.groupby(['group','time']).mean('y')
    #dfy
    return


@app.cell
def _(light_func, np, plt):

    _x= np.linspace(1,60,120)
    _y = np.array([light_func(xx, {'group':50}) for xx in _x])
    plt.plot(_x,_y)
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
    json_files = [c for c in os.listdir(RESULTS_PATH) if c.endswith('.json')]
    # Sort by creation time (newest first)
    candidate_params = sorted(json_files, key=lambda f: os.path.getctime(RESULTS_PATH / f), reverse=True)
    param_widget = mo.ui.radio(options=candidate_params, value=candidate_params[0])
    param_widget
    return (param_widget,)


@app.cell
def _(RESULTS_PATH, m, param_widget):
    # Use the new read_config function to load parameters from any JSON format
    dp = m.read_config(RESULTS_PATH / param_widget.value)
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
    t_vals = np.linspace(1, 59, 200)
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
def _():
    # _groups = df['group'].unique()
    # _fig, _ax = plt.subplot(3,2,1)
    # for _a, _g in zip(_ax, _groups):
    #     _d = df[df['group']==_g]
    #     a.hist(d)
    return


@app.cell
def _(RESULTS_PATH, datetime, json, m, mo, sliders):
    def onclick(arg):
        # Get current parameter values
        current_params = {p: sliders[p].value for p in m.parameters}
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"user_defined_{timestamp}.json"
        # Save to results directory
        with open(RESULTS_PATH / filename, 'w') as _f:
            json.dump(current_params, _f, indent=4)
        print('saved', arg)
    save_button = mo.ui.button(label="Save Parameters", on_click=onclick)
    return (save_button,)


@app.cell
def _():

    # Function to handle saving parameters
    return


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
    save_button,
    sliders,
    solve_ivp,
    system,
    t_vals,
    y0,
    ylim_w,
):
    import seaborn as sns
    # Create save button widget


    light_intensity = litp if (litp := pick_light_intensity_to_plot.value) is not None else 200
    print(light_intensity)
    sol = solve_ivp(
        lambda t, y: system(t, y, p_full, light_func, {'group': light_intensity}),
        (1, 59), y0, rtol=1e-4, atol=1e-7, t_eval=t_vals)
    print(sol)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)

    if (ylim := ylim_w.value) != "" and float(ylim):
        ax.set_ylim(top=float(ylim))

    for i, name in enumerate(m.active_states):
        if name in plot_nodes.value:
            ax.plot(sol.t, sol.y[i], label=name)

    if input_plot_widget.value:
        ax.plot(t_vals, [light_func(t, {'group': light_intensity}) for t in t_vals], label="Input")

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
        "State Variables to plot": plot_nodes,
        "Others": [input_plot_widget, plot_original_widget, pick_light_intensity_to_plot, ylim_w],
        "Parameter values": sliders,
        "Save Parameters": mo.vstack([save_button])
    })
    mo.hstack([ctrls, fig], widths=[0.3, 0.7], gap="1rem")
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


@app.cell
def _(dp, mo, plt):
    if 'fit_statistics' not in dp:
        mo.stop('a')

    _stats = dp['fit_statistics']
    _group_stats = _stats['group_statistics']

    # Create a 2x2 subplot layout
    _fig, _axes = plt.subplots(2, 2, figsize=(12, 8))
    _fig.suptitle(f"Fit Statistics - {dp.get('optimization_info', {}).get('method', 'Unknown')} Method", fontsize=14)

    # 1. R² by group (top-left)
    _groups = list(_group_stats.keys())
    _r_squared_vals = [_group_stats[g].get('r_squared', 0) for g in _groups if 'error' not in _group_stats[g]]
    _axes[0,0].bar(_groups, _r_squared_vals, color='steelblue', alpha=0.7)
    _axes[0,0].set_title('R² by Group')
    _axes[0,0].set_ylabel('R²')
    _axes[0,0].set_ylim(0, 1)

    # 2. RMSE by group (top-right)
    _rmse_vals = [_group_stats[g].get('rmse', 0) for g in _groups if 'error' not in _group_stats[g]]
    _axes[0,1].bar(_groups, _rmse_vals, color='coral', alpha=0.7)
    _axes[0,1].set_title('RMSE by Group')
    _axes[0,1].set_ylabel('RMSE')

    # 3. Observed vs Predicted Max Values (bottom-left)
    _obs_max = [_group_stats[g].get('max_observed', 0) for g in _groups if 'error' not in _group_stats[g]]
    _pred_max = [_group_stats[g].get('max_predicted', 0) for g in _groups if 'error' not in _group_stats[g]]
    _axes[1,0].scatter(_obs_max, _pred_max, alpha=0.7, s=60)
    _axes[1,0].plot([0, max(_obs_max)], [0, max(_obs_max)], 'k--', alpha=0.5)
    _axes[1,0].set_xlabel('Observed Max')
    _axes[1,0].set_ylabel('Predicted Max')
    _axes[1,0].set_title('Observed vs Predicted Peak Values')

    # 4. Model Info Table (bottom-right)
    _axes[1,1].axis('off')
    _info_text = f"""
    Overall Statistics:
    • Total Loss: {_stats['overall_mse']:.6f}
    • AIC: {_stats['aic']:.2f}
    • BIC: {_stats['bic']:.2f}
    • Data Points: {_stats['n_data_points']}
    • Fitted Params: {_stats['n_fitted_params']}
    • DOF: {_stats['degrees_of_freedom']}

    Optimization:
    • Method: {dp.get('optimization_info', {}).get('method', 'N/A')}
    • Success: {dp.get('success', 'N/A')}
    • Iterations: {dp.get('optimization_info', {}).get('n_iterations', 'N/A')}
    • Function Evals: {dp.get('optimization_info', {}).get('n_function_evaluations', 'N/A')}
    """
    _axes[1,1].text(0.05, 0.95, _info_text, transform=_axes[1,1].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    return


@app.cell
def _(dp, mo, pd):
    #if 'fit_statistics' not in dp:
        #return (mo.md("No fit statistics available in this result file."), )

    if 'fit_statistics' not in dp:
        mo.stop('a')

    _group_stats = dp['fit_statistics']['group_statistics']

    # Convert to DataFrame for nice display
    _rows = []
    for _group, _stats in _group_stats.items():
        if 'error' not in _stats:
            _rows.append({
                'Group': _group,
                'R²': f"{_stats.get('r_squared', 0):.4f}",
                'RMSE': f"{_stats.get('rmse', 0):.6f}",
                'MSE': f"{_stats.get('mse', 0):.6f}",
                'Max Observed': f"{_stats.get('max_observed', 0):.4f}",
                'Max Predicted': f"{_stats.get('max_predicted', 0):.4f}",
                'N Points': _stats.get('n_points', 0)
            })
        else:
            _rows.append({
                'Group': _group,
                'Error': _stats['error'],
                'R²': 'N/A', 'RMSE': 'N/A', 'MSE': 'N/A',
                'Max Observed': 'N/A', 'Max Predicted': 'N/A', 'N Points': 'N/A'
            })

    _df_stats = pd.DataFrame(_rows)
    return


@app.cell
def _(dp, mo, np, plt):
    #if 'fit_statistics' not in dp:
    #    return (mo.md("No fit statistics available."),)

    if 'fit_statistics' not in dp:
        mo.stop('a')
    _stats = dp['fit_statistics']
    _group_stats = _stats['group_statistics']

    # Extract data for plotting
    _groups = [g for g in _group_stats.keys() if 'error' not in _group_stats[g]]
    _r2_vals = [_group_stats[g]['r_squared'] for g in _groups]
    _rmse_vals = [_group_stats[g]['rmse'] for g in _groups]

    # Create polar plot for R² values
    _fig = plt.figure(figsize=(10, 5))

    # Left: Radar chart for R²
    _ax1 = _fig.add_subplot(121, projection='polar')
    _angles = [i * 2 * np.pi / len(_groups) for i in range(len(_groups))]
    _angles += _angles[:1]  # Complete the circle
    _r2_vals += _r2_vals[:1]  # Complete the circle

    _ax1.plot(_angles, _r2_vals, 'o-', linewidth=2, color='blue', alpha=0.7)
    _ax1.fill(_angles, _r2_vals, alpha=0.25, color='blue')
    _ax1.set_xticks(_angles[:-1])
    _ax1.set_xticklabels([f'Group {g}' for g in _groups])
    _ax1.set_ylim(0, 1)
    _ax1.set_title('R² by Group (Radar Chart)', pad=20)

    # Right: Bar chart for RMSE
    _ax2 = _fig.add_subplot(122)
    _bars = _ax2.bar(_groups, _rmse_vals, color='coral', alpha=0.7)
    _ax2.set_title('RMSE by Group')
    _ax2.set_xlabel('Group')
    _ax2.set_ylabel('RMSE')

    # Add value labels on bars
    for _bar, _val in zip(_bars, _rmse_vals):
        _ax2.text(_bar.get_x() + _bar.get_width()/2, _bar.get_height() + max(_rmse_vals)*0.01,
                f'{_val:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    return


if __name__ == "__main__":
    app.run()
