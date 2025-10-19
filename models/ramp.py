import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.integrate import solve_ivp

from simple_EGFR_transient import model_eqs, param_list, nodes
from experiments.ramp import read_parquet_and_clean, ramp_light_fn
from utils.utils import DATA_PATH, RESULTS_PATH
from model import Model

df_raw = read_parquet_and_clean(DATA_PATH / 'data_ramp.parquet', save_as=DATA_PATH / 'data_ramp_v1.csv')

# Plot raw data: y over time (as-is)
fig, ax = plt.subplots(figsize=(9, 5))
x = df_raw['time'].to_numpy()
y = df_raw['y'].to_numpy()
ax.scatter(x, y, s=6, alpha=0.6)
ax.set_title('Ramp data: y over time')
ax.set_xlabel('Time (s)')
ax.set_ylabel('y')
_plot_path = RESULTS_PATH / 'ramp_data_overview.png'
fig.savefig(_plot_path, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f'Saved data overview plot to {_plot_path}')
 # Deduplicate times for ODE solver: median y per time, sorted, with a single group
df = (
    df_raw.loc[:, ['time', 'y']]
      .groupby('time', as_index=False)
      .median()
      .sort_values('time')
)
df['group'] = 1

y0 = [0.05] * 5

if __name__ == '__main__':
  # Prioritize robust optimizers for biochemical parameter fitting
  #models = ['trust-constr', 'L-BFGS-B', 'SLSQP', 'Nelder-Mead', 'COBYLA', 'TNC']
  models = ['L-BFGS-B','Nelder-Mead']

  for mod in models:
    m = Model(name = f'ramp_v1',
          parameters = param_list,
          states = nodes,
          model_definition = model_eqs,
          t_func =ramp_light_fn,
          t_dep='light'
          )
    print(datetime.now(),'starting fitting....', mod)
    fit_result = m.fit(df,y0, parameters=RESULTS_PATH / 'user_defined_20250913_203158.json')
    m.save_results()
    print(f'done w/ {mod} at: ', datetime.now() )

    system = m.make_numerical()
    fitted_params = fit_result['fitted_params']
    p_full = np.array([fitted_params.get(param, 1.0) for param in m.parameters], dtype=float)

    # Plot a single line using median across all data (no per-stim grouping)
    gy = (
        df.loc[:, ['time', 'y']]
          .groupby('time', sort=True, as_index=False)
          .median()
          .sort_values('time')
    )

    times = gy['time'].to_numpy()
    observed = gy['y'].to_numpy()

    # Use a single representative stim_exposure for the simulation (from raw data)
    stim_vals = df_raw['stim_exposure'].to_numpy()
    pos_stim = stim_vals[stim_vals > 0]
    if pos_stim.size:
        stim_exposure_val = float(np.median(pos_stim))
    else:
        stim_exposure_val = float(np.median(stim_vals)) if stim_vals.size else 1.0

    t_args = {'stim_exposure': stim_exposure_val}

    sol = solve_ivp(
        lambda t, y: system(t, y, p_full, m.t_func, t_args),
        [times[0], times[-1]], y0,
        t_eval=times, method=m.ivp_method, rtol=1e-8
    )

    if not sol.success:
        print(f"Simulation failed for optimizer {mod}")
        continue

    predicted = np.asarray(sol.y)[-1].ravel()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(times, observed, s=16, alpha=0.7, label='Median training data')
    ax.plot(times, predicted, linewidth=2, label='Model simulation')

    ax.set_title(f"Ramp Fit vs Data ({mod})")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Normalized response')
    ax.legend(loc='best', fontsize=8)

    plot_path = RESULTS_PATH / f"ramp_fit_{mod.lower().replace('-', '_')}.png"
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved fit plot to {plot_path}")
