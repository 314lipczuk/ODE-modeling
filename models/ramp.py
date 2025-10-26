import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.integrate import solve_ivp

from models.transient import model_eqs, param_list, nodes
from experiments.ramp import read_parquet_and_clean, ramp_light_fn_linear, ramp_light_fn_withlog
from utils.utils import DATA_PATH, RESULTS_PATH
from model import Model
from os import environ
from itertools import product

df_raw = read_parquet_and_clean(DATA_PATH / 'data_ramp.parquet', save_as=DATA_PATH / 'data_ramp_v1.csv')

df = (
    df_raw.loc[:, ['time', 'y']]
      .groupby('time', as_index=False)
      .median()
      .sort_values('time')
)
df['group'] = 1

if environ.get('PLOT_INPUT') is not None:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = df['time'].to_numpy()
    y = df['y'].to_numpy()
    ax.scatter(x, y, s=6, alpha=0.6)
    ax.set_title('Ramp data: y over time')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('y')
    _plot_path = RESULTS_PATH / 'ramp_data_overview.png'
    fig.savefig(_plot_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved data overview plot to {_plot_path}')

y0 = [0.05] * 5
param_list = param_list


if __name__ == '__main__':
  # Prioritize robust optimizers for biochemical parameter fitting
  #models = ['trust-constr', 'L-BFGS-B', 'SLSQP', 'Nelder-Mead', 'COBYLA', 'TNC']
  models = ['L-BFGS-B'] #,'Nelder-Mead']
  light_fn = ramp_light_fn_linear

  # plot the training vs original data
  #for light_func in [ramp_light_fn_withlog, ramp_light_fn_linear]:
  
  for i, mod in enumerate(models):
    m = Model(name = f'RAMP_lfn[{light_fn.__name__}]',
          parameters = param_list,
          states = nodes,
          model_definition = model_eqs,
          t_func = light_fn,
          t_dep = 'light'
          )
    print(datetime.now(),'starting fitting....', mod)
    fit_result = m.fit(df,y0, parameters = RESULTS_PATH / 'user_defined_20250913_203158.json')
    m.save_results()
    print(f'done w/ {mod} at: ', datetime.now() )

    system = m.make_numerical()
    fitted_params = fit_result['fitted_params']
    p_full = np.asarray([fitted_params.get(name, 1.0) for name in m.parameters],
                    dtype=np.float64)
    times = np.linspace(1, 175, 175)

    sol = solve_ivp(
        lambda t, y: system(t, y,p_full, m.t_func, None),
        [times[0], times[-1]], y0,
        t_eval=times, method=m.ivp_method, rtol=1e-8
    )
    if not sol.success:
        print(f"Simulation failed for optimizer {mod} w/ {light_fn.__name__}")
        continue
    time = datetime.now().strftime('%d-%m-%Y_%H:%M:%S')
    fig, ax = plt.subplot()
    ax.plot(times, sol.y[-1], label="Solution")
    ax.plot(df['time'], df['y'], label="Exp. Data")
    name = f'RAMP_{time}_t_lin_{m}'
    ax.set_label(name)
    ax.saveplot(RESULTS_PATH / f'{name}.png')


# Note: Intro to AI; 21.10.2025  
# slide 59 -> what is the relationship between these two conditions; cna you go back to general case
# everything up to MRW from lec 4 is on midterm