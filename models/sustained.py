import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.integrate import solve_ivp

from models.transient import model_eqs, param_list, nodes
from experiments.sustained import read_parquet_and_clean , sustained_light_fn, scaling
from utils.utils import DATA_PATH, RESULTS_PATH
from model import Model
from os import environ
from itertools import product

groups = list(scaling.keys())

df_raw = read_parquet_and_clean(DATA_PATH / 'data_sustained.parquet', save_as=DATA_PATH / 'data_sustained_v1.csv')

df = (
  df_raw.loc[:, ['time', 'y', 'group']]
    .groupby(['group', 'time'], as_index=False)
    .median()
    .sort_values('time')
)

if environ.get('PLOT_INPUT') is not None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for gr in groups:
      d = df[df['group'] == gr]
      x = d['time'].to_numpy()
      y = d['y'].to_numpy()
      ax.plot(x,y,label=gr)
    ax.set_title('Sustained data: y over time')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('y')
    _plot_path = RESULTS_PATH / 'sustained_data_overview.png'
    fig.savefig(_plot_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved data overview plot to {_plot_path}')

y0 = [0.05] * 5


if __name__ == '__main__':
  # Prioritize robust optimizers for biochemical parameter fitting
  #models = ['trust-constr', 'L-BFGS-B', 'SLSQP', 'Nelder-Mead', 'COBYLA', 'TNC']
  models = ['L-BFGS-B'] #,'Nelder-Mead']

  # plot the training vs original data
  #for light_func in [ramp_light_fn_withlog, ramp_light_fn_linear]:
  
  for i, mod in enumerate(models):
    m = Model(name = f'Sustained_v1',
          parameters = param_list,
          states = nodes,
          model_definition = model_eqs,
          t_func = sustained_light_fn,
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
    fig, ax = plt.subplots()

    for gr in groups:
      d = df[df['group'] == gr]
      x = d['time'].to_numpy()
      y = d['y'].to_numpy()
      ax.plot(x,y,label=gr)
    
      sol = solve_ivp(
          lambda t, y: system(t, y,p_full, m.t_func, {"group":gr}),
          [times[0], times[-1]], y0,
          t_eval=times, method=m.ivp_method, rtol=1e-8
      )
      if not sol.success:
          print(f"Simulation failed for optimizer {mod} w/ {sustained_light_fn.__name__}, group:",gr)
          continue
      ax.plot(times, sol.y[-1], label=f"Sol: {gr}")
    time = datetime.now().strftime('%d-%m-%Y_%H:%M:%S')
    d = df[df['group'] == gr]
    ax.plot(d['time'], d['y'], label="Exp. Data")
    name = f'Sustained_{time}_t_lin_{m}'
    fig.legend()
    _plot_path = RESULTS_PATH / 'sustained_fit_overview.png'
    fig.savefig(_plot_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved data overview plot to {_plot_path}')
    
    ax.set_label(name)
    fig.savefig(RESULTS_PATH / f'{name}.png')

# Note: Intro to AI; 21.10.2025  
# slide 59 -> what is the relationship between these two conditions; cna you go back to general case
# everything up to MRW from lec 4 is on midterm

