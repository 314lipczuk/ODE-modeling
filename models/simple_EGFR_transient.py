import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from model import Model, EquationDescription
from typing import List
import pandas as pd
from sympy import Eq, Derivative, Symbol
from sympy.abc import t
import numpy as np
from utils.utils import DATA_PATH, RESULTS_PATH

def model_eqs(params: List[str], states: List[str]) -> EquationDescription:
    symbols_dict = {}
    
    for p in params: 
        symbols_dict[p] = Symbol(p)
    
    for s in states: 
        symbols_dict[s] = Symbol(s)
    
    symbols_dict['t'] = t
    symbols_dict['light'] = Symbol('light')  # assuming this is also a symbol
    
    s = symbols_dict  # shorter alias

    base_eqs = [
        Eq(Derivative(s['RAS_s'], s['t']), s['light'] * (s['RAS']/(s['K12']+s['RAS'])) - s['k21'] * (s['RAS_s']/(s['K21']+s['RAS_s']))),
        Eq(Derivative(s['RAF_s'], s['t']), s['k34'] * s['RAS_s'] * (s['RAF'] / (s['K34'] + s['RAF'])) - (s['knfb'] * s['NFB_s'] + s['k43']) * (s['RAF_s']/(s['K43']+s['RAF_s']))),
        Eq(Derivative(s['MEK_s'], s['t']), s['k56'] * s['RAF_s'] * (s['MEK'] / (s['K56'] + s['MEK'])) - s['k65'] * (s['MEK_s']/(s['K65']+s['MEK_s']))),
        Eq(Derivative(s['NFB_s'], s['t']), s['f12'] * s['ERK_s'] * (s['NFB'] / (s['F12'] + s['NFB'])) - s['f21']*(s['NFB_s']/(s['F21']+s['NFB_s']))),
        Eq(Derivative(s['ERK_s'], s['t']), s['k78'] * s['MEK_s'] * (s['ERK'] / (s['K78'] + s['ERK'])) - s['k87'] * (s['ERK_s']/(s['K87']+s['ERK_s']))),
    ]

    equations = []
    for eq in base_eqs:
      equations.append(eq.subs({k:(1 - Symbol(f'{k}_s')) for k in nodes if not k.endswith('_s')}))
    
    return {'base_equations': base_eqs, 'symbols': symbols_dict, "equations": equations}


param_list = ["K12", "k21", "K21",
"k34", "K34","knfb","k43","K43",
"k56", "K56","k65","K65",
"k78", "K78","k87","K87",
"f12", "F12", "f21", "F21",
]
def flatten_extend(matrix):
    flat_list = []
    for row in matrix:
        flat_list.extend(row)
    return flat_list

nodes = flatten_extend([ [f'{node}_s',node ] for node in  ["RAS", "RAF", "MEK",  "NFB", "ERK"]] )

def light_func(t, rest=None):
  # Smooth transitions to avoid solver issues
  delta_t = 0.2

  if (10-delta_t) < t < (10+delta_t): 
    modifier = float(rest['group'])
    if modifier == 0: return 0
    log_modifier = np.log(modifier)
    return log_modifier - log_modifier * (np.abs(10-t)/delta_t)
  else:
    return 0


m = Model(name = 'retvrn',
          parameters = param_list,
          states = nodes,
          model_definition = model_eqs,
          t_func = light_func,
          t_dep='light'
          )

def test_lightFn():
  _t = np.linspace(0,60, 150)
  result = np.array([light_func(it, {'group':0}) for it in _t])
  assert np.all(result == 0)

  #result = np.array([light_func(it, {'group':50}) for it in _t])
  #assert np.all(result[-50:-1] == 0)

  #result = np.array([light_func(it, {'group':50}) for it in _t])
  #assert np.all(result[0:10] == 0)


test_lightFn()
          

from utils.utils import read_parquet_and_clean
df = read_parquet_and_clean( DATA_PATH / 'exp_data.parquet', save_as= DATA_PATH / 'data_transient_v5.csv')
df = df.groupby(['group','time']).median('y')
df.reset_index(inplace=True)

y0 = [0.05] * 5
import json
from datetime import datetime
with open(RESULTS_PATH / 'user_defined_20250913_203158.json', 'r') as f:
  p0 = json.load(f)
if __name__ == '__main__':
  # Prioritize robust optimizers for biochemical parameter fitting
  #models = ['trust-constr', 'L-BFGS-B', 'SLSQP', 'Nelder-Mead', 'COBYLA', 'TNC']
  models = ['L-BFGS-B','Nelder-Mead']

  for mod in models:
    m = Model(name = f'simple_lf_logged_{mod}',
          parameters = param_list,
          states = nodes,
          model_definition = model_eqs,
          t_func = light_func,
          t_dep='light'
          )
    print(datetime.now(),'starting fitting....', mod  )
    m.fit(df,y0,p0, None,p0, None)
    m.save_results()
    print(f'done w/ {mod} at: ', datetime.now() )
