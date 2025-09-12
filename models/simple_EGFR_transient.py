import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from model import Model, EquationDescription
from typing import List
import pandas as pd
from sympy import Eq, Derivative, Symbol
from sympy.abc import t
import numpy as np
from utils.utils import DATA_PATH

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
  modifier = float(rest['group'])
  if modifier == 0: return 0
  base_intensity = 1.0        # Base light intensity
  max_additional = 1.0        # Maximum additional intensity
  saturation_point = 200.0    # Modifier value where we reach ~50% of max_additional
  
  # Sigmoid scaling: smooth saturation
  intensity_scale = base_intensity + max_additional * modifier / (modifier + saturation_point)

  if t <= 9:
    return 0
  elif t <= 9.1:  # Smooth rise
    return intensity_scale * (t - 9) / 0.1
  elif t <= 10.9:  # Plateau
    return intensity_scale
  elif t <= 11:   # Smooth fall
    return intensity_scale * (11 - t) / 0.1
  else:
    return 0

m = Model(name = 'transient_new_normalization',
          parameters = param_list,
          states = nodes,
          model_definition = model_eqs,
          t_func = light_func,
          t_dep='light'
          )
from utils.utils import read_parquet_and_clean
df = read_parquet_and_clean( DATA_PATH / 'exp_data.parquet', save_as= DATA_PATH / 'data_transient_v4.csv')
df = df.groupby(['group','time']).median('y')
df.reset_index(inplace=True)

y0 = [0.05] * 5
import json
from datetime import datetime
with open(DATA_PATH / 'egfr_fit_transient_1_params.json', 'r') as f:
  p0 = json.load(f)
if __name__ == '__main__':
  models = ['L-BFGS-B','Nelder-Mead']
#'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'SLSQP', 'dogleg', 'trust-ncg', 'trust-krylov', 'trust-exact', 'trust-constr'
  for mod in models:
    m = Model(name = f'trans_check_{mod}',
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
