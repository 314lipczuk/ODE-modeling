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
    
    eqs = [
        Eq(Derivative(s['RAS_s'], s['t']), s['light'] * (s['RAS']/(s['K12']+s['RAS'])) - s['k21'] * (s['RAS_s']/(s['K21']+s['RAS_s']))),
        Eq(Derivative(s['RAF_s'], s['t']), s['k34'] * s['RAS_s'] * (s['RAF'] / (s['K34'] + s['RAF'])) - (s['knfb'] * s['NFB_s'] + s['k43']) * (s['RAF_s']/(s['K43']+s['RAF_s']))),
        Eq(Derivative(s['MEK_s'], s['t']), s['k56'] * s['RAF_s'] * (s['MEK'] / (s['K56'] + s['MEK'])) - s['k65'] * (s['MEK_s']/(s['K65']+s['MEK_s']))),
        Eq(Derivative(s['NFB_s'], s['t']), s['f12'] * s['ERK_s'] * (s['NFB'] / (s['F12'] + s['NFB'])) - s['f21']*(s['NFB_s']/(s['F21']+s['NFB_s']))),
        Eq(Derivative(s['ERK_s'], s['t']), s['k78'] * s['MEK_s'] * (s['ERK'] / (s['K78'] + s['ERK'])) - s['k87'] * (s['ERK_s']/(s['K87']+s['ERK_s']))),
    ]
    return {'equations': eqs, 'symbols': symbols_dict}


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
  import numpy as np
  if t <= 9:
    return 0
  elif t <= 9.1:  # Smooth rise
    return (t - 9) / 0.1
  elif t <= 10.9:  # Plateau
    return 1
  elif t <= 11:   # Smooth fall
    return (11 - t) / 0.1
  else:
    return 0

m = Model(name = 'simple_egfr_transient',
          parameters = param_list,
          states = nodes,
          model_definition = model_eqs,
          t_func = light_func,
          t_dep='light'
          )
m.transform(
    [
        # change f.ex RAS into (1-RAS_s), for all states
        {k:(2 - Symbol(f'{k}_s')) for k in nodes if not k.endswith('_s')}
    ])


data = DATA_PATH / "data_transient_v2.csv"
df = pd.read_csv(data)

y0 = [0.05] * 5
import json
with open(DATA_PATH / 'egfr_fit_transient_1_params.json', 'r') as f:
  p0 = json.load(f)

if __name__ == '__main__':
  from datetime import datetime
  print(datetime.now(),'starting fitting....'  )
  m.fit(df,y0,p0, None,p0, None)
  m.save_results()
  print('done at: ', datetime.now() )
