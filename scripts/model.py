import numpy as np
from sympy import Symbol, Function, Eq, Derivative, lambdify, Expr
from sympy.core.relational import Equality
from sympy.abc import t
from typing import List, Dict, TypedDict, NotRequired, AnyStr
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import json

'''
TODO:   Think about time dilation, as a lot of the signalling phenomena happen on timescale 
        invisible from frame point of view (50ms pulse of light looks identical to every other if
        our time resolution is 1 frame (1 second)). 
        On the second thought, that should probably happen outside of this function,
        maybe on the level of preparing the dataframe? Just a helper func or sth?

'''


class Model:
    name = ''
    params = []    
    eqs = []
    fit_result = None
    def __init__(self, name, states, parameters, model_definition, t_func, t_dep):
        '''
        We give it a name, we give it all the symbolic equations, and params (defaults?)

        Pipeline:
            data ---------------------------------------.
                                                        |
            eqs + params + light_fn -> numeric system  -L--> make loss function -> feed to minimizer

        What's a good API to insert EQs? Do i need to input symbol for t?           
        '''
        self.name = name
        self.states = states
        self.parameters = parameters
        self.model_definition_f = model_definition
        model = model_definition(parameters, states)
        self.eqs = model['equations']
        self.symbols = model['symbols']
        self.t_func = t_func # function, f.ex def light_fn(t, args*)
        self.t_dep = t_dep # symbol
        
    def transform(self, ts):
        for transform in ts:
            for i, eq in enumerate(self.eqs):
                self.eqs[i] = eq.subs(transform)
                
    def _make_numerical(self, module='numpy'):

        free_vars = set()
        for eq in self.eqs: free_vars.update(eq.rhs.free_symbols)
        # Find which state names have their corresponding symbols in free_vars
        active_states = []
        for state_name in self.states:
            if self.symbols[state_name] in free_vars:
                active_states.append(state_name)
        # ^ cannot just use self.states from the start cause the 
        #   self.transform() could have reduced number of parameters.
        
        states = [self.symbols[s] for s in active_states]

        parameters = [self.symbols[p] for p in self.parameters]

        arg_list = (t, *states, *parameters, self.symbols[self.t_dep])
        numerical_funcs = [lambdify(arg_list, eq.rhs, modules=module )
                           for eq in self.eqs ]

        def system(t, y, params, t_func, t_args):
            res = t_func(t, *t_args)
            args = [t, *y, *params, res]
            return [f(*args) for f in numerical_funcs]
            
        return system

    def fit(self, dataframe, y0, p0, t_args, params_to_fit, params_to_fix): 
        if params_to_fix is None: params_to_fix = {}
        
        # Create parameter name to index mapping from model parameters
        param_name_to_index = {param: i for i, param in enumerate(self.parameters)}
        
        # Prepare dataframe
        df_processed = dataframe.copy()
        df_processed['time'] = (df_processed['frame'] - 1) * 60.0  # Convert frame to seconds
        df_processed['pulse_duration_ms'] = df_processed['stim_exposure']  # Stimulation duration
        df_processed['y'] = df_processed['cnr_norm']  
        df_processed['start_time_ms'] = (10 - 1) * 60 * 1000  # Frame 10 = 540 seconds = 540000 ms
        
        param_names = list(params_to_fit.keys())
        param_indices = [param_name_to_index[name] for name in param_names]
        
        # Get the numerical system
        system = self._make_numerical()

        def objective(p_fit_values):
            # Build parameter vector for this model only
            p_full = np.zeros(len(self.parameters))
            
            # Start with provided p0 values
            for i, param_name in enumerate(self.parameters):
                if param_name in params_to_fit:
                    # Find which fitted parameter this is
                    fit_idx = param_names.index(param_name)
                    p_full[i] = p_fit_values[fit_idx]
                elif param_name in params_to_fix:
                    p_full[i] = params_to_fix[param_name]
                else:
                    # Use default from p0
                    p_full[i] = p0[i] if i < len(p0) else 1.0
            
            total_loss = 0.0
            
            # Group by unique experiments (stim_exposure, uid combinations)
            unique_experiments = df_processed[['pulse_duration_ms', 'uid']].drop_duplicates()
            
            for _, exp_row in unique_experiments.iterrows():
                exp_data = df_processed[
                    (df_processed['pulse_duration_ms'] == exp_row['pulse_duration_ms']) & 
                    (df_processed['uid'] == exp_row['uid'])
                ].sort_values('time')
                
                times = exp_data['time'].values
                real_data = exp_data['y'].values
                
                try:
                    sol = solve_ivp(
                        lambda t, y: system(t, y, p_full, self.t_func, self.t_dep),
                        [times[0], times[-1]], y0, 
                        t_eval=times, method='LSODA', rtol=1e-8
                    )
                    
                    assert sol.success, "sol.success should be True"
                    # Use the last state as the observable (in simple model: y, in EGFR: KTR_s)
                    obs_pred = sol.y[-1]  # Last state variable
                    loss = np.sum((obs_pred - real_data)**2)
                    total_loss += loss
                
                except Exception as e:
                    print(f"Error solving ODE: {e}")
                    return 1e10
            
            return total_loss
        
        # Initial parameter guess
        p_init = [params_to_fit[name] for name in param_names]
        
        # Optimization
        result = minimize(
            objective, p_init, 
            method='L-BFGS-B',
            bounds=[(0.001, 5)] * len(p_init),
            options={'maxiter': 1000}
        )
        
        # Package results
        fitted_params = dict(zip(param_names, result.x))
        
        self.fit_result = {
            'fitted_params': fitted_params,
            'loss': result.fun,
            'success': result.success,
            'message': result.message,
            'n_experiments': len(df_processed[['pulse_duration_ms', 'uid']].drop_duplicates())
        }
        
        return self.fit_result




class EquationDescription(TypedDict):
    eqations:List[Equality]
    constraints: NotRequired[List[Equality]]
    symbols:NotRequired[List[Symbol]]

def model_eqs(params: List[str], states: List[str]) -> EquationDescription:
    # Create a dictionary to hold all symbols
    symbols_dict = {}
    
    # Create parameter symbols
    for p in params: 
        symbols_dict[p] = Symbol(p)
    
    # Create state symbols  
    for s in states: 
        symbols_dict[s] = Symbol(s)
    
    # Also create time symbol
    symbols_dict['t'] = Symbol('t')
    symbols_dict['light'] = Symbol('light')  # assuming this is also a symbol
    
    # Access symbols from dictionary
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
from itertools import chain

def flatten_extend(matrix):
    flat_list = []
    for row in matrix:
        flat_list.extend(row)
    return flat_list
nodes = flatten_extend([ [f'{node}_s',node ] for node in  ["RAS", "RAF", "MEK", "ERK", "NFB"]] )

m = Model(name = 'basic_egfr',
          parameters = param_list,
          states = nodes,
          model_definition = model_eqs,
          t_func=lambda t:t,
          t_dep='light'
          )
print(m.eqs)
print('#' * 30)
m.transform(
    [
        # change f.ex RAS into (1-RAS_s), for all states
        {k:(1 - Symbol(f'{k}_s')) for k in nodes if not k.endswith('_s')}
    ])
print(m.eqs)
syst = m._make_numerical()
from scipy.integrate import solve_ivp
import numpy as np

def tfunc(t, tp=None):
    match tp:
        case 'a': return t
        case 'b': return 1
        case _: return 2

sol = solve_ivp(
    lambda t,y:syst(t,y, np.ones(len(param_list)), tfunc, []),
    [0,3], np.array([0.1,0.1,0.1,0.1,0.1])
)