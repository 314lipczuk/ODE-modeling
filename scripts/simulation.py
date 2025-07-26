from sympy import symbols, Function, Derivative, Eq, Symbol, lambdify
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

param_list = ["r12", "r23", "r31", "r21",
"k12", "K12", "k21", "K21",
"k34", "K34","knfb","k43","K43",
"k56", "K56","k65","K65",
"k78", "K78","k87","K87",
"k910","K910",
"s12", "s21",
"f12", "F12", "f21", "F21",
"ktr_init", "rho"]

# horrible horrible hack. TODO: When i know more about code like this, re-factor this into something more sane, with no namespace-hacking.
for p in param_list: globals()[p] = Symbol(p)

nodes = ["RAS", "RAF", "MEK", "ERK", "EGFR", "NFB", "KTR"] 
nodes.extend(*[[f'{b}_s' for b in nodes]])
nodes.append('EGFR_endo')

t = symbols('t')
EGF_fn = Function("EGF")
light_fn = Function("light")
EGF = EGF_fn(t)
light = light_fn(t)


for n in nodes: globals()[n] = Function(n)(t)

eqs = [dEGFR_dt := Eq(Derivative(EGFR, t),          -r12 * EGF * EGFR + r21 * EGFR_s + r31 * EGFR_endo),
    dEGFRs_dt := Eq(Derivative(EGFR_s,t),           r12 * EGF * EGFR - (r21 + r31 ) * EGFR_s),
    dEGFR_endo_dt := Eq(Derivative(EGFR_endo, t),   r23 * EGFR_s - r31 * EGFR_endo ),

    dRAS_dt := Eq(Derivative(RAS, t ),              -(k12*EGFR_s+light) * (RAS/(K12+RAS)) + k21 * (RAS_s/(K21+RAS_s))),
    dRASs_dt:= Eq(Derivative(RAS_s, t ),            (k12*EGFR_s+light) * (RAS/(K12+RAS)) - k21 * (RAS_s/(K21+RAS_s))),

    dRAF_dt := Eq(Derivative(RAF, t ),              -(k34*RAS_s) * (RAF/(K34+RAF)) + (knfb * NFB_s + k43) * (RAF_s/(K43+RAF_s))),
    dRAFs_dt := Eq(Derivative(RAF_s, t ),           k34 * RAS_s * (RAF/(K34+RAF)) - (knfb * NFB_s + k43) * (RAF_s/(K43+RAF_s))),

    dMEK_dt := Eq(Derivative(MEK, t),               -k56 * RAF_s * (MEK/(K56+MEK)) + k65 * (MEK_s/(K65+MEK_s))),
    dMEKs_dt := Eq(Derivative(MEK_s, t),            k56 * RAF_s * (MEK/(K56+MEK)) - k65 * (MEK_s/(K65+MEK_s))),

    dERK_dt := Eq(Derivative(ERK, t),               -k78 * MEK_s * (ERK/(K78+ERK)) + k87 * (ERK_s/(K87+ERK_s))),
    dERKs_dt := Eq(Derivative(ERK_s, t),            k78 * MEK_s * (ERK/(K78+ERK)) - k87 * (ERK_s/(K87+ERK_s))),

    dNFB_dt := Eq(Derivative(NFB, t),               -f12 * ERK_s * (NFB/F12+NFB) + f21*(NFB_s/(F21+NFB_s))),
    dNFBs_dt := Eq(Derivative(NFB_s, t),            f12 * ERK_s * (NFB/F12+NFB) - f21*(NFB_s/(F21+NFB_s))),

    dKTR_dt := Eq(Derivative(KTR, t),               -(k910*ERK_s*(KTR/(K910+KTR)) + s12*KTR) + s21 * KTR_s),
    dKTRs_dt := Eq(Derivative(KTR_s, t),            (k910*ERK_s*(KTR/(K910+KTR)) + s12*KTR) - s21 * KTR_s) ]

rhs_exprs = [e.rhs for e in eqs]

for n in nodes: globals()[f'{n}_'] = Symbol(n) 

t_ = Symbol("t")
#EGF_ = Function("EGF")(t_)  # Keep EGF(t_) as a time-varying input
#light_ = Function("light")(t_)  # Keep EGF(t_) as a time-varying input
EGF_ = EGF_fn(t_)
light_ =light_fn(t_)

# TODO: figure out a way to automatically generate the 'flat' symbols, and its dependants
# basic ideae is easy and tempting, but it gets harder when you realise that input nodes can be
# anywhere in the network, and so basic layering is hard to do. 
# The ideal API for this would be to
# INPUT: DEs, annotate inputs & parameters
# OUTPUT: xyz_system function, or at least lambdified set of equations.
# For now however, doing it manually is fine.

flat_symbols = {globals()[k]:Symbol(k) for k in nodes}
subs_map = flat_symbols.copy() 
subs_map[t] = t_ 

subs_map[Function("EGF")(t)] = EGF_
subs_map[Function("light")(t)] = light_

parameters = [globals()[p] for p in param_list]

arg_list = (t_, *flat_symbols.values(), *parameters, EGF_, light_)

# Lambdified RHS functions
rhs_funcs = [
    lambdify(arg_list, rhs.subs(subs_map), modules='numpy')
    for rhs in rhs_exprs
]

def egfr_system(t, y, param_values, EGF_func, light_func):
    EGF_val = EGF_func(t)
    light_val = light_func(t)
    args = [t, *y, *param_values, EGF_val, light_val]
    return [f(*args) for f in rhs_funcs]

# Input function
EGF_input = lambda t: 2.0 if t < 10 else 0.5
light_input = lambda t: 1.0

# Taken from github
initial_cond_defaults = {
    "RAS":1,
    "RAS_s":0,
    "RAF":1,
    "RAF_s":0,
    "MEK":1,
    "MEK_s":0,
    "ERK_s":0,
    "EGFR":1,
    "EGFR_s":0,
    "EGFR_endo":0,
    "NFB":1,
    "NFB_s":0,
    "KTR":0.5,
    "KTR_s":0.5,
    }

# Finishin up for today; Next day work:
# TODO: Do notebook-based interactive visuals

param_defaults = { k:1 for k in param_list }
param_values = [param_defaults[p] for p in param_list]

state_vars = [str(eq.lhs.args[0]) for eq in eqs]
assert len(state_vars) == len(rhs_exprs), "Sanity check for length of non-input nodes"

y0 = [(initial_cond_defaults.get(n) or 0.5) for n in state_vars]

# Time span
t_span = (0, 30)
t_eval = np.linspace(*t_span, 300)

# Solve
sol = solve_ivp(lambda t, y: egfr_system(t, y, param_values, EGF_input, light_input),
                t_span, y0, t_eval=t_eval)

def plot_ivp_solution(sol, labels=None, title="Simulation Results"):
    """
    Plot the output of a solve_ivp solution object.

    Parameters:
        sol : OdeResult
            Output from scipy.integrate.solve_ivp
        labels : list of str
            Labels for each state variable (same order as in sol.y)
        title : str
            Plot title
    """
    plt.figure(figsize=(10, 6))
    num_vars = sol.y.shape[0]

    for i in range(num_vars):
        label = labels[i] if labels else f"y{i}"
        plt.plot(sol.t, sol.y[i], label=label)

    plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

