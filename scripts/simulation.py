from sympy import symbols, Function, Derivative, Eq, Symbol, lambdify
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

param_list = ["K12", "k21", "K21",
"k34", "K34","knfb","k43","K43",
"k56", "K56","k65","K65",
"k78", "K78","k87","K87",
"k910","K910",
"s12", "s21",
"f12", "F12", "f21", "F21",
"ktr_init", "rho"]

nodes = ["RAS", "RAF", "MEK", "ERK", "NFB", "KTR"]
param_list.extend([f'tot{node}' for node in nodes])
active_variants = [f'{n}_s' for n in nodes]

# horrible horrible hack. TODO: When i know more about code like this, re-factor this into something more sane, with no namespace-hacking.
for p in param_list: globals()[p] = Symbol(p)

#nodes.extend(*[[f'{b}_s' for b in nodes]])

t = symbols('t')
light_fn = Function("light")
light = light_fn(t)

#for n in nodes: globals()[n] = Function(n)(t)
for n in active_variants: globals()[n] = Function(n)(t)
for n in nodes: globals()[n] = Symbol(n)

#eqs = [dRASs_dt:= Eq(Derivative(RAS_s, t ),            light * (RAS/(K12+RAS)) - k21 * (RAS_s/(K21+RAS_s))),
#    dRAS_dt := Eq(Derivative(RAS, t ),           -light * (RAS/(K12+RAS)) + k21 * (RAS_s/(K21+RAS_s))),
#    
#
#    dRAF_dt := Eq(Derivative(RAF, t ),              -(k34*RAS_s) * (RAF/(K34+RAF)) + (knfb * NFB_s + k43) * (RAF_s/(K43+RAF_s))),
#    dRAFs_dt := Eq(Derivative(RAF_s, t ),           k34 * RAS_s * (RAF/(K34+RAF)) - (knfb * NFB_s + k43) * (RAF_s/(K43+RAF_s))),
#
#    dMEK_dt := Eq(Derivative(MEK, t),               -k56 * RAF_s * (MEK/(K56+MEK)) + k65 * (MEK_s/(K65+MEK_s))),
#    dMEKs_dt := Eq(Derivative(MEK_s, t),            k56 * RAF_s * (MEK/(K56+MEK)) - k65 * (MEK_s/(K65+MEK_s))),
#
#    dERK_dt := Eq(Derivative(ERK, t),               -k78 * MEK_s * (ERK/(K78+ERK)) + k87 * (ERK_s/(K87+ERK_s))),
#    dERKs_dt := Eq(Derivative(ERK_s, t),            k78 * MEK_s * (ERK/(K78+ERK)) - k87 * (ERK_s/(K87+ERK_s))),
#
#    dNFB_dt := Eq(Derivative(NFB, t),               -f12 * ERK_s * (NFB/F12+NFB) + f21*(NFB_s/(F21+NFB_s))),
#    dNFBs_dt := Eq(Derivative(NFB_s, t),            f12 * ERK_s * (NFB/F12+NFB) - f21*(NFB_s/(F21+NFB_s))),
#
#    dKTR_dt := Eq(Derivative(KTR, t),               -(k910*ERK_s*(KTR/(K910+KTR)) + s12*KTR) + s21 * KTR_s),
#    dKTRs_dt := Eq(Derivative(KTR_s, t),            (k910*ERK_s*(KTR/(K910+KTR)) + s12*KTR) - s21 * KTR_s) ]
#

base_eqs = [dRASs_dt:= Eq(Derivative(RAS_s, t ),         light * (RAS/(K12+RAS)) - k21 * (RAS_s/(K21+RAS_s))),
    #dRAS_dt := Eq(Derivative(RAS, t ),           -light * (RAS/(K12+RAS)) + k21 * (RAS_s/(K21+RAS_s))),
    

    #dRAF_dt := Eq(Derivative(RAF, t ),              -(k34*RAS_s) * (RAF/(K34+RAF)) + (knfb * NFB_s + k43) * (RAF_s/(K43+RAF_s))),
    dRAFs_dt := Eq(Derivative(RAF_s, t ),           k34 * RAS_s * (RAF/(K34+RAF)) - (knfb * NFB_s + k43) * (RAF_s/(K43+RAF_s))),

    #dMEK_dt := Eq(Derivative(MEK, t),               -k56 * RAF_s * (MEK/(K56+MEK)) + k65 * (MEK_s/(K65+MEK_s))),
    dMEKs_dt := Eq(Derivative(MEK_s, t),            k56 * RAF_s * (MEK/(K56+MEK)) - k65 * (MEK_s/(K65+MEK_s))),

    #dERK_dt := Eq(Derivative(ERK, t),               -k78 * MEK_s * (ERK/(K78+ERK)) + k87 * (ERK_s/(K87+ERK_s))),
    dERKs_dt := Eq(Derivative(ERK_s, t),            k78 * MEK_s * (ERK/(K78+ERK)) - k87 * (ERK_s/(K87+ERK_s))),

    #dNFB_dt := Eq(Derivative(NFB, t),               -f12 * ERK_s * (NFB/F12+NFB) + f21*(NFB_s/(F21+NFB_s))),
    dNFBs_dt := Eq(Derivative(NFB_s, t),            f12 * ERK_s * (NFB/F12+NFB) - f21*(NFB_s/(F21+NFB_s))),

    #dKTR_dt := Eq(Derivative(KTR, t),               -(k910*ERK_s*(KTR/(K910+KTR)) + s12*KTR) + s21 * KTR_s),
    dKTRs_dt := Eq(Derivative(KTR_s, t),            (k910*ERK_s*(KTR/(K910+KTR)) + s12*KTR) - s21 * KTR_s) ]


# Substitute the 2 variants by v1 and total - v1
conservation_substitution = {globals()[k]:globals()[f'tot{k}']-globals()[f'{k}_s'] for k in nodes }
#for e in eqs:
#    e = e.subs(conservation_substitution)
eqs = [e.subs(conservation_substitution) for e in base_eqs]

rhs_exprs = [e.rhs for e in eqs]

#for n in nodes: globals()[f'{n}_'] = Symbol(n) 
for n in active_variants: globals()[f'{n}_'] = Symbol(n) 


t_ = Symbol("t")
light_ =light_fn(t_)

# TODO: figure out a way to automatically generate the 'flat' symbols, and its dependants
# basic ideae is easy and tempting, but it gets harder when you realise that input nodes can be
# anywhere in the network, and so basic layering is hard to do. 
# The ideal API for this would be to
# INPUT: DEs, annotate inputs & parameters
# OUTPUT: xyz_system function, or at least lambdified set of equations.
# For now however, doing it manually is fine.

flat_symbols = {globals()[k]:Symbol(k) for k in active_variants}
subs_map = flat_symbols.copy() 
subs_map[t] = t_ 

subs_map[Function("light")(t)] = light_

parameters = [globals()[p] for p in param_list]

arg_list = (t_, *flat_symbols.values(), *parameters, light_)

# Lambdified RHS functions
rhs_funcs = [
    lambdify(arg_list, rhs.subs(subs_map), modules='numpy')
    for rhs in rhs_exprs
]

def egfr_system(t, y, param_values, light_func):
    light_val = light_func(t)
    args = [t, *y, *param_values, light_val]
    return [f(*args) for f in rhs_funcs]

# Input function
#EGF_input = lambda t: 2.0 if t < 10 else 0.5
light_input = lambda t: 1.0

# Taken from github
initial_cond_defaults = {
    "RAS":0.9,
    "RAS_s":0.01,
    "RAF":0.7,
    "RAF_s":0,
    "MEK":0.68,
    "MEK_s":0,
    "ERK":0.26,
    "ERK_s":0,
    "NFB":0.1,
    "NFB_s":0,
    "KTR":0.5,
    "KTR_s":0.5,
    }

param_defaults = {
    # Binding/activation Michaelis-Menten constants
    "K12": 0.5, "K21": 0.3,
    "K34": 0.4, "K43": 0.3,
    "K56": 0.4, "K65": 0.3,
    "K78": 0.4, "K87": 0.3,
    "K910": 0.4,

    # Rate constants: activation
    "k34": 1.0,
    "k56": 1.2,
    "k78": 1.0,
    "k910": 1.0,
    "f12": 0.8,
    "s12": 0.5,

    # Rate constants: deactivation
    "k21": 0.5,
    "k43": 0.6,
    "k65": 0.5,
    "k87": 0.4,
    "f21": 0.5,
    "s21": 0.3,

    # Feedback coupling
    "knfb": 0.5,

    # NFB affinities
    "F12": 0.4,
    "F21": 0.3,

    # Total protein concentrations (dimensionless units)
    #"totRAS": 0.1,
    #"totRAF": 0.7,
    #"totMEK": 0.68,
    #"totERK": 0.26,
    #"totNFB": 0.1,
    #"totKTR": 1.0,

    "totRAS": 1,
    "totRAF": 1,
    "totMEK": 1,
    "totERK": 1,
    "totNFB": 1,
    "totKTR": 1,

    # Other
    "ktr_init": 0.5,
    "rho": 0.0,
}

param_values = [param_defaults[p] for p in param_list]

state_vars = [str(eq.lhs.args[0]) for eq in eqs]
assert len(state_vars) == len(rhs_exprs), "Sanity check for length of non-input nodes"

print(state_vars)
y0 = [initial_cond_defaults[n[0:-3]] for n in state_vars]

# Time span
t_span = (0, 30)
t_eval = np.linspace(*t_span, 300)

# Solve
sol = solve_ivp(lambda t, y: egfr_system(t, y, param_values, light_input),
                t_span, y0, t_eval=t_eval)
