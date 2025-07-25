from sympy import symbols, Function, Derivative, Eq, Symbol
import numpy as np
from scipy.integrate import solve_ivp as solve
import matplotlib.pyplot as plt

# Parameters
params = {
    'r12': 1.0,
    'r21': 0.5,
    'r23': 0.3,
    'r31': 0.2,
}
param_list = ["r12", "r23", "r31", "r21",
"k12", "K12", "k21", "K21",
"k34", "K34","knfb","k43","K43",
"k56", "K56","k65","K65",
"k78", "K78","k87","K87",
"k910","K910",
"s12", "s21",
"f12", "F12", "f21", "F21",
"ktr_init", "rho"]

# horrible horrible hack. TODO: When i know more about code like this, re-factor this into something more sane.
for p in param_list: globals()[p] = Symbol(p)

nodes = ["RAS", "RAF", "MEK", "ERK", "EGFR", "NFB", "KTR"] 
nodes.extend(*[[f'{b}_s' for b in nodes]])
nodes.append('EGFR_endo')

t = symbols('t')
for n in nodes: globals()[n] = Function(n)(t)

#r12, r21, r23, r31 = symbols("r12 r21 r23 r31")
eqs = [dEGFR_dt := Eq(Derivative(EGFR, t),          -r12 * EGF * EGFR + r21 * EGFR_s + r31 * EGFR_endo),
    dEGFRs_dt := Eq(Derivative(EGFR_s,t),           r12 * EGF * EGFR - (r21 + r31 ) * EGFR_s),
    dEGFR_endo_dt := Eq(Derivative(EGFR_endo, t),   r23 * EGFR_s - r31 * EGFR_endo ),
    dRAS_dt := Eq(Derivative(RAS, t ),              -(k12*EGFR_s+light) * (RAS/(K12+RAS)) + k21 * (RAS_s/(K21+RAS_s)),
    dRASs_dt:= Eq(Derivative(RAS_s, t ),            (k12*EGFR_s+light) * (RAS/(K12+RAS)) - k21 * (RAS_s/(K21+RAS_s)),
    dRAF_dt := Eq(Derivative(RAF, t ),              -(k34*RAS_s) * (RAF/(K34+RAF)) + (knfb * NFB_s + k43) * (RAF_s/(K43+RAF_s)),
    dRAFs_dt := Eq(Derivative(RAF_s, t ),           k34 * RAS_s * (RAF/(K34+RAF)) - (knfb * NFB_s + k43) * (RAF_s/(K43+RAF_s)),
    dMEK_dt := Eq(Derivative(MEK, t),               -k56 * RAF_s * (MEK/(K56+MEK)) + k65 * (MEK_s/(K65+MEK_s)),
    dMEKs_dt := Eq(Derivative(MEK_s, t),            k56 * RAF_s * (MEK/(K56+MEK)) - k65 * (MEK_s/(K65+MEK_s)),
    dERK_dt := Eq(Derivative(ERK, t),               -k78 * MEK_s * (ERK/(K78+ERK)) + k87 * (ERK_s/(K87+ERK_s)),
    dERKs_dt := Eq(Derivative(ERK_s, t),            k78 * MEK_s * (ERK/(K78+ERK)) - k87 * (ERK_s/(K87+ERK_s)),

    dNFB_dt := Eq(Derivative(NFB, t),               -f12 * ERK_s * (NFB/F12+NFB) + f21*(NFB_s/(F21+NFB_s)),
    dNFBs_dt := Eq(Derivative(NFB_s, t),            f12 * ERK_s * (NFB/F12+NFB) - f21*(NFB_s/(F21+NFB_s)),

    dKTR_dt := Eq(Derivative(KTR, t),               -(k910*ERK_s*(KTR/(K910+KTR)) + s12*KTR) + s21 * KTR_s,
    dKTRs_dt := Eq(Derivative(KTR_s, t),            (k910*ERK_s*(KTR/(K910+KTR)) + s12*KTR) - s21 * KTR_s
]

rhs_exprs = [e.rhs for e in eqs]

"""
What's my goal here?

Define individual DEs, inputs, parameters,
and for a given input simulate pathway to produce an output.
Have an introspection to every latent variable, maybe plot selected ones.
"""


# Flat symbolic variables
#EGFR_, EGFR_s_, EGFR_endo_ = symbols("EGFR EGFR_s EGFR_endo")
for n in nodes: globals()[f'{n}_'] = Symbol(n) 

t_ = Symbol("t")
EGF_ = Function("EGF")(t_)  # Keep EGF(t_) as a time-varying input

# TODO: figure out a way to automatically generate the 'flat' symbols, and its dependants
# basic ideae is easy and tempting, but it gets harder when you realise that input nodes can be
# anywhere in the network, and so basic layering is hard to do. 
# The ideal API for this would be to
# INPUT: DEs, annotate inputs & parameters
# OUTPUT: xyz_system function, or at least lambdified set of equations.
# For now however, doing it manually is fine.

#symbol_list = "EGFR EGFR_s EGFR_endo"
#flat_sb = {k:f'{k}_' for k in symbol_list.split(' ')}

subs_map = {
    EGFR: EGFR_,
    EGFR_s: EGFR_s_,
    EGFR_endo: EGFR_endo_,
    t: t_,
    EGF: EGF_
}

from sympy import lambdify

# Parameters list
parameters = [r12, r21, r23, r31]

# Full argument list: t, states, params, input
arg_list = (t_, EGFR_, EGFR_s_, EGFR_endo_, *parameters, EGF_)

# Lambdified RHS functions
rhs_funcs = [
    lambdify(arg_list, rhs.subs(subs_map), modules='numpy')
    for rhs in rhs_exprs
]


def egfr_system(t, y, param_values, EGF_func):
    EGFR, EGFR_s, EGFR_endo = y
    EGF_val = EGF_func(t)

    args = [t, EGFR, EGFR_s, EGFR_endo, *param_values, EGF_val]

    return [f(*args) for f in rhs_funcs]


from scipy.integrate import solve_ivp
import numpy as np

# Parameter values
param_values = [1.0, 0.5, 0.3, 0.2]  # r12, r21, r23, r31

# Input function
EGF_input = lambda t: 2.0 if t < 10 else 0.5

# Initial conditions
y0 = [1.0, 0.0, 0.0]  # [EGFR, EGFR_s, EGFR_endo]

# Time span
t_span = (0, 30)
t_eval = np.linspace(*t_span, 300)

# Solve
sol = solve_ivp(lambda t, y: egfr_system(t, y, param_values, EGF_input),
                t_span, y0, t_eval=t_eval)

print(type(sol))

import matplotlib.pyplot as plt

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


# Example usage
labels = ["EGFR", "EGFR_s", "EGFR_endo"]
plot_ivp_solution(sol, labels=labels, title="EGFR Pathway Dynamics")

