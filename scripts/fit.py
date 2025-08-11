import numpy as np
from simulation import eqs, param_list, t, initial_cond_defaults

from symfit import variables, parameters, Parameter, D, ODEModel, Fit
import sympy as sp

defaults = {
    "RAS_s":0.01,
    "RAF_s":0.1,
    "MEK_s":0.1,
    "ERK_s":0.1,
    "NFB_s":0.1,
    "KTR_s":0.5,
    }

def build_symfit_odemodel(
    eqs,
    t_sympy,
    initials,
    *,
    t0=0.0,
    known_subs=None,
    param_inits=None,
    param_bounds=None,
    include_params=None,
    exclude_params=None,
    method="LSODA",
    **integrator_kwargs
):
    """
    Convert SymPy ODEs into a symfit ODEModel ready for fitting.

    Parameters
    ----------
    eqs : list[sympy.Eq]
        Each equation must be Eq(Derivative(state, t), rhs). The `state` may be
        either a Symbol (x) or an AppliedUndef (x(t)).
    t_sympy : sympy.Symbol
        Time symbol used in `eqs` (e.g., t).
    initials : dict
        Initial conditions mapping for states at t0. Keys may be:
          - state names as str, e.g. "ERK_s"
          - SymPy Symbol, e.g. Symbol("ERK_s")
          - SymPy AppliedUndef, e.g. Function("ERK_s")(t)
        Values are floats.
    t0 : float, default 0.0
        Initial time.
    known_subs : dict, optional
        Symbolic substitutions applied to RHS *before* parameter detection,
        e.g. {Function("light")(t): 1.0, Symbol("totERK"): 1.0}. Values may be
        numeric or SymPy expressions in terms of `t_sympy`.
    param_inits : dict[str, float], optional
        Initial guesses for parameters by name.
    param_bounds : dict[str, tuple(float, float)], optional
        Bounds for parameters by name: {"k": (lo, hi), ...}.
    include_params : set[str], optional
        If provided, restrict parameters to this set of names (after detection).
    exclude_params : set[str], optional
        Names to exclude from parameterization (treated as known).
    method : str, default "LSODA"
        SciPy integrator name passed to symfit.ODEModel.
    **integrator_kwargs
        Additional kwargs for SciPy's solver (e.g., rtol, atol, max_step).

    Returns
    -------
    ode : symfit.ODEModel
    state_vars : dict[str, symfit.Variable]
        Mapping from state name to symfit Variable.
    param_objs : dict[str, symfit.Parameter]
        Mapping from parameter name to symfit Parameter.
    """
    import sympy as sp
    from symfit import variables, Parameter, D, ODEModel

    known_subs = known_subs or {}
    param_inits = param_inits or {}
    param_bounds = param_bounds or {}
    include_params = set(include_params) if include_params else None
    exclude_params = set(exclude_params) if exclude_params else set()

    # ---- 1) Extract states (order is the order of eqs)
    lhs_info = []  # [(state_name:str, rhs_expr:SymPy, state_obj)]
    state_names = []
    for eq in eqs:
        if not isinstance(eq.lhs, sp.Derivative):
            raise ValueError(f"LHS must be Derivative(..., t). Got: {eq.lhs}")
        deriv_arg = eq.lhs.args[0]
        # Accept x(t) or x
        if isinstance(deriv_arg, sp.core.function.AppliedUndef) and len(deriv_arg.args) == 1 and deriv_arg.args[0] == t_sympy:
            name = deriv_arg.func.__name__
        elif isinstance(deriv_arg, sp.Symbol):
            name = deriv_arg.name
        else:
            raise ValueError(f"State must be x(t) or Symbol x. Got: {deriv_arg}")

        # Apply known substitutions on RHS early (e.g., light(t) -> constant)
        rhs_proc = eq.rhs.subs(known_subs) if known_subs else eq.rhs
        lhs_info.append((name, rhs_proc, deriv_arg))
        if name not in state_names:
            state_names.append(name)

    # ---- 2) Create symfit variables for time and states
    (t_fit,) = variables("t")
    if state_names:
        state_vars_tuple = variables(", ".join(state_names))
    else:
        state_vars_tuple = tuple()
    state_vars = {name: var for name, var in zip(state_names, state_vars_tuple)}

    # Substitute states in RHS: replace BOTH x(t) and bare x -> symfit var
    subs_states = {}
    for name in state_names:
        subs_states[sp.Function(name)(t_sympy)] = state_vars[name]
        subs_states[sp.Symbol(name)] = state_vars[name]

    # ---- 3) Detect parameter symbols: free symbols not in {t, states}
    rhs_syms = set()
    for _, rhs, _ in lhs_info:
        rhs_syms |= rhs.free_symbols
    rhs_syms.discard(t_sympy)
    for name in state_names:
        rhs_syms.discard(sp.Symbol(name))

    # After known_subs, some symbols may still be there; filter by include/exclude
    candidate_param_syms = sorted(rhs_syms, key=lambda s: s.name)
    if include_params is not None:
        candidate_param_syms = [s for s in candidate_param_syms if s.name in include_params]
    if exclude_params:
        candidate_param_syms = [s for s in candidate_param_syms if s.name not in exclude_params]

    # Build symfit Parameter objects
    param_objs = {s.name: Parameter(s.name) for s in candidate_param_syms}
    for name, val in param_inits.items():
        if name in param_objs:
            param_objs[name].value = float(val)
    for name, (lo, hi) in param_bounds.items():
        if name in param_objs:
            param_objs[name].min = lo
            param_objs[name].max = hi

    sympy_to_param = {sp.Symbol(n): p for n, p in param_objs.items()}

    # ---- 4) Build ODE dictionary { D(x, t): RHS_in_symfit }
    ode_dict = {}
    for name, rhs, _ in lhs_info:
        rhs_sf = rhs.xreplace(subs_states)
        if sympy_to_param:
            rhs_sf = rhs_sf.xreplace(sympy_to_param)
        ode_dict[D(state_vars[name], t_fit)] = rhs_sf

    # ---- 5) Initial conditions map
    init_map = {t_fit: float(t0)}

    def _key_to_name(k):
        if isinstance(k, str):
            return k
        if isinstance(k, sp.core.function.AppliedUndef):
            return k.func.__name__
        if isinstance(k, sp.Symbol):
            return k.name
        raise ValueError(f"Initials key must be str, Symbol or f(t). Got: {k!r}")

    for k, v in initials.items():
        nm = _key_to_name(k)
        if nm not in state_vars:
            raise ValueError(f"Initial for unknown state '{nm}'. Known states: {list(state_vars)}")
        init_map[state_vars[nm]] = float(v)

    # ---- 6) Build symfit ODEModel
    ode = ODEModel(ode_dict, initial=init_map, method=method, **integrator_kwargs)
    return ode, state_vars, param_objs


ode, sv, po = build_symfit_odemodel( eqs, t, defaults)
