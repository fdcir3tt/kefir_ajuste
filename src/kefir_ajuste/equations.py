import torch
import numpy as np
import pymc as pm

def verhulst_eq(x:float,y:float,parameters:dict[str,float])->float:
    r = parameters.get("r")
    m = parameters.get("m")
    return r * y * (1 - y / m)

def gompertz_eq_solution(x,parameters:dict[str,float]):
    K  = parameters.get("K")
    Y0 = parameters.get("Y0")
    r  = parameters.get("r")
    return K * np.exp(-np.log(K / Y0) * np.exp(-r * x))

def gompertz_latent_eq(x,parameters:dict[str,float]):
    K  = parameters.get("K")
    Y0 = parameters.get("Y0")
    r  = parameters.get("r")
    return K * pm.math.exp(-pm.math.log(K / Y0) * pm.math.exp(-r * x))


def verhulst(dy_dt:torch.Tensor,t:float,y:float,parameters:dict[str,float])->float:
    """
    Evaluate the residual of the Verhulst (logistic growth) ODE.

    Computes the residual of the logistic growth equation, which equals
    zero when the solution satisfies the ODE exactly. This form is
    suitable for use as a physics-informed residual in a PINN framework.

    Parameters
    ----------
    dy_dt : torch.Tensor
        Time derivative of the state variable, typically computed via
        automatic differentiation.
    t : float
        Current time value (not directly used but kept for API
        consistency with other ODE residual functions).
    y : float
        Current value of the state variable (population or concentration).
    parameters : dict[str, float]
        Dictionary of model parameters containing:

        - ``"kappa"`` : float
            Intrinsic growth rate.
        - ``"L"`` : float
            Carrying capacity.

    Returns
    -------
    float
        Residual value of the ODE. Returns zero when ``y`` exactly
        satisfies the logistic equation.

    Notes
    -----
    The Verhulst logistic equation is defined as:

    .. math::

        \\frac{dy}{dt} = \\kappa \\, y \\left(1 - \\frac{y}{L}\\right)

    The residual returned by this function is:

    .. math::

        r = \\frac{dy}{dt} - \\kappa \\, y \\left(1 - \\frac{y}{L}\\right)
    """
    r = parameters["r"]
    m = parameters["m"]
    return dy_dt - r * y * (1 - y / m)
