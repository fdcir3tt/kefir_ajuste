import torch

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
    kappa = parameters["kappa"]
    L = parameters["L"]
    return dy_dt - kappa * y * (1 - y / L)

