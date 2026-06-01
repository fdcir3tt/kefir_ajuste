import torch 
import deepxde as dde



def multi_polynomial(t:torch.Tensor,I:torch.Tensor,T:torch.Tensor, coef:list[dde.Variable],grade:int)->torch.Tensor:
    """
    Evaluate a 2D polynomial correction term over intensity and exposure time.

    This function constructs a polynomial surface of degree ``grade`` in the
    variables intensity (I) and temperature/exposure time (T), using trainable
    coefficients.

    Parameters
    ----------
    t : torch.Tensor
        Time variable (not directly used in computation but kept for API
        consistency with other correction functions).
    I : torch.Tensor
        Intensity input tensor of shape (batch_size,).
    T : torch.Tensor
        Exposure/temperature tensor of shape (batch_size,).
    coef : list of dde.Variable
        List of trainable coefficients representing the polynomial weights.
    grade : int
        Maximum polynomial degree. Only terms satisfying i + j <= grade are used.

    Returns
    -------
    torch.Tensor
        Output tensor of shape (batch_size, 1) representing the evaluated
        polynomial correction.

    Notes
    -----
    The polynomial is evaluated as:

    .. math::

        \\sum_{i,j} c_{i,j} \\cdot I^i \\cdot T^j, \\quad i + j \\leq \\text{grade}

    The coefficient list is reshaped into a (grade+1, grade+1) matrix where
    entry [i, j] corresponds to the weight of the I^i * T^j term.
    """
    coef_tensor = torch.stack([v for v in coef]).view(grade+1, grade+1)

    batch_size = I.shape[0]

    I = I.view(batch_size, 1)
    T = T.view(batch_size, 1)
        
    result = 0.0
    #print(I)

    for i in range(grade+1):
        for j in range(grade+1):
            if i + j <= grade:
                result += coef_tensor[i, j] * (I[:, 0] ** i) * (T[:, 0] ** j)

    return result.view(-1, 1)

def intensity_function(t:torch.Tensor,I:torch.Tensor,T:torch.Tensor, coef:list[dde.Variable])->torch.Tensor:
    """
    Compute a sinusoidally modulated intensity-based correction function.

    This function models a time-dependent signal where the amplitude depends
    on a linear interaction between intensity (I) and exposure time (T),
    and the temporal dynamics are given by a sine function.

    Parameters
    ----------
    t : torch.Tensor
        Time variable tensor.
    I : torch.Tensor
        Intensity input tensor.
    T : torch.Tensor
        Exposure/temperature tensor.
    coef : list of dde.Variable
        List of four trainable coefficients:

        - ``coef[0]``: bias term.
        - ``coef[1]``: intensity scaling.
        - ``coef[2]``: exposure time scaling.
        - ``coef[3]``: interaction term (I * T).

    Returns
    -------
    torch.Tensor
        Tensor of shape (batch_size, 1) representing the corrected signal.

    Notes
    -----
    The output is computed as:

    .. math::

        (c_0 + c_1 I + c_2 T + c_3 I T) \\cdot \\sin\\!\\left(\\frac{2\\pi t}{12}\\right)
    """
    intensity = (coef[0]+coef[1]*I+coef[2]*T+coef[3]*I*T)
    sine_term = torch.sin(2 * torch.pi * t / 12)
    return intensity*sine_term

def fourier_term(t:torch.Tensor,I:torch.Tensor,T:torch.Tensor, coef:list[dde.Variable])->torch.Tensor:
    """
    Compute a Fourier-modulated correction term.

    This function defines a nonlinear interaction between intensity (I),
    exposure time (T), and periodic temporal dynamics using sine functions.

    Parameters
    ----------
    t : torch.Tensor
        Time variable tensor.
    I : torch.Tensor
        Intensity input tensor.
    T : torch.Tensor
        Exposure/temperature tensor.
    coef : list of dde.Variable
        List of two trainable coefficients:

        - ``coef[0]``: amplitude scaling.
        - ``coef[1]``: frequency scaling applied to T inside the sine.

    Returns
    -------
    torch.Tensor
        Tensor of shape (batch_size, 1) representing the correction term.

    Notes
    -----
    The output is computed as:

    .. math::

        c_0 \\cdot I \\cdot \\sin(c_1 T) \\cdot \\sin\\!\\left(\\frac{2\\pi t}{15}\\right)
    """
    intensity = coef[0] * I * torch.sin(coef[1]* T)
    sine_term = torch.sin(2 * torch.pi * t / 15)
    return intensity*sine_term


    
