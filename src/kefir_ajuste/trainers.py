import os
import numpy as np
import pandas as pd
import deepxde as dde
import random
import torch 

from typing import Callable
from kefir_ajuste.utils import get_learned_parameters,identity_collocation,\
                               load_train_data,load_initial_conditions,load_time_domain,\
                               split_train_data
from pathlib import Path
from deepxde.icbc.boundary_conditions import PointSetBC

VARIABLES_PATH = Path("learned_parameters.dat")

os.environ["DDE_BACKEND"] = "pytorch"
dde.backend.set_default_backend("pytorch")

def verhulst(
    dataset:pd.DataFrame,
    epochs: int = 15000,
    lr: float = 0.001,
    collocation_method: Callable= identity_collocation,
    **kwargs
):
    """
    
    """


# ============================================================
#                         CARGAR DATOS
# ============================================================
    
    t_train, y_train, t_test, y_test = load_train_data(dataset)
    t0,y0 = load_initial_conditions(dataset)
    t0,tf = load_time_domain(dataset)

# ============================================================
#               CONFIGURACION ENTRENAMIENTO
# ============================================================
    
    r = dde.Variable(0.04)
    k = dde.Variable(51.0)

    def ode(t, y):
        dy_dt = dde.grad.jacobian(y, t, i=0, j=0)
        return dy_dt - r * y * (1 - y / k)

    geom = dde.geometry.TimeDomain(t0, tf)

    ic = dde.icbc.IC(
        geom,
        lambda t: y0,
        lambda _, on_initial: on_initial,
    )
    if collocation_method.__name__ == "all_data_collocation":
        y = np.concatenate([y_train,y_test])
        t = np.concatenate([t_train,t_test])
        anchor_t,observe_y = collocation_method(t,y,**kwargs)
    else:   
        anchor_t,observe_y = collocation_method(t_train,y_train,**kwargs)
    


    data_pinn = dde.data.PDE(
        geometry=geom,
        pde=ode,
        bcs=[ic, observe_y],
        num_domain=200,
        num_boundary=2,
        num_test=100,
        anchors=anchor_t,
    )

# ============================================================
#                         RED NEURONAL
# ============================================================

    layer_size = [1, 50, 50, 50, 1]

    net = dde.nn.FNN(
        layer_size,
        activation="tanh",
        kernel_initializer="Glorot uniform",
    )

    model = dde.Model(data_pinn, net)

    model.compile(
        optimizer="adam",
        lr=lr,
        external_trainable_variables=[r, k],
    )
    
    variable = dde.callbacks.VariableValue(
                                        var_list=[r,k], 
                                        period=600, 
                                        filename=VARIABLES_PATH
                                    )
    callbacks = [
        variable
    ]
# ============================================================
#                        ENTRENAMIENTO
# ============================================================
    loss_history, _ = model.train(iterations=epochs,
                                 callbacks=callbacks)
    

    learned_params = get_learned_parameters(model='verhulst')
    os.remove(VARIABLES_PATH)
    y_true = y_test
    y_pred = model.predict(t_test)
    return model, loss_history, learned_params, y_true, y_pred


        
def multi_polynomial_model(
    dataset:pd.DataFrame,
    grade: int,
    epochs: int,
    collocation_method:Callable = lambda x,y:PointSetBC(x,y),
    lr: float = 0.001,
    **kwargs

):

# ============================================================
#                         CARGAR DATOS
# ============================================================

    t_train, y_train, t_test, y_test = split_train_data(dataset)
    t0,y0 = load_initial_conditions(dataset)
    t0,tf = load_time_domain(dataset)

# ============================================================
#                 CONFIGURACION ENTRENAMIENTO
# ============================================================
    
    p_coef = [dde.Variable(float(torch.rand(1))) for _ in range((grade+1) * (grade+1))]
    for i in range(grade+1):
        for j in range(grade+1):
            if i + j > grade:
                index = i * (grade + 1) + j  # Calculate index for flattened 2D array
                p_coef[index] = dde.Variable(torch.tensor(0.0))  # Set value to 0

    kappa = 0.046 
    L = 47.81 
    
    w = dataset["intensidad(W/cm^2)"].iloc[0]
    T_period = dataset["periodo de exposición(s)"].iloc[0]
    variable_path=Path('learned_parameters.dat')
    trainable_variables = p_coef
    def multi_polynomial(x: float, y: float, coef: list[torch.Tensor], grade: int):
        # Rebuild the 2D coefficient matrix from the flattened vector
        coef_tensor = torch.stack([v for v in coef]).view(grade+1, grade+1)  # Flatten and reshape

        rows, cols = coef_tensor.shape

        i = torch.arange(rows).view(-1, 1)
        j = torch.arange(cols).view(1, -1)
        mask = (i + j) <= grade

        return torch.sum(coef_tensor * (x ** i) * (y ** j) * mask)

    def ode(t, y):
        dy_dt = dde.grad.jacobian(y, t, i=0, j=0)

        w_t = torch.tensor(w,dtype=torch.float32)
        T_t = torch.tensor(T_period,dtype=torch.float32)

        poly_term = multi_polynomial(w_t, T_t, p_coef,grade)

        return dy_dt - kappa * y * (1 - y / L) - poly_term

# ============================================================
#                         PINN SETUP
# ============================================================

    geom = dde.geometry.TimeDomain(t0, tf)

    ic = dde.icbc.IC(
            geom,
            lambda t: y0,
            lambda _, on_initial: on_initial,
        )
    
    # Colocación de puntos de entrenamiento 

    if collocation_method.__name__ == "all_data_collocation":
        y = np.concatenate([y_train,y_test])
        t = np.concatenate([t_train,t_test])
        anchor_t,observe_y = collocation_method(t,y,**kwargs)
    else:   
        anchor_t,observe_y = collocation_method(t_train,y_train,**kwargs)

    data_pinn = dde.data.PDE(
            geometry=geom,
            pde=ode,
            bcs=[ic, observe_y],
            num_domain=200,
            num_boundary=2,
            num_test=100,
            anchors=anchor_t,
        )

    net = dde.nn.FNN([1, 50, 50, 50, 1], "tanh", "Glorot uniform")
    model = dde.Model(data_pinn, net)

    model.compile(
            optimizer="adam",
            lr=lr,
            external_trainable_variables=trainable_variables
        )
    variable = dde.callbacks.VariableValue(
                                        var_list=trainable_variables, 
                                        period=600, 
                                        filename=variable_path
                                    )
    callbacks = [
        variable
    ]

# ============================================================
#                        ENTRENAMIENTO
# ============================================================
    loss_history, _ = model.train(iterations=epochs,
                                 callbacks=callbacks)
    
    learned_params = get_learned_parameters(model=f'verhulst_multi_polynomial',
                                            n=grade)
    learned_params ["learning_rate"] = lr
    learned_params ["initial_rate"] = kappa
    learned_params ["initial_saturation_concentration"] = L
    os.remove(variable_path)
    y_true = y_test
    y_pred = model.predict(t_test)
        
        
    return model, loss_history, learned_params, y_true, y_pred
        

    
