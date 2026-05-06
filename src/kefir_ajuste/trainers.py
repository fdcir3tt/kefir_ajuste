import os
import numpy as np
import pandas as pd
import deepxde as dde
import torch 

from typing import Callable
from kefir_ajuste.collocation_methods import identity_collocation
from kefir_ajuste.utils import get_learned_parameters,split_train_data
from kefir_ajuste.data import load_initial_conditions,load_time_domain
from pathlib import Path

VARIABLES_PATH = Path("learned_parameters.dat")

os.environ["DDE_BACKEND"] = "pytorch"
dde.backend.set_default_backend("pytorch")

def verhulst(dataset:pd.DataFrame,
            epochs: int = 15000,
            lr: float = 0.001,
            collocation_method: Callable= identity_collocation,
            **kwargs):
    """
    
    """


# ============================================================
#                         CARGAR DATOS
# ============================================================
    
    X_train, y_train, X_test, y_test = split_train_data(dataset)
    t_train =  X_train[:, 2].reshape(-1, 1)
    t_test =  X_test[:, 2].reshape(-1, 1)
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
    anchor_t = anchor_t.reshape(-1, 1)
    observe_bc= dde.icbc.PointSetBC(
                                    anchor_t.astype(np.float32),
                                    observe_y.astype(np.float32),
                                    component=0,
                                    shuffle=False
                                )

    data_pinn = dde.data.PDE(
        geometry=geom,
        pde=ode,
        bcs=[ observe_bc],
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
        loss='MSE',
        lr=lr,
        external_trainable_variables=[r, k],
    )
    
    variable = dde.callbacks.VariableValue(
                                        var_list=[r,k], 
                                        period=600, 
                                        filename=VARIABLES_PATH
                                    )
    callbacks = [variable]
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


def multi_polynomial(t:torch.Tensor,I:torch.Tensor,T:torch.Tensor, coef:list[dde.Variable],grade:int)->torch.Tensor:
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
    intensity = (coef[0]+coef[1]*I+coef[2]*T+coef[3]*I*T)
    sine_term = torch.sin(2 * torch.pi * t / 12)
    return intensity*sine_term

def fourier_term(t:torch.Tensor,I:torch.Tensor,T:torch.Tensor, coef:list[dde.Variable])->torch.Tensor:
    intensity = coef[0] * I * torch.sin(coef[1]* T)
    sine_term = torch.sin(2 * torch.pi * t / 15)
    return intensity*sine_term


    
