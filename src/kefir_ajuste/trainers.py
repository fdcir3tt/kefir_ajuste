import os
import numpy as np
import pandas as pd
import deepxde as dde
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

def intensity_function(I, T, coef,t):
    intensity = (coef[0]+coef[1]*I+coef[2]*T+coef[3]*I*T)
    sine_term = torch.sin(2 * torch.pi * t / 15)
    return intensity*sine_term

def multi_polynomial(I, T, coef, t,grade):
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
        
def physics_discovery(
    dataset:pd.DataFrame,
    epochs: int,
    correction_function:Callable,
    collocation_method:Callable = lambda x,y:PointSetBC(x,y),
    lr: float = 0.001,
    **kwargs

):

# ============================================================
#                         CARGAR DATOS
# ============================================================

    
    t0,y0 = load_initial_conditions(dataset)
    t0,tf = load_time_domain(dataset)

    X_train, y_train, X_test,y_test = split_train_data(dataset)
# ============================================================
#                 CONFIGURACION ENTRENAMIENTO
# ============================================================
    
    if correction_function.__name__=="multi_polynomial":
        grade = kwargs.get("grade")
        c_coef = [dde.Variable(float(torch.rand(1))) for _ in range((grade+1) * (grade+1))]
        for i in range(grade+1):
            for j in range(grade+1):
                if i + j > grade:
                    index = i * (grade + 1) + j  # Calculate index for flattened 2D array
                    c_coef[index] = dde.Variable(torch.tensor(0.0))  # Set value to 0

    if correction_function.__name__=="intensity_function":
        c_coef = [dde.Variable(torch.rand(1)) for _ in range(4)]

    kappa = 0.046 
    L = 47.81 
    variable_path=Path('learned_parameters.dat')
    trainable_variables = c_coef

    

    def ode(x, y):
        I_t = x[:, 0:1]
        T_t = x[:, 1:2]
        t = x[:, 2:3]

        dy_dt = dde.grad.jacobian(y, x, i=0, j=2)

        delta = correction_function(I_t, T_t, c_coef,t,**kwargs)

        return dy_dt - kappa * y * (1 - y / L) - delta

# ============================================================
#                         PINN SETUP
# ============================================================
    
    t_min, t_max = float(t0), float(tf)
    I_min, I_max = X_train[:, 0].min(), X_train[:, 0].max()
    T_min, T_max = X_train[:, 1].min(), X_train[:, 1].max()
    
    geom_space = dde.geometry.Rectangle(
    [I_min, T_min],
    [I_max, T_max]
)
    timedomain = dde.geometry.TimeDomain(t_min, t_max)
    geom = dde.geometry.GeometryXTime(geom_space, timedomain)
    
    # Colocación de puntos de entrenamiento 
    if collocation_method.__name__ == "all_data_collocation":
        y = np.concatenate([y_train,y_test])
        X = np.vstack((X_train, X_test))
    
        anchor_X, observe_y = collocation_method(X, y, **kwargs)
    else:   
        anchor_X,observe_y = collocation_method(X_train,y_train,**kwargs)
    
    observe_bc = dde.icbc.PointSetBC(
                                    anchor_X.astype(np.float32),
                                    observe_y.astype(np.float32),
                                    component=0,
                                    shuffle=False
                                )
    data_pinn = dde.data.PDE(
            geometry=geom,
            pde=ode,
            bcs=[observe_bc],
            num_domain=200,       # however many PDE collocation points you want
            num_boundary=0,
            anchors=anchor_X.astype(np.float32),
        )

    net = dde.nn.FNN([3, 50, 50, 50, 1], "tanh", "Glorot uniform")
    model = dde.Model(data_pinn, net)
    print("Train X shape:", data_pinn.train_x.shape)
    print("Anchor count:", anchor_X.shape[0])
    print("Observe y count:", observe_y.shape[0])
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
    if correction_function.__name__=="multi_polynomial":
        learned_params = get_learned_parameters(model=correction_function.__name__,
                                                n=grade)
    if correction_function.__name__=="intensity_function":
        learned_params = get_learned_parameters(model=correction_function.__name__)
    learned_params ["learning_rate"] = lr
    learned_params ["initial_rate"] = kappa
    learned_params ["initial_saturation_concentration"] = L
    os.remove(variable_path)
    y_true = y_test
    y_pred = model.predict(X_test)
        
        
    return model, loss_history, learned_params, y_true, y_pred
        

    
