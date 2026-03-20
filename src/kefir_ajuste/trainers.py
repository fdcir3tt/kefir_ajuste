import os
import numpy as np
import pandas as pd
import deepxde as dde
import random
from kefir_ajuste.utils import get_learned_parameters,load_train_data,load_initial_conditions,load_time_domain
from pathlib import Path



os.environ["DDE_BACKEND"] = "pytorch"
dde.backend.set_default_backend("pytorch")

def train_verhulst(
    treatment: int,
    epochs: int = 15000,
    lr: float = 0.001,
):
    """
    Train a PINN model using the Verhulst (logistic) ODE.

    Parameters
    ----------
    treatment : int
        Treatment number (used to load dataset).
    epochs : int
        Training iterations.
    lr : float
        Learning rate.
    data_dir : Path | str
        Directory where tratamiento_X.csv files are stored.

    Returns
    -------
    model : dde.Model
    loss_history : list[float]
    """


# ============================================================
#                         CARGAR DATOS
# ============================================================
    
    t_train, y_train, t_test, y_test = load_train_data(treatment)
    t0,y0 = load_initial_conditions(treatment)
    t0,tf = load_time_domain(treatment)

# ============================================================
#               CONFIGURACION ENTRENAMIENTO
# ============================================================
    variables_path=Path('verhulst_'+str(treatment)+'.dat')
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

    observe_y = dde.icbc.PointSetBC(t_train, y_train)

    data_pinn = dde.data.PDE(
        geometry=geom,
        pde=ode,
        bcs=[ic, observe_y],
        num_domain=200,
        num_boundary=2,
        num_test=100,
        anchors=t_train,
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
                                        filename=variables_path
                                    )
    callbacks = [
        variable
    ]
# ============================================================
#                        ENTRENAMIENTO
# ============================================================
    loss_history, _ = model.train(iterations=epochs,
                                 callbacks=callbacks)
    

    learned_params = get_learned_parameters(model='verhulst',treatment=treatment)
    os.remove(variables_path)
    y_true = y_test
    y_pred = model.predict(t_test)
    return model, loss_history, learned_params, y_true, y_pred

def train_polynomial(
    treatment: int,
    grade: int,
    epochs: int,
    lr: float = 0.001,
):

# ============================================================
#                         CARGAR DATOS
# ============================================================

    t_train, y_train, t_test, y_test = load_train_data(treatment)
    t0,y0 = load_initial_conditions(treatment)
    t0,tf = load_time_domain(treatment)

# ============================================================
#                 CONFIGURACION ENTRENAMIENTO
# ============================================================
    

    intensity_dict = {
            2: {"frequency": 20.0, "period": 15.0},
            3: {"frequency": 20.0, "period": 60.0},
            4: {"frequency": 34.0, "period": 15.0},
            5: {"frequency": 34.0, "period": 60.0},
        }
    variables_path=Path('verhulst_polynomial_'+str(treatment)+'.dat')
    w = intensity_dict[treatment]["frequency"]
    T_period = intensity_dict[treatment]["period"]

    w_coef = [dde.Variable(random.random()) for _ in range(grade)]
    T_coef = [dde.Variable(random.random()) for _ in range(grade)]

    r = dde.Variable(0.04)
    k = dde.Variable(51.0)

    def polynomial(x, coef):
        return sum(c * (x ** i) for i, c in enumerate(coef))

    def ode(t, y):
        dy_dt = dde.grad.jacobian(y, t, i=0, j=0)
        return (
                dy_dt
                - r * y * (1 - y / k)
                - polynomial(w, w_coef)
                - polynomial(T_period, T_coef)
            )

# ============================================================
#                         PINN SETUP
# ============================================================

    geom = dde.geometry.TimeDomain(t0, tf)

    ic = dde.icbc.IC(
            geom,
            lambda t: y0,
            lambda _, on_initial: on_initial,
        )

    observe_y = dde.icbc.PointSetBC(t_train, y_train)

    data_pinn = dde.data.PDE(
            geometry=geom,
            pde=ode,
            bcs=[ic, observe_y],
            num_domain=200,
            num_boundary=2,
            num_test=100,
            anchors=t_train,
        )

    net = dde.nn.FNN([1, 50, 50, 50, 1], "tanh", "Glorot uniform")
    model = dde.Model(data_pinn, net)

    model.compile(
            optimizer="adam",
            lr=lr,
            external_trainable_variables=[r, k] + w_coef + T_coef,
        )
    variable = dde.callbacks.VariableValue(
                                        var_list=[r,k]+ w_coef + T_coef, 
                                        period=600, 
                                        filename=variables_path
                                    )
    callbacks = [
        variable
    ]

# ============================================================
#                        ENTRENAMIENTO
# ============================================================
    loss_history, _ = model.train(iterations=epochs,
                                 callbacks=callbacks)
    
    learned_params = get_learned_parameters(model='verhulst_polynomial',
                                            treatment=treatment,
                                            n=grade+1,
                                            m=2*grade+1)
    os.remove(variables_path)
    y_true = y_test
    y_pred = model.predict(t_test)
        
        
    return model, loss_history, learned_params, y_true, y_pred
        


def train_two_variable_polynomial(
    treatment: int,
    grade: int,
    epochs: int,
    lr: float = 0.001,
):

# ============================================================
#                         CARGAR DATOS
# ============================================================

    t_train, y_train, t_test, y_test = load_train_data(treatment)
    t0,y0 = load_initial_conditions(treatment)
    t0,tf = load_time_domain(treatment)

# ============================================================
#                 CONFIGURACION ENTRENAMIENTO
# ============================================================
    

    intensity_dict = {
            2: {"frequency": 20.0, "period": 15.0},
            3: {"frequency": 20.0, "period": 60.0},
            4: {"frequency": 34.0, "period": 15.0},
            5: {"frequency": 34.0, "period": 60.0},
        }
    variables_path=Path('verhulst_multi_polynomial_'+str(treatment)+'.dat')
    w = intensity_dict[treatment]["frequency"]
    T_period = intensity_dict[treatment]["period"]

    p_coef = np.array([
        [dde.Variable(random.random()) if i + j <= grade else 0 for j in range(grade)] 
        for i in range(grade)])   

    

    r = dde.Variable(0.04)
    k = dde.Variable(51.0)

    def multi_polynomial(x:float,y:float,coef:np.ndarray):
        result=0
        rows, cols = coef.shape

        for i in range(rows):
            for j in range(cols):
                result+=coef[i,j]*(x**i)*(y**j)
        return result

    def ode(t, y):
        dy_dt = dde.grad.jacobian(y, t, i=0, j=0)
        return (
                dy_dt
                - r * y * (1 - y / k)
                - multi_polynomial(w,T_period, p_coef)
                
            )

# ============================================================
#                         PINN SETUP
# ============================================================

    geom = dde.geometry.TimeDomain(t0, tf)

    ic = dde.icbc.IC(
            geom,
            lambda t: y0,
            lambda _, on_initial: on_initial,
        )

    observe_y = dde.icbc.PointSetBC(t_train, y_train)

    data_pinn = dde.data.PDE(
            geometry=geom,
            pde=ode,
            bcs=[ic, observe_y],
            num_domain=200,
            num_boundary=2,
            num_test=100,
            anchors=t_train,
        )

    net = dde.nn.FNN([1, 50, 50, 50, 1], "tanh", "Glorot uniform")
    model = dde.Model(data_pinn, net)

    model.compile(
            optimizer="adam",
            lr=lr,
            external_trainable_variables=[r, k] + p_coef,
        )
    variable = dde.callbacks.VariableValue(
                                        var_list=[r,k]+ p_coef, 
                                        period=600, 
                                        filename=variables_path
                                    )
    callbacks = [
        variable
    ]

# ============================================================
#                        ENTRENAMIENTO
# ============================================================
    loss_history, _ = model.train(iterations=epochs,
                                 callbacks=callbacks)
    
    learned_params = get_learned_parameters(model='verhulst_multi_polynomial',
                                            treatment=treatment)
    os.remove(variables_path)
    y_true = y_test
    y_pred = model.predict(t_test)
        
        
    return model, loss_history, learned_params, y_true, y_pred
        

    
