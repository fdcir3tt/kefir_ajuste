import os
import mlflow
import deepxde as dde
import numpy as np
import torch

from pathlib import Path
from typing import Callable
from mlflow.data.pandas_dataset import PandasDataset
from kefir_ajuste.utils import get_learned_parameters,log_run,ensure_experiment_active,plot_inverse_problem_solution
from kefir_ajuste.collocation_methods import identity_collocation
from kefir_ajuste.data import load_data,split_train_data,load_initial_conditions,load_time_domain
from kefir_ajuste.trainers import verhulst

# ==============================================================================
#                               Global config
# ==============================================================================

mlflow.set_tracking_uri("file:./mlruns")
DATA_FILE_NAME = "tratamiento_1.csv"
EXPERIMENT_NAME = "Inverse Problem"
epochs = 1000
lr = 0.01
model_equation = verhulst
collocation_method = identity_collocation
variables_path = Path("learned_parameters.dat")

r = dde.Variable(0.04)
k = dde.Variable(51.0)
trainable_variables = [r,k]
n_phys_points= 200
collocation_args = {}

def make_time_domain(t0:float,tf:float)->dde.geometry.GeometryXTime:
    """ 
    Define problem time domain space.
    
    Parameters
    ----------
    t0: float
        Initial time instant in domain.
    tf: float
        Final point in domain.

    Returns
    -------
    deepxde.geometry.TimeDomain
        Geometry object that represents the time domain in which the problem takes place.

    Examples
    --------
    >>> 
    
    
    """
    t_min, t_max = float(t0), float(tf)
    
    timedomain = dde.geometry.TimeDomain(t_min, t_max)
    
    return timedomain

def make_boundary_conditions(collocation_method:Callable,collocation_args:dict[str,int])->tuple[dde.icbc.PointSetBC,torch.Tensor]:
    """
    Create boundary condition objects from collocation data.

    This function applies a collocation method to training (and optionally test)
    data to generate anchor points and observed values, which are then used to
    construct a ``PointSetBC`` boundary condition object.

    Parameters
    ----------
    collocation_method : Callable
        Function used to generate collocation points and observed values.
        It must accept input data, target values, and a dictionary of arguments,
        and return a tuple ``(anchor_X, observe_y)``.
    collocation_args : dict of str to int
        Dictionary of arguments passed to the collocation method.

    Returns
    -------
    observe_bc : deepxde.icbc.PointSetBC
        Boundary condition object constructed from the collocation points and
        observed values.
    anchor_X : torch.Tensor
        Collocation points used to define the boundary condition, converted
        to ``float32``.

    Notes
    -----
    If the collocation method name is ``"all_data_collocation"``, both training
    and test datasets are combined before applying the method. Otherwise, only
    the training data is used.

    The function assumes the existence of global variables ``X_train``,
    ``y_train``, ``X_test``, and ``y_test``.

    Examples
    --------
    >>> bc, anchors = make_boundary_conditions(method, {"n_points": 100})
    >>> bc
    <dde.icbc.PointSetBC object>
    """
    if collocation_method.__name__ == "all_data_collocation":
        y = np.concatenate([y_train,y_test])
        X = np.vstack((X_train, X_test))
    
        anchor_X, observe_y = collocation_method(X, y, collocation_args)
    else:   
        anchor_X,observe_y = collocation_method(X_train,y_train,collocation_args)
    
    observe_bc = dde.icbc.PointSetBC(anchor_X.astype(np.float32),
                                    observe_y.astype(np.float32),
                                    component=0,
                                    shuffle=False)
    return observe_bc,anchor_X.astype(np.float32)

def make_net(layers:list[int],activation_func:str,weights_init:str,data_config)->dde.Model:
    """
    Construct a neural network model for a PINN.

    This function creates a fully connected neural network (FNN) using the
    specified architecture, activation function, and weight initialization,
    and wraps it into a ``dde.Model`` object with the provided data configuration.

    Parameters
    ----------
    layers : list of int
        List specifying the number of neurons in each layer, including input
        and output layers.
    activation_func : str
        Name of the activation function to use in the network.
    weights_init : str
        Initialization method for the network weights.
    data_config : object
        Data configuration object required by ``dde.Model``.

    Returns
    -------
    deepxde.Model
        Constructed model combining the neural network and data configuration.

    Examples
    --------
    >>> model = make_net([1, 20, 20, 1], "tanh", "Glorot uniform", data)
    """
    net = dde.nn.FNN(layers, activation_func, weights_init)
    model = dde.Model(data_config, net)
    return model
def train_pinn(optimizer_method:str,loss_func:str,learning_rate:float,epochs:int,trainable_variables,learned_period:int,learned_variables_path:Path)->tuple[dde.model.LossHistory,dde.Model]:
    """
    Train a physics-informed neural network (PINN).

    This function compiles a global ``dde.Model`` with the specified optimizer,
    loss function, and learning rate, and trains it for a given number of
    iterations. During training, selected variables are monitored and saved
    periodically via a callback.

    Parameters
    ----------
    optimizer_method : str
        Optimization algorithm to use (e.g., ``"adam"``).
    loss_func : str
        Loss function used during training.
    learning_rate : float
        Learning rate for the optimizer.
    epochs : int
        Number of training iterations.
    trainable_variables : list
        List of external variables to be optimized during training.
    learned_period : int
        Frequency (in iterations) at which variable values are recorded.
    learned_variables_path : pathlib.Path
        File path where the learned variable values will be saved.

    Returns
    -------
    loss_history : dde.model.LossHistory
        Object containing the history of training losses.
    model : dde.Model
        Trained model instance.

    Notes
    -----
    This function assumes that a global variable ``model`` exists and has been
    initialized prior to calling this function.

    Examples
    --------
    >>> history, trained_model = train_pinn(
    ...     "adam", "MSE", 1e-3, 10000, vars, 100, Path("vars.dat")
    ... )
    """
    model.compile(optimizer = optimizer_method,
                  loss = loss_func,
                  lr = learning_rate,
                  external_trainable_variables = trainable_variables)
    
    variable  = dde.callbacks.VariableValue(var_list = trainable_variables, 
                                            period = learned_period, 
                                            filename = learned_variables_path)
    callbacks = [variable]
    loss_history, _ = model.train(iterations=epochs,
                                  callbacks=callbacks)
    return loss_history,model


# ==============================================================================
#                         Polynomial Experiments
# ==============================================================================

ensure_experiment_active(EXPERIMENT_NAME)
mlflow.set_experiment(EXPERIMENT_NAME)

dataset = load_data(DATA_FILE_NAME)
dataset = dataset[["tiempo(h)","concentracion(g/cm3)"]]
dataset_source_url = f"data/processed/{DATA_FILE_NAME}"
mlflow_dataset: PandasDataset = mlflow.data.from_pandas(dataset, 
                                                        source=dataset_source_url,
                                                        targets="concentracion(g/cm3)",
                                                        name=DATA_FILE_NAME)


run_name = f"{model_equation.__name__}_{lr}_{epochs}"
with mlflow.start_run(run_name=run_name):
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("learning_rate", lr)


    X_train, y_train, X_test, y_test = split_train_data(dataset,"concentracion(g/cm3)")
    t_train =  X_train.reshape(-1, 1)
    t_test =  X_test.reshape(-1, 1)
    t0,y0 = load_initial_conditions(dataset)
    t0,tf = load_time_domain(dataset)

# ============================================================
#               CONFIGURACION ENTRENAMIENTO
# ============================================================
    
    
    def ode(t, y):
        dy_dt = dde.grad.jacobian(y, t, i=0, j=0)
        return dy_dt - r * y * (1 - y / k)

    geom = make_time_domain(t0,tf)

    observe_bc,anchor_t = make_boundary_conditions(collocation_method,collocation_args)

    data_pinn = dde.data.PDE(
        geometry=geom,
        pde=ode,
        bcs=[ observe_bc],
        num_domain=n_phys_points,
        num_boundary=2,
        num_test=100,
        anchors=anchor_t,
    )

# ============================================================
#                         RED NEURONAL
# ============================================================

    model = make_net([1, 50, 50, 50, 1],"tanh", "Glorot uniform",data_pinn)
    loss_history, model = train_pinn("adam","MSE",lr,epochs,trainable_variables,600,variables_path)
    learned_parameters = get_learned_parameters(model='verhulst')
    os.remove(variables_path)
    y_true = y_test
    y_pred = model.predict(t_test)
    
    
    log_run(dataset=dataset,
                     model=model,
                     model_name=f"verhulst_IP_PINN",
                     loss_history=loss_history,
                     collocation_method=collocation_method,
                     learned_params=learned_parameters,
                     plot_solution= plot_inverse_problem_solution,
                     y_true=y_true,
                     y_pred=y_pred)