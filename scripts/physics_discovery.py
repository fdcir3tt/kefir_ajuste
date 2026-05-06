import os
import mlflow
import deepxde as dde
import torch
import numpy as np

from pathlib import Path
from numpy.typing import NDArray
from typing import Callable,Any
from mlflow.data.pandas_dataset import PandasDataset
from kefir_ajuste.utils import ensure_experiment_active,log_run,plot_physics_discovery_solution,split_train_data,get_learned_parameters
from kefir_ajuste.data import load_data,load_initial_conditions,load_time_domain
from kefir_ajuste.collocation_methods import equal_collocation,identity_collocation
from kefir_ajuste.trainers import multi_polynomial,intensity_function,fourier_term

# ==============================================================================
#                               Global config
# ==============================================================================

mlflow.set_tracking_uri("file:./mlruns")

EXPERIMENT_NAME = "Physics Discovery"
DATA_FILE_NAME = "control_dataset.csv"
EPOCHS = 1000
LEARNING_RATE = 0.01
PHYSICAL_COLLOCATION_POINTS=200
COLLOCATION_METHOD = identity_collocation
COLLOCATION_ARGS = {"collocation_skip":2}
delta =  intensity_function
grade = 6
kwargs = {"grade":grade}

# ==============================================================================
#                         Experiments
# ==============================================================================

ensure_experiment_active(EXPERIMENT_NAME)
mlflow.set_experiment(EXPERIMENT_NAME)

def make_coeficient_list(correction_function:Callable,**kwargs)->list[dde.Variable]:
    if correction_function.__name__=="multi_polynomial":
        grade = kwargs.get("grade")
        c_coef = [dde.Variable(float(torch.rand(1))) for _ in range((grade+1) * (grade+1))]
        for i in range(grade+1):
            for j in range(grade+1):
                if i + j > grade:
                    index = i * (grade + 1) + j  
                    c_coef[index] = dde.Variable(torch.tensor(0.0))  

    if correction_function.__name__=="intensity_function":
        c_coef = [dde.Variable(torch.rand(1)) for _ in range(4)]
    if correction_function.__name__=="fourier_term":
        c_coef = [dde.Variable(torch.rand(1)) for _ in range(2)]
    return c_coef

def make_geometry(X_train: NDArray[np.float32],t0:float,tf:float)->dde.geometry.GeometryXTime:
    t_min, t_max = float(t0), float(tf)
    I_min, I_max = X_train[:, 0].min(), X_train[:, 0].max()
    T_min, T_max = X_train[:, 1].min(), X_train[:, 1].max()
    
    geom_space = dde.geometry.Rectangle(
    [I_min, T_min],
    [I_max, T_max]
)
    timedomain = dde.geometry.TimeDomain(t_min, t_max)
    geom       = dde.geometry.GeometryXTime(geom_space, timedomain)
    return geom

def make_boundary_conditions(collocation_method:Callable,collocation_args:dict[str,int])->tuple[dde.icbc.PointSetBC,torch.Tensor]:
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
    
    net = dde.nn.FNN(layers, activation_func, weights_init)
    model = dde.Model(data_config, net)
    return model

def train_pinn(optimizer_method:str,loss_func:str,learning_rate:float,epochs:int,trainable_variables,learned_period:int,learned_variables_path:Path)->tuple[dde.model.LossHistory,dde.Model]:
    
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

def get_learned_coeficients(correction_function:Callable,learning_rate:float,number_of_pde_collocation_points:int,initial_growth_rate:float,initial_saturation_concentration:float)->dict[str,Any]:
    if correction_function.__name__=="multi_polynomial":
        learned_params = get_learned_parameters(model=correction_function.__name__,
                                                n=grade)
    else:
        learned_params = get_learned_parameters(model=correction_function.__name__)

    learned_params ["learning_rate"] = learning_rate
    learned_params ["initial_rate"]  = initial_growth_rate
    learned_params ["n_phys_points"] = number_of_pde_collocation_points
    learned_params ["initial_saturation_concentration"] = initial_saturation_concentration
    return learned_params 

dataset = load_data(DATA_FILE_NAME)
dataset_source_url = f"data/processed/{DATA_FILE_NAME}"
mlflow_dataset: PandasDataset = mlflow.data.from_pandas(dataset, 
                                                        source=dataset_source_url,
                                                        targets="concentracion(g/cm3)",
                                                        name=DATA_FILE_NAME)


run_name = delta.__name__
with mlflow.start_run(run_name=run_name):

    mlflow.log_input(mlflow_dataset, context="discovery")
    mlflow.log_param("epochs", EPOCHS)
                                                                    
                                                                                                                                      
    epochs = EPOCHS
    lr = LEARNING_RATE
    correction_function = delta
    n_phys_points=PHYSICAL_COLLOCATION_POINTS
    collocation_method= COLLOCATION_METHOD
    collocation_args = COLLOCATION_ARGS


# ============================================================
#                         CARGAR DATOS
# ============================================================

    
    t0,y0 = load_initial_conditions(dataset)
    t0,tf = load_time_domain(dataset)

    X_train, y_train, X_test,y_test = split_train_data(dataset)
# ============================================================
#                 CONFIGURACION ENTRENAMIENTO
# ============================================================
    # Definir coeficientes de corrección 
    
    c_coef = make_coeficient_list(correction_function,**kwargs)
    kappa = 0.046 
    L = 47.81 
    variable_path= Path('learned_parameters.dat')
    trainable_variables = c_coef

    def ode(x:torch.Tensor, y:torch.Tensor)->torch.Tensor:
        I_t = x[:, 0:1]
        T_t = x[:, 1:2]
        t = x[:, 2:3]

        dy_dt = dde.grad.jacobian(y, x, i=0, j=2)
        if correction_function.__name__ == "multi_polynomial":
            delta = correction_function(t,I_t, T_t, c_coef,grade)
        else:
            delta = correction_function(t,I_t, T_t, c_coef)

        return dy_dt - kappa * y * (1 - y / L) - delta

# ============================================================
#                         PINN SETUP
# ============================================================
    
    geom = make_geometry(X_train,t0,tf)

    # Colocación de puntos de entrenamiento 
    observe_bc,anchor_X = make_boundary_conditions(collocation_method,collocation_args)

    data_pinn = dde.data.PDE(geometry=geom,
                             pde=ode,
                             bcs=[observe_bc],
                             num_domain=n_phys_points,       # PDE collocation
                             num_boundary=0,
                             anchors=anchor_X)

    
    model = make_net([3, 50, 50, 50, 1],"tanh", "Glorot uniform",data_pinn)
    
    loss_history, model = train_pinn("adam","MSE",lr,epochs,trainable_variables,600,variable_path)
# ============================================================
#                        ENTRENAMIENTO
# ============================================================
    
    learned_parameters = get_learned_coeficients(correction_function,lr,n_phys_points,kappa,L)

    os.remove(variable_path)
    y_true = y_test
    y_pred = model.predict(X_test)
       

    log_run(dataset=dataset,
            model=model,
            model_name=delta.__name__,
            collocation_method = COLLOCATION_METHOD,
            loss_history=loss_history,
            learned_params=learned_parameters,
            plot_solution=plot_physics_discovery_solution,
            y_true=y_true,
            y_pred=y_pred)