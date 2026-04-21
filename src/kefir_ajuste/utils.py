import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import mlflow
import deepxde as dde
import mlflow.pytorch

from typing import Callable
from pathlib import Path
from mlflow.tracking import MlflowClient
from deepxde.icbc.boundary_conditions import PointSetBC

def get_learned_parameters(model:str,n:int|None =None,m:int|None =None):
    with open(file=f'learned_parameters.dat',mode='r') as f:
        for line in f:
            pass
    last_line = line.strip()


    epoch_str, values_str = last_line.split(" ", 1)

    params = [float(x) for x in values_str.strip("[]").split(",")]
    if model=='verhulst' or model=='verhulst_equal_collocation' or model=='verhulst_all_data' :
        param_dict = {'r':params[0],'k':params[1]}

    if model=='verhulst_polynomial':
        param_dict = {'r':params[0],
                      'k':params[1],
                      'w_coef':params[1:n],'T_coef':params[n:m]}
        
    if 'verhulst_multi_polynomial' in model:
        param_dict = {
                      'p_coef':torch.tensor(params).reshape(n + 1, n + 1)}    
    return param_dict


def load_data(treatment:int)->pd.DataFrame:
    file_path = Path("data") / "processed" / f"tratamiento_{treatment}.csv"
    data = pd.read_csv(file_path)
    return data

def load_initial_conditions(treatment:int)->tuple[float,float,float]:
    data = load_data(treatment)
    t0 = data["tiempo(h)"].iloc[0]
    y0 = data["concentracion(g/cm3)"].iloc[0]

    return t0,y0

def load_time_domain(treatment:int)->tuple[float,float]:
    data = load_data(treatment)
    t0 = data["tiempo(h)"].iloc[0]
    tf = data["tiempo(h)"].iloc[-1]
    return t0,tf

def load_train_data(treatment:int)->tuple[np.ndarray]:
    data = load_data(treatment)
    t = data["tiempo(h)"].to_numpy().reshape(-1, 1)
    y = data["concentracion(g/cm3)"].to_numpy().reshape(-1, 1)

    split = int(0.8 * len(t))
    t_train, y_train = t[:split], y[:split]
    t_test, y_test = t[split:], y[split:]
    return t_train,y_train,t_test,y_test

def plot_solution(model,treatment:int):
    domain=load_time_domain(treatment)
    
    T = np.linspace(domain[0], domain[1], 200).reshape(-1, 1)
    pred = model.predict(T)
    
    t_train,y_train,t_test,y_test =load_train_data(treatment)
    plt.figure(figsize=(8, 5))
    plt.plot(T, pred, "--", label="Predicción PINN", linewidth=4)
    plt.scatter(t_train, y_train, color="black", label="Datos de entrenamiento")
    plt.scatter(t_test, y_test, color="red", label="Datos test")

    plt.xlabel("Tiempo de Fermentación(h)")
    plt.ylabel("Concentración (g/cm³)")
    plt.legend()
    plt.grid()

def all_data_collocation(t,y)->PointSetBC:
    return t,PointSetBC(t,y)

def identity_collocation(t_train:np.ndarray,y_train:np.ndarray)->tuple[np.ndarray,PointSetBC]:
    return t_train,PointSetBC(t_train, y_train)

def random_collocation(t_train:np.ndarray,y_train:np.ndarray,collocation_size:int,seed:int)->PointSetBC:
    if seed:
        np.random.seed(seed)
    idx = np.random.choice(len(t_train), size=collocation_size, replace=False)

    t_sub = t_train[idx]
    y_sub = y_train[idx]

    return t_sub,PointSetBC(t_sub, y_sub)

def equal_collocation(t_train:np.ndarray,y_train:np.ndarray,collocation_skip:int)->PointSetBC:
    idx = np.arange(1, len(t_train), collocation_skip)

    t_sub = t_train[idx]
    y_sub = y_train[idx]

    return t_sub,PointSetBC(t_sub, y_sub)

def get_treatment_name(treatment_index:int)->str:
    return {1:"Testigo (T1) Kéfir sin ultrasonicar",
            2:"15 seg. 20 W/cm2 (T2)",
            3:"60 seg. 20 W/cm2 (T3)",
            4:"15 seg. 34 W/cm2 (T4)",
            5:"60 seg. 34 W/cm2 (T5)"}.get(treatment_index)

def compute_regression_metrics(y_true, y_pred, n_params):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    residuals = y_true - y_pred
    rss = np.sum(residuals ** 2)
    tss = np.sum((y_true - np.mean(y_true)) ** 2)
    n = len(y_true)

    # Métricas de regresión:
    rmse = np.sqrt( (1/n) * rss )
    mae = np.mean(np.abs(residuals))
    mape = np.mean(100 * np.abs(residuals) / np.where(y_true == 0, np.nan, y_true))

    # Métricas de comparación de modelos:
    r2 = 1 - rss / tss if tss != 0 else np.nan
    aic = n * np.log(rss / n) + 2 * n_params
    bic = n * np.log(rss / n) + n_params * np.log(n)

    return {
        "rmse":float(rmse),
        "mae": float(mae),
        "mape": float(mape),
        "r2": float(r2),
        "aic": float(aic),
        "bic": float(bic),
    }

def ensure_experiment_active(experiment_name: str) -> str:
    """
    Ensures an MLflow experiment exists and is active.
    
    If the experiment exists but is deleted, it restores it.
    If it doesn't exist, it creates it.

    Returns:
        experiment_id (str)
    """
    client = MlflowClient()
    experiments = client.search_experiments(view_type=mlflow.entities.ViewType.ALL)

    for exp in experiments:
        if exp.name == experiment_name:
            if exp.lifecycle_stage == "deleted":
                client.restore_experiment(exp.experiment_id)
                print(f"Restored deleted experiment: {experiment_name}")
            else:
                print(f"Experiment already active: {experiment_name}")

            return exp.experiment_id

    exp_id = client.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    print(f"Created new experiment: {experiment_name}")
    return exp_id
    


def log_run(treatment:int,
            model,
                     model_name:str,
                     collocation_method:Callable|None,
                     loss_history,
                     learned_params,
                     y_true,
                     y_pred):
    """Common logging logic shared by all models."""

    dde.utils.plot_loss_history(loss_history)
    mlflow.log_figure(plt.gcf(), "loss_plot.png")
    plt.close() 

    plot_solution(model=model,
                  treatment=treatment)
    mlflow.log_figure(plt.gcf(), "solution_plot.png")
    plt.close() 

    
    n_params = len(learned_params)
    metrics = compute_regression_metrics(
        y_true=y_true,
        y_pred=y_pred,
        n_params=n_params,
    )

    print("Logging metrics...")
    for metric,value in metrics.items():
        mlflow.log_metric(metric, value)


    print(f"Logging Model {model_name}...")
    # Log modelo
    mlflow.pytorch.log_model(model.net, 
                             name=model_name)

    print("Logging learned parameters...")
    # Log parametros aprendidos
    mlflow.log_params(params=learned_params)
    
    if collocation_method:
        mlflow.log_param("collocation_method",collocation_method.__name__)
    else:
        mlflow.log_param("collocation_method",collocation_method)

