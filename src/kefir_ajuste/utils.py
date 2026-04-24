import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import mlflow
import deepxde as dde
import mlflow.pytorch

from matplotlib.figure import Figure
from typing import Callable,Any
from mlflow.tracking import MlflowClient

def get_learned_parameters(model:str,n:int|None =None,):
    with open(file=f'learned_parameters.dat',mode='r') as f:
        for line in f:
            pass
    last_line = line.strip()


    epoch_str, values_str = last_line.split(" ", 1)

    params = [float(x) for x in values_str.strip("[]").split(",")]
    if 'verhulst' in model:
        param_dict = {'r':params[0],'k':params[1]}
        
    if 'multi_polynomial' in model:
        param_dict = {
                      'p_coef':torch.tensor(params).reshape(n + 1, n + 1)}   
        
    if 'intensity_function' in model:
        param_dict = {
                      'intensity_coef':torch.tensor(params).reshape(2, 2)} 
    if 'fourier_term' in model:
        param_dict = {
                      'fourier_coef':torch.tensor(params).reshape(-1, 1)}   
    return param_dict

def split_train_data(data:pd.DataFrame)->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    X = data[["intensidad(W/cm^2)","periodo de exposición(s)","tiempo(h)"]].to_numpy()
    y = data["concentracion(g/cm3)"].to_numpy().reshape(-1, 1)

    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    return X_train,y_train,X_test,y_test

def plot_inverse_problem_solution(model, data: pd.DataFrame)->list[tuple[Figure,str]]:
    
    t0 = data["tiempo(h)"].min()
    tf = data["tiempo(h)"].max()
    domain = (t0,tf)
    X_train,y_train,X_test,y_test =split_train_data(data)
    t_train =  X_train[:, 2].reshape(-1, 1)
    t_test =  X_test[:, 2].reshape(-1, 1)
        

    
    T = np.linspace(domain[0], domain[1], 200).reshape(-1, 1)
    pred = model.predict(T)
        
        

    fig, ax = plt.subplots(figsize=(8, 5))
    plt.plot(T, pred, "--", label="Predicción PINN", linewidth=4)
    plt.scatter(t_train, y_train, color="black", label="Datos de entrenamiento")
    plt.scatter(t_test, y_test, color="red", label="Datos test")

    plt.xlabel("Tiempo de Fermentación(h)")
    plt.ylabel("Concentración (g/cm³)")
    plt.legend()
    plt.grid()
    return [(fig,"data_plot")]


def plot_physics_discovery_solution(model, data: pd.DataFrame)->list[tuple[Figure,str]]:
    
    t_min = data["tiempo(h)"].min()
    t_max = data["tiempo(h)"].max()
    
    X_train, y_train, X_test, y_test = split_train_data(data)
        
    t_plot = np.linspace(t_min, t_max, 200)

        # ── One figure per (I, T) treatment ──────────────────────────────────────
    all_conditions = np.unique(np.vstack([X_train[:, :2], X_test[:, :2]]), axis=0)

    figures = []
    for I_val, T_val in all_conditions:

        grid = np.column_stack([np.full(200, I_val),
                                np.full(200, T_val),
                                t_plot ]).astype(np.float32)
        pred = model.predict(grid)

            # Training points that belong to this condition
        train_mask = (X_train[:, 0] == I_val) & (X_train[:, 1] == T_val)
        test_mask  = (X_test[:, 0]  == I_val) & (X_test[:, 1]  == T_val)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(t_plot, pred, "--", linewidth=2, label="Predicción PINN")
        if train_mask.any():
            ax.scatter(X_train[train_mask, 2], y_train[train_mask],
                        color="black", label="Entrenamiento")
        if test_mask.any():
            ax.scatter(X_test[test_mask, 2], y_test[test_mask],
                        color="red", label="Test")

        ax.set_title(f"Tratamiento I={I_val:.2f} W/cm², T={T_val:.2f} °C")
        ax.set_xlabel("Tiempo de Fermentación (h)")
        ax.set_ylabel("Concentración (g/cm³)")
        ax.legend()
        ax.grid()
        fig.tight_layout()
        figures.append((fig, f"treatment_I{I_val:.2f}_T{T_val:.2f}"))

        # ── Test-only figure ─────────────────────────────────────────────────────
    fig_test, ax_test = plt.subplots(figsize=(8, 5))
    for I_val, T_val in all_conditions:
        test_mask = (X_test[:, 0] == I_val) & (X_test[:, 1] == T_val)
        if not test_mask.any():
            continue

        grid = np.column_stack([
                np.full(200, I_val),
                np.full(200, T_val),
                t_plot
            ]).astype(np.float32)
        pred = model.predict(grid)

        ax_test.plot(t_plot, pred, "--", linewidth=2,
                        label=f"PINN (I={I_val:.2f}, T={T_val:.2f})")
        ax_test.scatter(X_test[test_mask, 2], y_test[test_mask],
                            label=f"Test (I={I_val:.2f}, T={T_val:.2f})")

    ax_test.set_title("Datos de Test — Todas las condiciones")
    ax_test.set_xlabel("Tiempo de Fermentación (h)")
    ax_test.set_ylabel("Concentración (g/cm³)")
    ax_test.legend()
    ax_test.grid()
    fig_test.tight_layout()
    figures.append((fig_test, "test_all_conditions"))

    return figures

def get_treatment_name(treatment_index:int)->str:
    return {1:"Testigo (T1) Kéfir sin ultrasonicar",
            2:"15 seg. 20 W/cm2 (T2)",
            3:"60 seg. 20 W/cm2 (T3)",
            4:"15 seg. 34 W/cm2 (T4)",
            5:"60 seg. 34 W/cm2 (T5)"}.get(treatment_index)

def compute_regression_metrics(y_true, y_pred, n_params)->dict[str,float]:
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
    


def log_run(
            dataset:pd.DataFrame,
            model,
            model_name:str,
            collocation_method:Callable|None,
            loss_history,
            learned_params:dict[str,Any],
            plot_solution:Callable,
            y_true:np.ndarray,
            y_pred:np.ndarray)->None:
    """Common logging logic shared by all models."""

    dde.utils.plot_loss_history(loss_history)
    mlflow.log_figure(plt.gcf(), "loss_plot.png")
    plt.close() 

    figures = plot_solution(model=model, data=dataset)
    for fig, name in figures:
        mlflow.log_figure(fig, f"{name}.png")
        plt.close(fig)
    

    
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

