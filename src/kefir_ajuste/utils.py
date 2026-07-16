import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import mlflow
import deepxde as dde
import mlflow.pytorch

from numpy.typing import NDArray
from matplotlib.figure import Figure
from typing import Callable,Any
from mlflow.tracking import MlflowClient
from kefir_ajuste.ode_solvers import runge_kutta
from kefir_ajuste.equations import verhulst_eq

def get_learned_parameters(model:str,n:int|None =None,)->dict[str,float]:
    """
    Extract learned parameters from a file and format them by model type.

    This function reads the last line of a parameter log file and parses
    the stored values into a dictionary whose structure depends on the
    specified model name.

    Parameters
    ----------
    model : str
        Name of the model used to determine how parameters are structured.
        Supported patterns include ``"verhulst"``, ``"multi_polynomial"``,
        ``"intensity_function"``, and ``"fourier_term"``.
    n : int, optional
        Polynomial degree used when ``model`` corresponds to a
        ``"multi_polynomial"`` model. Required in that case.

    Returns
    -------
    dict of str to float or torch.Tensor
        Dictionary containing the learned parameters. The structure depends
        on the model type.

    Notes
    -----
    The function assumes that the file ``'learned_parameters.dat'`` exists
    and that its last line contains parameter values in list format.

    Examples
    --------
    >>> params = get_learned_parameters("verhulst")
    >>> params["r"]
    0.12
    """
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



def plot_inverse_problem_solution(model:dde.Model, data_dict: dict[str,Any])->list[tuple[Figure,str]]:
    """
    Plot model predictions for the inverse problem.

    This function generates a single plot comparing model predictions with
    training and test data over time.

    Parameters
    ----------
    model : deepxde.Model
        Trained model with a ``predict`` method.
    data : pandas.DataFrame
        Dataset containing time and concentration values.

    Returns
    -------
    list of tuple of (matplotlib.figure.Figure, str)
        List containing a single figure and its associated name.

    Notes
    -----
    The prediction is performed over a uniform grid of 200 time points.
    """
    equations_dict = {"verhulst":verhulst_eq}

    t_min = data_dict.get("t_min")
    t_max = data_dict.get("t_max")

    X_train, y_train = data_dict.get("train_data")
    X_test,  y_test  = data_dict.get("test_data")

    equation_params = data_dict.get("equation_parameters")
    model_equation  = data_dict.get("model_equation")
    model_equation  = equations_dict.get(model_equation)

    t_train =  X_train.reshape(-1, 1)
    t_test  =  X_test.reshape(-1, 1)
    y_0     =  y_train[0][0]
    
    for key,param in equation_params.items():
        equation_params[key] = float(param)



    numeric_solution = runge_kutta(f=model_equation,
                                   parameters=equation_params,
                                   y0=y_0,
                                   interval=(t_min,t_max),
                                   n=50)
    
    
    t_rk, y_rk = zip(*numeric_solution) 

    T = np.linspace(t_min, t_max, 200).reshape(-1, 1)
    pred = model.predict(T)
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.plot(T, pred, "--", label="Predicción PINN", linewidth=4)
    ax.plot(t_rk, y_rk, "-",  color="tab:orange",  linewidth=2, label="Solución Numérica (RK)")
    plt.scatter(t_train, y_train, color="black", label="Datos de entrenamiento")
    plt.scatter(t_test, y_test, color="red", label="Datos test")

    plt.xlabel("Tiempo de Fermentación(h)")
    plt.ylabel("Concentración (g/cm³)")
    plt.legend()
    plt.grid()
    return [(fig,"data_plot")]

def plot_physics_discovery_solution(model: dde.Model, data_dict: dict[str, Any]) -> list[tuple[Figure, str]]:
    """
    Plot model predictions for all treatment conditions in a single figure.

    Each unique (intensity, exposure time) combination is assigned a distinct
    color. Training points, test points, and PINN predictions share the same
    color per treatment for easy visual grouping.

    Parameters
    ----------
    model : object
        Trained model with a ``predict`` method.
    data_dict : dict
        Dictionary containing:
        - "t_min" / "t_max"  : time axis bounds
        - "train_data"       : (X_train, y_train) tuple
        - "test_data"        : (X_test,  y_test)  tuple

    Returns
    -------
    list of tuple of (matplotlib.figure.Figure, str)
        Single-element list with the combined figure and its name.
    """
    t_min = data_dict.get("t_min")
    t_max = data_dict.get("t_max")

    X_train, y_train = data_dict.get("train_data")
    X_test,  y_test  = data_dict.get("test_data")

    
    
    t_plot = np.linspace(t_min, t_max, 200)
    all_conditions = np.unique(
        np.vstack([X_train[:, 1:3], X_test[:, 1:3]]), axis=0
    )
    print(f"Treatments: {all_conditions}")

    # One color per treatment
    cmap   = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(all_conditions))]

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, (I_val, T_val) in enumerate(all_conditions):
        color = colors[idx]
        label_base = f"I={I_val:.2f} W/cm², T={T_val:.2f} s"

        # ── PINN prediction curve ────────────────────────────────────────────
        grid = np.column_stack([
            t_plot,
            np.full(200, I_val),
            np.full(200, T_val),
        ]).astype(np.float32)
        pred = model.predict(grid)

        ax.plot(t_plot, pred, "--", color=color, linewidth=2,
                label=f"PINN — {label_base}")

        # ── Training scatter ─────────────────────────────────────────────────
        train_mask = (X_train[:, 1] == I_val) & (X_train[:, 2] == T_val)
        if train_mask.any():
            ax.scatter(X_train[train_mask, 0], y_train[train_mask],
                       color=color, marker="o", edgecolors="black",
                       linewidths=0.6, s=50,
                       label=f"Entrenamiento — {label_base}")

        # ── Test scatter ─────────────────────────────────────────────────────
        test_mask = (X_test[:, 1] == I_val) & (X_test[:, 2] == T_val)
        if test_mask.any():
            ax.scatter(X_test[test_mask, 0], y_test[test_mask],
                       color=color, marker="^", edgecolors="black",
                       linewidths=0.6, s=60,
                       label=f"Test — {label_base}")

    ax.set_title("Predicciones PINN — Todos los tratamientos")
    ax.set_xlabel("Tiempo de Fermentación (h)")
    ax.set_ylabel("Concentración (g/cm³)")
    ax.legend(fontsize=8, ncols=2, loc="best")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()

    return [(fig, "all_treatments")]

def get_treatment_name(treatment_index:int)->str:
    """
    Map a treatment index to its descriptive name.

    Parameters
    ----------
    treatment_index : int
        Identifier of the treatment.

    Returns
    -------
    str
        Human-readable name of the treatment. Returns ``None`` if the index
        is not recognized.
    """
    return {1:"Testigo (T1) Kéfir sin ultrasonicar",
            2:"15 seg. 20 W/cm2 (T2)",
            3:"60 seg. 20 W/cm2 (T3)",
            4:"15 seg. 34 W/cm2 (T4)",
            5:"60 seg. 34 W/cm2 (T5)"}.get(treatment_index)

def compute_regression_metrics(y_true, y_pred, n_params)->dict[str,float]:
    """
    Compute regression and model selection metrics.

    Parameters
    ----------
    y_true : array_like
        Ground truth target values.
    y_pred : array_like
        Predicted values from the model.
    n_params : int
        Number of parameters in the model.

    Returns
    -------
    dict of str to float
        Dictionary containing RMSE, MAE, MAPE, R², AIC, and BIC metrics.

    Notes
    -----
    MAPE ignores zero values in ``y_true`` to avoid division errors.
    """
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
    Ensure that an MLflow experiment exists and is active.

    If the experiment exists but is marked as deleted, it is restored.
    If it does not exist, it is created.

    Parameters
    ----------
    experiment_name : str
        Name of the MLflow experiment.

    Returns
    -------
    str
        Experiment ID.

    Notes
    -----
    This function interacts with the MLflow tracking server and may modify
    experiment state.
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
            model:dde.Model,
            model_name:str,
            collocation_method:Callable|None,
            loss_history,
            learned_params:dict[str,Any],
            log_params:dict[str,Any],
            plot_solution:Callable,
            data_dict:dict[str,Any])->None:
    """
    Log training results, metrics, and artifacts to MLflow.

    This function records loss history plots, prediction figures, regression
    metrics, trained models, and learned parameters.

    Parameters
    ----------
    dataset : pandas.DataFrame
        Dataset used for training and evaluation.
    model : deepxde.Model
        Trained model containing a ``net`` attribute.
    model_name : str
        Name used to log the model in MLflow.
    collocation_method : Callable or None
        Collocation method used during training. Logged as a parameter.
    loss_history : object
        Training loss history returned by the model.
    learned_params : dict of str to Any
        Dictionary of learned parameters to log.
    plot_solution : Callable
        Function that generates plots from the model and dataset.
    y_true : ndarray
        Ground truth values.
    y_pred : ndarray
        Model predictions.

    Returns
    -------
    None

    Notes
    -----
    This function assumes an active MLflow run.
    """
    y_true = data_dict.get("y_true")
    y_pred = data_dict.get("y_pred")
    X_train,y_train = data_dict.get("train_data")
    y_pred_train = model.predict(X_train)

    dde.utils.plot_loss_history(loss_history)
    mlflow.log_figure(plt.gcf(), "loss_plot.png")
    plt.close() 

    figures = plot_solution(model=model, data_dict=data_dict)
    for fig, name in figures:
        mlflow.log_figure(fig, f"{name}.png")
        plt.close(fig)
    

    
    n_params = len(learned_params) + model.net.num_trainable_parameters()
    test_metrics = compute_regression_metrics(
        y_true=y_true,
        y_pred=y_pred,
        n_params=n_params,
    )


    train_metrics = compute_regression_metrics(
        y_true=y_train,
        y_pred=y_pred_train,
        n_params=n_params,
    )

    print("Logging metrics...")
    for metric,value in test_metrics.items():
        mlflow.log_metric(f"test_{metric}", value)

    for metric,value in train_metrics.items():
        mlflow.log_metric(f"train_{metric}", value)

    print(f"Logging Model {model_name}...")
    # Log modelo
    mlflow.pytorch.log_model(model.net, 
                             name=model_name)

    print("Logging learned parameters...")
    # Log parametros aprendidos
    mlflow.log_params(params=log_params)
    mlflow.log_params(params=learned_params)
    
    if collocation_method:
        mlflow.log_param("collocation_method",collocation_method.__name__)
    else:
        mlflow.log_param("collocation_method",collocation_method)

