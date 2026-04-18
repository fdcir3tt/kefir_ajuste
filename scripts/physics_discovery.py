import itertools
import mlflow
import mlflow.pytorch
import numpy as np
import deepxde as dde
import matplotlib.pyplot as plt

from typing import Callable
from mlflow.data.pandas_dataset import PandasDataset
from mlflow.tracking import MlflowClient
from kefir_ajuste.utils import plot_solution,load_data,equal_collocation
from kefir_ajuste.trainers import train_multi_polynomial
# ==============================================================================
#                               Global config
# ==============================================================================

mlflow.set_tracking_uri("file:./mlruns")

experiment_name = "Physics Discovery"
treatments = range(2,6)
grades =[1]
n_iterations = [1000]
learning_rate = 0.01
seed = 42
collocation_method = equal_collocation
collocation_args = {"collocation_skip":4}

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
    
def log_training_run(treatment,
                     model,
                     model_name:str,
                     collocation_method:Callable,
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
    
    mlflow.log_param("collocation_method",collocation_method.__name__)
    
    

# ==============================================================================
#                         Polynomial Experiments
# ==============================================================================

ensure_experiment_active(experiment_name)
mlflow.set_experiment(experiment_name)
for treatment in treatments:
    if treatment <= 1:
        continue

    dataset = load_data(treatment)
    dataset_source_url = f"data/processed/tratamiento_{treatment}.csv"
    mlflow_dataset: PandasDataset = mlflow.data.from_pandas(dataset, source=dataset_source_url,targets="concentracion(g/cm3)",name=f"tratamiento_{treatment}.csv")


    run_name = f"T{treatment}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_input(mlflow_dataset, context="discovery")
        for grade, epochs in itertools.product(grades, n_iterations):

                mlflow.log_param("treatment", treatment)
                mlflow.log_param("grade", grade)
                mlflow.log_param("epochs", epochs)

                model, loss_history,learned_parameters, y_true, y_pred = train_multi_polynomial(
                                                                                                treatment=treatment,
                                                                                                grade=grade,
                                                                                                epochs=epochs,
                                                                                                lr=learning_rate,
                                                                                                collocation_method=collocation_method,
                                                                                                collocation_skip=2

                                                                                            )

                log_training_run(treatment=treatment,
                                 model=model,
                                 model_name=f"multi_polynomial_order_{grade}",
                                 collocation_method = collocation_method,
                                 loss_history=loss_history,
                                 learned_params=learned_parameters,
                                 y_true=y_true,
                                 y_pred=y_pred)