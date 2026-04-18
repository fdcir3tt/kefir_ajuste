import itertools
import mlflow
import mlflow.pytorch
import numpy as np
import deepxde as dde
import matplotlib.pyplot as plt

from mlflow.tracking import MlflowClient
from kefir_ajuste.utils import plot_solution
from kefir_ajuste.trainers import train_verhulst, train_polynomial
# ==============================================================================
#                               Global config
# ==============================================================================

mlflow.set_tracking_uri("file:./mlruns")

treatments = range(1,6)
grades = range(5,7)
n_iterations = [15000, 20000, 30000,50000]


treatments = [1]
grades = []
#n_iterations = [2000,3000]

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

    final_train_loss = np.array(loss_history.loss_train[-1])
    final_test_loss = np.array(loss_history.loss_test[-1])

    mlflow.log_metric(f"final_train_loss", float(final_train_loss[-1]))
    mlflow.log_metric(f"final_test_loss", float(final_test_loss[-1]))
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
    
    


# ==============================================================================
#                           Verhulst Experiments
# ==============================================================================

ensure_experiment_active("verhulst")
mlflow.set_experiment("verhulst")
for treatment in treatments:

    with mlflow.start_run(run_name=f"T{treatment}"):
        mlflow.log_param("family", "verhulst")
        for epochs in n_iterations:
            with mlflow.start_run(
                run_name=f"T{treatment}_E{epochs}",
                nested=True,
            ):

                mlflow.log_param("treatment", treatment)
                mlflow.log_param("epochs", epochs)

                model, loss_history,learned_parameters, y_true, y_pred = train_verhulst(
                                                                                        treatment=treatment,
                                                                                        epochs=epochs,
                                                                                    )

                log_training_run(treatment=treatment,
                                 model=model,
                                 model_name="verhulst",
                                 loss_history=loss_history,
                                 learned_params=learned_parameters,
                                 y_true=y_true,
                                 y_pred=y_pred)


# ==============================================================================
#                         Polynomial Experiments
# ==============================================================================

ensure_experiment_active("verhulst_polynomial")
mlflow.set_experiment("verhulst_polynomial")

for treatment in treatments:

    if treatment <= 1:
        continue

    with mlflow.start_run(run_name=f"T{treatment}"):
        mlflow.log_param("family", "verhulst_polynomial")
        for grade, epochs in itertools.product(grades, n_iterations):

            with mlflow.start_run(
                run_name=f"T{treatment}_G{grade}_E{epochs}",
                nested=True,
            ):

                mlflow.log_param("treatment", treatment)
                mlflow.log_param("grade", grade)
                mlflow.log_param("epochs", epochs)

                model, loss_history,learned_parameters, y_true, y_pred = train_polynomial(
                    treatment=treatment,
                    grade=grade,
                    epochs=epochs,
                )

                log_training_run(treatment=treatment,
                                 model=model,
                                 model_name=f"verhulst_polynomial_order_{grade}",
                                 loss_history=loss_history,
                                 learned_params=learned_parameters,
                                 y_true=y_true,
                                 y_pred=y_pred)