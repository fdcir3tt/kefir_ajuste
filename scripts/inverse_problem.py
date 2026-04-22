import mlflow

from mlflow.data.pandas_dataset import PandasDataset
from kefir_ajuste.utils import log_run,ensure_experiment_active,identity_collocation,load_data
from kefir_ajuste.trainers import verhulst

# ==============================================================================
#                               Global config
# ==============================================================================

mlflow.set_tracking_uri("file:./mlruns")
DATA_FILE_NAME = "tratamiento_1.csv"
EXPERIMENT_NAME = "Inverse Problem"
EPOCHS = 10000
LEARNING_RATE = 0.01
model_equation = verhulst
collocation_method = identity_collocation




# ==============================================================================
#                         Polynomial Experiments
# ==============================================================================

ensure_experiment_active(EXPERIMENT_NAME)
mlflow.set_experiment(EXPERIMENT_NAME)

dataset = load_data(DATA_FILE_NAME)
dataset_source_url = f"data/processed/{DATA_FILE_NAME}"
mlflow_dataset: PandasDataset = mlflow.data.from_pandas(dataset, source=dataset_source_url,targets="concentracion(g/cm3)",name=DATA_FILE_NAME)


run_name = f"{model_equation.__name__}_{LEARNING_RATE}_{EPOCHS}"
with mlflow.start_run(run_name=run_name):
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("learning_rate", LEARNING_RATE)

    model, loss_history,learned_parameters, y_true, y_pred = model_equation(dataset=dataset,
                                                                            epochs=EPOCHS,
                                                                            lr=LEARNING_RATE,
                                                                            collocation_method=collocation_method
                                                                             )

    log_run(dataset=dataset,
                     model=model,
                     model_name=f"verhulst_IP_PINN",
                     loss_history=loss_history,
                     collocation_method=collocation_method,
                     learned_params=learned_parameters,
                     y_true=y_true,
                     y_pred=y_pred)