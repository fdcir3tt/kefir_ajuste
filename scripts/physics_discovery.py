import mlflow

from mlflow.data.pandas_dataset import PandasDataset
from kefir_ajuste.utils import ensure_experiment_active,log_run,\
                               load_data,equal_collocation,identity_collocation
from kefir_ajuste.trainers import physics_discovery,multi_polynomial

# ==============================================================================
#                               Global config
# ==============================================================================

mlflow.set_tracking_uri("file:./mlruns")

EXPERIMENT_NAME = "Physics Discovery"
DATA_FILE_NAME = "control_dataset.csv"
GRADE = 1
EPOCHS = 1000
LEARNING_RATE = 0.01
COLLOCATION_METHOD = identity_collocation
delta = multi_polynomial

# ==============================================================================
#                         Polynomial Experiments
# ==============================================================================

ensure_experiment_active(EXPERIMENT_NAME)
mlflow.set_experiment(EXPERIMENT_NAME)


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

    model, loss_history,learned_parameters, y_true, y_pred = physics_discovery( dataset=dataset,
                                                                                correction_function=delta,
                                                                                grade=GRADE,
                                                                                epochs=EPOCHS,
                                                                                lr=LEARNING_RATE,
                                                                                collocation_method=COLLOCATION_METHOD,
                                                                                )

    log_run(experiment="physics_discovery",
            dataset=dataset,
            model=model,
            model_name=f"multi_polynomial",
            collocation_method = COLLOCATION_METHOD,
            loss_history=loss_history,
            learned_params=learned_parameters,
            y_true=y_true,
            y_pred=y_pred)