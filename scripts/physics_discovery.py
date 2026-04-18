import mlflow

from mlflow.data.pandas_dataset import PandasDataset
from kefir_ajuste.utils import ensure_experiment_active,log_run,load_data,equal_collocation,get_treatment_name
from kefir_ajuste.trainers import multi_polynomial_model

# ==============================================================================
#                               Global config
# ==============================================================================

mlflow.set_tracking_uri("file:./mlruns")

EXPERIMENT_NAME = "Physics Discovery"
TREATMENT = 2
GRADE = 1
EPOCHS = 1000
LEARNING_RATE = 0.01
COLLOCATION_METHOD = equal_collocation 
training_method = multi_polynomial_model

# ==============================================================================
#                         Polynomial Experiments
# ==============================================================================

ensure_experiment_active(EXPERIMENT_NAME)
mlflow.set_experiment(EXPERIMENT_NAME)


dataset = load_data(TREATMENT)
dataset_source_url = f"data/processed/tratamiento_{TREATMENT}.csv"
mlflow_dataset: PandasDataset = mlflow.data.from_pandas(dataset, source=dataset_source_url,targets="concentracion(g/cm3)",name=f"tratamiento_{TREATMENT}.csv")


run_name = training_method.__name__
with mlflow.start_run(run_name=run_name):
    mlflow.log_input(mlflow_dataset, context="discovery")
    mlflow.log_param("treatment", get_treatment_name(TREATMENT))
    mlflow.log_param("epochs", EPOCHS)

    model, loss_history,learned_parameters, y_true, y_pred = training_method(treatment=TREATMENT,
                                                                             grade=GRADE,
                                                                             epochs=EPOCHS,
                                                                             lr=LEARNING_RATE,
                                                                             collocation_method=COLLOCATION_METHOD,
                                                                             collocation_skip=2)

    log_run(treatment=TREATMENT,
                     model=model,
                     model_name=f"multi_polynomial",
                     collocation_method = COLLOCATION_METHOD,
                     loss_history=loss_history,
                     learned_params=learned_parameters,
                     y_true=y_true,
                     y_pred=y_pred)