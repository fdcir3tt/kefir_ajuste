import mlflow

from kefir_ajuste.utils import log_run,ensure_experiment_active,identity_collocation
from kefir_ajuste.trainers import verhulst

# ==============================================================================
#                               Global config
# ==============================================================================

mlflow.set_tracking_uri("file:./mlruns")

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




run_name = f"{model_equation.__name__}_{LEARNING_RATE}_{EPOCHS}"
with mlflow.start_run(run_name=run_name):
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("learning_rate", LEARNING_RATE)

    model, loss_history,learned_parameters, y_true, y_pred = model_equation(treatment=1,
                                                                      epochs=EPOCHS,
                                                                      lr=LEARNING_RATE,
                                                                      collocation_method=collocation_method
                                                                      )

    log_run(treatment=1,
                     model=model,
                     model_name=f"verhulst_IP_PINN",
                     loss_history=loss_history,
                     collocation_method=collocation_method,
                     learned_params=learned_parameters,
                     y_true=y_true,
                     y_pred=y_pred)