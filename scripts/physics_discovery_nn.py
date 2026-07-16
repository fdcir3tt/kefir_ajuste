import mlflow
import deepxde as dde
import torch
import torch.nn as nn
import numpy as np
import random

from pathlib import Path
from typing import Any,Callable
from mlflow.data.pandas_dataset import PandasDataset
from kefir_ajuste.utils import ensure_experiment_active,compute_regression_metrics
from kefir_ajuste.data import load_data,load_initial_conditions,load_time_domain,split_train_data


import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class SimpleLossHistory:
    """
    Minimal object mimicking dde.model.LossHistory's interface,
    so it can be passed to dde.utils.plot_loss_history / log_run
    without depending on deepxde's training loop.
    """
    def __init__(self, loss_train: list[float], loss_test: list[float] | None = None):
        n = len(loss_train)
        # dde hace np.sum(loss) por cada entrada -> cada entrada debe ser iterable
        self.loss_train = [[l] for l in loss_train]
        self.loss_test = [[l] for l in (loss_test if loss_test is not None else loss_train)]
        self.steps = list(range(n))
        # Debe tener un elemento por cada step; cada elemento vacío = "sin métricas"
        self.metrics_test = [[] for _ in range(n)]
# ==============================================================================
#                               Global config
# ==============================================================================

mlflow.set_tracking_uri("file:./mlruns")

# Mapeos de nombre -> clase, análogos a lo que hace internamente deepxde
_OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
    "adamw": torch.optim.AdamW,
}

_LOSSES = {
    "mse": nn.MSELoss,
    "mae": nn.L1Loss,
    "huber": nn.SmoothL1Loss,
    "crossentropy": nn.CrossEntropyLoss,
}


_ACTIVATIONS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
}

_INITIALIZERS = {
    "glorot uniform": nn.init.xavier_uniform_,
    "glorot normal": nn.init.xavier_normal_,
    "he uniform": nn.init.kaiming_uniform_,
    "he normal": nn.init.kaiming_normal_,
    "zeros": nn.init.zeros_,
}


EXPERIMENT_NAME = "Neural Network Training"
DATA_FILE_NAME = "control_dataset.csv"
EPOCHS = 10000
LEARNING_RATE = 0.01
SEED = None






# ==============================================================================
#                         Experiments
# ==============================================================================

ensure_experiment_active(EXPERIMENT_NAME)
mlflow.set_experiment(EXPERIMENT_NAME)

def set_seed(seed: int | None = None) -> int:
    """Fixes global seed and returns it """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)         
    torch.cuda.manual_seed_all(seed) 
    dde.config.set_random_seed(seed)

    return seed


def make_net(layers: list[int],
             activation_func: str,
             weights_init: str) -> nn.Module:
    """
    Construct a fully connected neural network (perceptron/MLP) in PyTorch.

    This function creates a feedforward neural network (FNN) using the
    specified architecture, activation function, and weight initialization
    method.

    Parameters
    ----------
    layers : list of int
        List specifying the number of neurons in each layer, including input
        and output layers (e.g., ``[1, 20, 20, 1]``).
    activation_func : str
        Name of the activation function to use between layers
        (e.g., ``"tanh"``, ``"relu"``).
    weights_init : str
        Initialization method for the network weights
        (e.g., ``"Glorot uniform"``, ``"He normal"``).

    Returns
    -------
    torch.nn.Module
        Constructed feedforward neural network.

    Examples
    --------
    >>> model = make_net([1, 20, 20, 1], "tanh", "Glorot uniform")
    """
    activation_cls = _ACTIVATIONS.get(activation_func.lower())
    if activation_cls is None:
        raise ValueError(f"Función de activación no soportada: {activation_func}")

    init_fn = _INITIALIZERS.get(weights_init.lower())
    if init_fn is None:
        raise ValueError(f"Inicialización no soportada: {weights_init}")

    modules = []
    for i in range(len(layers) - 1):
        linear = nn.Linear(layers[i], layers[i + 1])
        init_fn(linear.weight)
        nn.init.zeros_(linear.bias)
        modules.append(linear)

        # No activación después de la última capa (capa de salida)
        if i < len(layers) - 2:
            modules.append(activation_cls())

    net = nn.Sequential(*modules)
    return net


def train_perceptron(optimizer_method: str,
                      loss_func: str,
                      learning_rate: float,
                      epochs: int,
                      X_train: torch.Tensor,
                      y_train: torch.Tensor) -> tuple[list[float], nn.Module]:
    """
    Train a plain PyTorch multilayer perceptron (MLP).

    This function builds the optimizer and loss function from string
    identifiers, then trains the global ``model`` for a given number of
    epochs on the provided training data. Every ``learned_period`` epochs
    (and at the final epoch) it logs the loss and the current parameter
    values to ``learned_variables_path``.

    Parameters
    ----------
    optimizer_method : str
        Optimization algorithm to use (e.g., ``"adam"``, ``"sgd"``).
    loss_func : str
        Loss function used during training (e.g., ``"mse"``, ``"mae"``).
    learning_rate : float
        Learning rate for the optimizer.
    epochs : int
        Number of training epochs.
    learned_period : int
        Frequency (in epochs) at which parameter values are logged.
    learned_variables_path : pathlib.Path
        File path where the logged values will be saved.
    X_train : torch.Tensor
        Input features tensor.
    y_train : torch.Tensor
        Target values tensor.

    Returns
    -------
    loss_history : list[float]
        Loss value recorded at every epoch.
    model : torch.nn.Module
        Trained model instance.

    Notes
    -----
    This function assumes that a global variable ``model`` exists and has
    been initialized prior to calling this function. All parameters of
    ``model`` are optimized.

    Examples
    --------
    >>> history, trained_model = train_perceptron(
    ...     "adam", "mse", 1e-3, 10000,
    ...     100, Path("vars.dat"), X_train, y_train
    ... )
    """
    optimizer_cls = _OPTIMIZERS.get(optimizer_method.lower())
    if optimizer_cls is None:
        raise ValueError(f"Optimizador no soportado: {optimizer_method}")

    loss_cls = _LOSSES.get(loss_func.lower())
    if loss_cls is None:
        raise ValueError(f"Función de pérdida no soportada: {loss_func}")

    optimizer = optimizer_cls(model.parameters(), lr=learning_rate)
    criterion = loss_cls()

    loss_history = []

    
    for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()

            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

            

    return SimpleLossHistory(loss_history), model

def plot_physics_discovery_solution(model: torch.nn.Module,
                                     data_dict: dict[str, Any],
                                     device: str = "cpu") -> list[tuple[Figure, str]]:
    """
    Plot model predictions for all treatment conditions in a single figure.

    Each unique (intensity, exposure time) combination is assigned a distinct
    color. Training points, test points, and network predictions share the
    same color per treatment for easy visual grouping.

    Parameters
    ----------
    model : torch.nn.Module
        Trained PyTorch model with a ``forward`` method.
    data_dict : dict
        Dictionary containing:
        - "t_min" / "t_max"  : time axis bounds
        - "train_data"       : (X_train, y_train) tuple
        - "test_data"        : (X_test,  y_test)  tuple
    device : str, default "cpu"
        Device on which to run inference (e.g., ``"cpu"`` or ``"cuda"``).

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

    model.eval()
    model.to(device)

    for idx, (I_val, T_val) in enumerate(all_conditions):
        color = colors[idx]
        label_base = f"I={I_val:.2f} W/cm², T={T_val:.2f} s"

        # ── Predicción de la red ────────────────────────────────────────────
        grid = np.column_stack([
            t_plot,
            np.full(200, I_val),
            np.full(200, T_val),
        ]).astype(np.float32)

        grid_tensor = torch.from_numpy(grid).to(device)
        with torch.no_grad():
            pred = model(grid_tensor).cpu().numpy()

        ax.plot(t_plot, pred, "--", color=color, linewidth=2,
                label=f"Red Neuronal No informada — {label_base}")

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

    ax.set_title("Predicciones Perceptrón — Todos los tratamientos")
    ax.set_xlabel("Tiempo de Fermentación (h)")
    ax.set_ylabel("Concentración (g/cm³)")
    ax.legend(fontsize=8, ncols=2, loc="best")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()

    return [(fig, "all_treatments")]


def log_run(
            model: nn.Module,
            model_name: str,
            collocation_method: Callable | str | None,
            loss_history,
            learned_params: dict[str, Any],
            log_params: dict[str, Any],
            plot_solution: Callable,
            data_dict: dict[str, Any]) -> None:
    """
    Log training results, metrics, and artifacts to MLflow.

    This function records loss history plots, prediction figures, regression
    metrics, trained models, and learned parameters.

    Parameters
    ----------
    model : torch.nn.Module
        Trained PyTorch model (plain perceptron/MLP).
    model_name : str
        Name used to log the model in MLflow.
    collocation_method : Callable, str, or None
        Collocation method used during training (PINN-specific). Pass
        ``None`` or an empty string for plain data-driven networks.
        Logged as a parameter.
    loss_history : object
        Training loss history. Must expose ``.loss_train``, ``.loss_test``,
        ``.steps``, and ``.metrics_test`` (see ``SimpleLossHistory``).
    learned_params : dict of str to Any
        Dictionary of learned parameters to log.
    log_params : dict of str to Any
        Additional hyperparameters/config to log.
    plot_solution : Callable
        Function that generates plots from the model and dataset.
    data_dict : dict of str to Any
        Must contain ``"y_true"`` and ``"y_pred"`` for metric computation,
        plus whatever ``plot_solution`` needs.

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
    
    # Predicción de entrenamiento
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    with torch.no_grad():
        X_train_tensor = torch.as_tensor(X_train, dtype=torch.float32, device=device)
        y_pred_train = model(X_train_tensor).cpu().numpy()
    model.train(was_training) 

    dde.utils.plot_loss_history(loss_history)
    mlflow.log_figure(plt.gcf(), "loss_plot.png")
    plt.close()

    figures = plot_solution(model=model, data_dict=data_dict)
    for fig, name in figures:
        mlflow.log_figure(fig, f"{name}.png")
        plt.close(fig)

    # ── Conteo de parámetros entrenables ────────────────────────────────
    # dde.Model expone model.net.num_trainable_parameters(); un nn.Module
    # plano no, así que lo calculamos manualmente.
    n_net_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_params = len(learned_params) + n_net_params

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
    # dde.Model requería model.net; un nn.Module plano se loguea directo.
    mlflow.pytorch.log_model(model, name=model_name)

    print("Logging learned parameters...")
    mlflow.log_params(params=log_params)
    mlflow.log_params(params=learned_params)

    if collocation_method:
        name = (collocation_method.__name__
                if callable(collocation_method) else str(collocation_method))
        mlflow.log_param("collocation_method", name)
    else:
        mlflow.log_param("collocation_method", collocation_method)


dataset = load_data(DATA_FILE_NAME)
dataset = dataset.drop(columns=["tratamiento","Unnamed: 0"])
dataset_source_url = f"data/processed/{DATA_FILE_NAME}"
mlflow_dataset: PandasDataset = mlflow.data.from_pandas(dataset, 
                                                        source=dataset_source_url,
                                                        targets="concentracion(g/cm3)",
                                                        name=DATA_FILE_NAME)


run_name = "Perceptron"
with mlflow.start_run(run_name=run_name):

    mlflow.log_input(mlflow_dataset, context="training")
    mlflow.log_param("epochs", EPOCHS)
                                                                    
    seed = set_seed(SEED)                                                                                                                            
    epochs = EPOCHS
    lr = LEARNING_RATE
    
    


# ============================================================
#                         CARGAR DATOS
# ============================================================
    t0,y0 = load_initial_conditions(dataset)
    t0,tf = load_time_domain(dataset)
    
    X_train, y_train, X_test,y_test = split_train_data(dataset,"concentracion(g/cm3)")
    
    X_train_t = torch.tensor(X_train.values if hasattr(X_train, "values") else X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values if hasattr(y_train, "values") else y_train, dtype=torch.float32).reshape(-1, 1)
    X_test_t  = torch.tensor(X_test.values if hasattr(X_test, "values") else X_test, dtype=torch.float32)
    
    variable_path= Path('learned_parameters.dat')

    model = make_net([3, 50, 50, 50, 1], "tanh", "Glorot uniform")

    loss_history, model = train_perceptron(
        optimizer_method="adam",
        loss_func="mse",
        learning_rate=lr,
        epochs=epochs,
        X_train=X_train_t,
        y_train=y_train_t,
    )

    log_params = {
        "seed": seed,
        "learning_rate": lr,
        
    }

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t).cpu().numpy()

    data_dict = {
        "test_data": (X_test, y_test),
        "train_data": (X_train, y_train),
        "y_true": y_test,
        "y_pred": y_pred,
        "t_min": t0,
        "t_max": tf,
    }
    log_run(model=model,
            model_name="not_informed",
            collocation_method = None,
            loss_history = loss_history,
            log_params= log_params,
            learned_params ={},
            plot_solution= plot_physics_discovery_solution,
            data_dict = data_dict)