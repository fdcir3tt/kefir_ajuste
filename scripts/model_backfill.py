import os
import mlflow
import torch
import numpy as np
import matplotlib.pyplot as plt
import deepxde as dde

from kefir_ajuste.utils import compute_regression_metrics,plot_physics_discovery_solution
from kefir_ajuste.data import load_data,load_initial_conditions,load_time_domain,split_train_data
from kefir_ajuste.collocation_methods import identity_collocation
from kefir_ajuste.correction_funcs import intensity_function


run_id    = "01faca481f4e46b9b04723fd8cc2a649"
model_uri = f"runs:/{run_id}/intensity_function"
DATA_FILE_NAME = "control_dataset.csv"
collocation_method = identity_collocation
collocation_args = {}
n_phys_points = 200

def make_boundary_conditions(collocation_method,collocation_args:dict[str,int])->tuple[dde.icbc.PointSetBC,torch.Tensor]:
    """
    Create boundary condition objects from collocation data.

    This function applies a collocation method to training (and optionally test)
    data to generate anchor points and observed values, which are then used to
    construct a ``PointSetBC`` boundary condition object.

    Parameters
    ----------
    collocation_method : Callable
        Function used to generate collocation points and observed values.
        It must accept input data, target values, and a dictionary of arguments,
        and return a tuple ``(anchor_X, observe_y)``.
    collocation_args : dict of str to int
        Dictionary of arguments passed to the collocation method.

    Returns
    -------
    observe_bc : deepxde.icbc.PointSetBC
        Boundary condition object constructed from the collocation points and
        observed values.
    anchor_X : torch.Tensor
        Collocation points used to define the boundary condition, converted
        to ``float32``.

    Notes
    -----
    If the collocation method name is ``"all_data_collocation"``, both training
    and test datasets are combined before applying the method. Otherwise, only
    the training data is used.

    The function assumes the existence of global variables ``X_train``,
    ``y_train``, ``X_test``, and ``y_test``.

    Examples
    --------
    >>> bc, anchors = make_boundary_conditions(method, {"n_points": 100})
    >>> bc
    <dde.icbc.PointSetBC object>
    """
    if collocation_method.__name__ == "all_data_collocation":
        y = np.concatenate([y_train,y_test])
        X = np.vstack((X_train, X_test))
    
        anchor_X, observe_y = collocation_method(X, y, collocation_args)
    else:   
        anchor_X,observe_y = collocation_method(X_train,y_train,collocation_args)
    
    observe_bc = dde.icbc.PointSetBC(anchor_X.astype(np.float32),
                                    observe_y.astype(np.float32),
                                    component=0,
                                    shuffle=False)
    return observe_bc,anchor_X.astype(np.float32)

def make_geometry(X_train,t0:float,tf:float)->dde.geometry.GeometryXTime:
    """ 
    Define problem geometry space.
    
    Parameters
    ----------
    X_train: NDArray
        Numpy array of the training data inputs. Expected shape is (<number_of_points>,3).
    t0: float
        Initial time instant in domain.
    tf: float
        Final point in domain.

    Returns
    -------
    deepxde.geometry.GeometryXTime
        Geometry object that represents the geometry in which the problem takes place.

    Examples
    --------
    >>> 
    
    
    """
    t_min, t_max = float(t0), float(tf)
    I_min, I_max = X_train[:, 1].min(), X_train[:, 1].max()
    T_min, T_max = X_train[:, 2].min(), X_train[:, 2].max()
    geom_space = dde.geometry.Rectangle(
    [I_min, T_min],
    [I_max, T_max]
)
    timedomain = dde.geometry.TimeDomain(t_min, t_max)
    geom       = dde.geometry.GeometryXTime(geom_space, timedomain)
    return geom

def ode(x:torch.Tensor, y:torch.Tensor)->torch.Tensor:
        correction_function = delta
        I_t = x[:, 0:1]
        T_t = x[:, 1:2]
        t = x[:, 2:3]

        dy_dt = dde.grad.jacobian(y, x, i=0, j=2)
        if correction_function.__name__ == "multi_polynomial":
            delta = correction_function(t,I_t, T_t, c_coef,grade)
        else:
            delta = correction_function(t,I_t, T_t, c_coef)

        return model_equation(dy_dt,t,y,model_parameters) - delta*y


def plot_fill(model: dde.Model, data_dict: dict):
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
            np.full(200, I_val),
            np.full(200, T_val),
            t_plot,
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

# -----------  Load Data  -------------
dataset = load_data(DATA_FILE_NAME)
dataset = dataset.drop(columns=["tratamiento","Unnamed: 0"])
dataset_source_url = f"data/processed/{DATA_FILE_NAME}"

t0,y0 = load_initial_conditions(dataset)
t0,tf = load_time_domain(dataset)
    
X_train, y_train, X_test,y_test = split_train_data(dataset,"concentracion(g/cm3)")

X_test_fixed = X_test.copy()
X_test_fixed = X_test_fixed[:, [1, 2, 0]]


#-------- Load DeepXDE model ---------------

geom = make_geometry(X_train,t0,tf)

observe_bc,anchor_X = make_boundary_conditions(collocation_method,collocation_args)

data_pinn = dde.data.PDE(geometry=geom,
                             pde=ode,
                             bcs=[observe_bc],
                             num_domain=n_phys_points,       # PDE collocation
                             num_boundary=0,
                             anchors=anchor_X)
net = dde.nn.FNN([3, 50, 50, 50, 1], "tanh", "Glorot uniform")
saved_net = mlflow.pytorch.load_model(model_uri)

net.load_state_dict(saved_net.state_dict())
net.eval()

model = dde.Model(data_pinn, net)
model.compile("adam", lr=1e-3)


# ----------- Prediction ----------------#
y_pred = model.predict(X_test_fixed)
data_dict = {
        "test_data" : (X_test,y_test),
        "train_data": (X_train,y_train),
        "y_true": y_test,
        "y_pred": y_pred,
        "t_min" :t0,
        "t_max" :tf,
    }
# ---------------  Plot ----------------- #    

figures = plot_fill(model=model, data_dict=data_dict)
for fig, name in figures:
        os.makedirs("figures/best", exist_ok=True)
        path = os.path.join("figures/best", name)
        fig.savefig(path, bbox_inches="tight", dpi=300)
        
        plt.close(fig)

# ---------- Metrics ----------------- #

n_params = 4 + model.net.num_trainable_parameters()
metrics = compute_regression_metrics(
        y_true=y_test,
        y_pred=y_pred,
        n_params=n_params,
    )

print (f"Métricas:\n{metrics}")