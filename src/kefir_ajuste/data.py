import numpy as np
import pandas as pd

from pathlib import Path
from numpy.typing import NDArray

def load_data(file_name:str)->pd.DataFrame:
    """
    Load a processed dataset from disk.

    Reads a CSV file from the ``data/processed`` directory and
    returns it as a pandas DataFrame.

    Parameters
    ----------
    file_name : str
        Name of the CSV file to load (e.g., ``"dataset.csv"``).

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.

    Notes
    -----
    The file path is constructed as ``data/processed/<file_name>``.
    """
    file_path = Path("data") / "processed" / file_name
    data = pd.read_csv(file_path)
    return data

def load_initial_conditions(data:pd.DataFrame)->tuple[float,float]:
    """
    Extract initial conditions from a dataset.

    Retrieves the first recorded time value and the first concentration
    value from the dataset, assuming they represent the initial state
    of the system.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataset containing at least the columns ``"tiempo(h)"`` and
        ``"concentracion(g/cm3)"``.

    Returns
    -------
    t0 : float
        Initial time value.
    y0 : float
        Initial concentration value.

    Notes
    -----
    Assumes the first row of the dataset corresponds to the initial
    condition.
    """
    t0 = data["tiempo(h)"].iloc[0]
    y0 = data["concentracion(g/cm3)"].iloc[0]

    return t0,y0

def load_time_domain(data:pd.DataFrame)->tuple[float,float]:
    """
    Compute the time domain bounds of a dataset.

    Extracts the minimum and maximum time values from the dataset,
    defining the temporal domain of the experiment.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataset containing a ``"tiempo(h)"`` column.

    Returns
    -------
    t0 : float
        Minimum time value in the dataset.
    tf : float
        Maximum time value in the dataset.

    Notes
    -----
    Does not assume sorted input; uses ``min()`` and ``max()``
    directly on the time column.
    """
    t0 = data["tiempo(h)"].min()
    tf = data["tiempo(h)"].max()
    return t0,tf




def split_train_data(data:pd.DataFrame,target_column:str)->tuple[NDArray[np.float32],NDArray[np.float32],NDArray[np.float32],NDArray[np.float32]]:
    """
    Split a dataset into training and test sets.

    Extracts input features and target values from a DataFrame and
    performs an 80/20 split without shuffling.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset.
    target_column : str
        Name of the column to use as the prediction target. All remaining
        columns are used as input features.

    Returns
    -------
    X_train : ndarray of shape (n_train, n_features)
        Training input features.
    y_train : ndarray of shape (n_train, 1)
        Training target values.
    X_test : ndarray of shape (n_test, n_features)
        Test input features.
    y_test : ndarray of shape (n_test, 1)
        Test target values.

    Notes
    -----
    The split is deterministic and preserves the original row order.
    The cutoff index is computed as ``int(0.8 * len(data))``.
    """
    X = data.drop(columns=[target_column]).to_numpy()
    y = data[target_column].to_numpy().reshape(-1, 1)

    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    return X_train,y_train,X_test,y_test