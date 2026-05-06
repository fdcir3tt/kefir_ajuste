import pandas as pd

from pathlib import Path

def load_data(file_name:str)->pd.DataFrame:
    """
    Load a processed dataset from disk.

    This function reads a CSV file from the ``data/processed`` directory and
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
    The file path is constructed as:
    ``data/processed/<file_name>``.
    """
    file_path = Path("data") / "processed" / file_name
    data = pd.read_csv(file_path)
    return data

def load_initial_conditions(data:pd.DataFrame)->tuple[float,float]:
    """
    Extract initial conditions from a dataset.

    This function retrieves the first recorded time value and the first
    concentration value from the dataset, assuming they represent the
    initial state of the system.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataset containing at least the columns:
        ``"tiempo(h)"`` and ``"concentracion(g/cm3)"``.

    Returns
    -------
    t0 : float
        Initial time value.
    y0 : float
        Initial concentration value.

    Notes
    -----
    The function assumes that the first row of the dataset corresponds
    to the initial condition.
    """
    t0 = data["tiempo(h)"].iloc[0]
    y0 = data["concentracion(g/cm3)"].iloc[0]

    return t0,y0

def load_time_domain(data:pd.DataFrame)->tuple[float,float]:
    """
    Compute the time domain of a dataset.

    This function extracts the minimum and maximum time values from the
    dataset, defining the temporal domain of the experiment.

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
    This function does not assume sorted input data.
    """
    t0 = data["tiempo(h)"].min()
    tf = data["tiempo(h)"].max()
    return t0,tf




