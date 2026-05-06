import pandas as pd

from pathlib import Path

def load_data(file_name:str)->pd.DataFrame:
    file_path = Path("data") / "processed" / file_name
    data = pd.read_csv(file_path)
    return data

def load_initial_conditions(data:pd.DataFrame)->tuple[float,float]:
    t0 = data["tiempo(h)"].iloc[0]
    y0 = data["concentracion(g/cm3)"].iloc[0]

    return t0,y0

def load_time_domain(data:pd.DataFrame)->tuple[float,float]:
    t0 = data["tiempo(h)"].min()
    tf = data["tiempo(h)"].max()
    return t0,tf




