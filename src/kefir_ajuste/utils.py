import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def get_learned_parameters(model:str,treatment:int,n:int|None =None,m:int|None =None):
    with open(file=f'{model}_{str(treatment)}.dat',mode='r') as f:
        for line in f:
            pass
    last_line = line.strip()


    epoch_str, values_str = last_line.split(" ", 1)
    

    params = [float(x) for x in values_str.strip("[]").split(",")]
    if model=='verhulst':
        param_dict = {'r':params[0],'k':params[1]}
    if model=='verhulst_polynomial':
        param_dict = {'r':params[0],
                      'k':params[1],
                      'w_coef':params[1:n],'T_coef':params[n:m]}
    if model=='verhulst_multi_polynomial':
        param_dict = {'r':params[0],
                      'k':params[1],
                      'p_coef':params[1:]}
    return param_dict


def load_data(treatment:int)->pd.DataFrame:
    file_path = Path("data") / "processed" / f"tratamiento_{treatment}.csv"
    data = pd.read_csv(file_path)
    return data

def load_initial_conditions(treatment:int)->tuple[float,float,float]:
    data = load_data(treatment)
    t0 = data["tiempo(h)"].iloc[0]
    y0 = data["concentracion(g/cm3)"].iloc[0]

    return t0,y0

def load_time_domain(treatment:int)->tuple[float,float]:
    data = load_data(treatment)
    t0 = data["tiempo(h)"].iloc[0]
    tf = data["tiempo(h)"].iloc[-1]
    return t0,tf

def load_train_data(treatment:int)->tuple:
    data = load_data(treatment)
    t = data["tiempo(h)"].to_numpy().reshape(-1, 1)
    y = data["concentracion(g/cm3)"].to_numpy().reshape(-1, 1)

    split = int(0.8 * len(t))
    t_train, y_train = t[:split], y[:split]
    t_test, y_test = t[split:], y[split:]
    return t_train,y_train,t_test,y_test

def plot_solution(model,treatment):
    domain=load_time_domain(treatment)
    
    T = np.linspace(domain[0], domain[1], 200).reshape(-1, 1)
    pred = model.predict(T)
    
    t_train,y_train,t_test,y_test =load_train_data(treatment)
    plt.figure(figsize=(8, 5))
    plt.plot(T, pred, "--", label="Predicción PINN", linewidth=4)
    plt.scatter(t_train, y_train, color="black", label="Datos de entrenamiento")
    plt.scatter(t_test, y_test, color="red", label="Datos test")

    plt.xlabel("Tiempo de Fermentación(h)")
    plt.ylabel("Concentración (g/cm³)")
    plt.legend()
    plt.grid()