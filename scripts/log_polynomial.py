import os 
import pandas as pd
import deepxde as dde
import numpy as np
import random
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description="Train a PINN model")

parser.add_argument("--treatment", type=str, default=2,
                    help="Treatment")
parser.add_argument("--lr", type=float, default=0.001,
                    help="Learning rate")
parser.add_argument("--epochs", type=int, default=10000,
                    help="Number of training epochs")
parser.add_argument("--grade", type=str, default=3,
                    help="Directory to save the model")


args = parser.parse_args()
# ============================================================================== #
#                               CONFIGURACIÓN
# ============================================================================== #

os.environ["DDE_BACKEND"] = "pytorch"
dde.backend.set_default_backend("pytorch")
TRATAMIENTO = args.treatment
POLYNOMIAL_GRADE = args.grade
FILE_PATH = Path('data') / 'processed' / f'tratamiento_{TRATAMIENTO}.csv'
MODEL_PATH = Path('models')/'verhulst_polynomial'/str(POLYNOMIAL_GRADE)
FIGURES_PATH = Path('figures')/'verhulst'/'polynomial'/str(POLYNOMIAL_GRADE)
ITERATIONS = args.epochs
LEARNING_RATE =args.lr

intensity_dict= { '2':{'frequency':20.0,'period':15.0},
                  '3':{'frequency':20.0,'period':60.0},
                  '4':{'frequency':34.0,'period':15.0},
                  '5':{'frequency':34.0,'period':60.0}
                }

w = intensity_dict[TRATAMIENTO]['frequency']
T_period = intensity_dict[TRATAMIENTO]['period']

w_coef ,T_coef = [],[]
for i in range(int(POLYNOMIAL_GRADE)):
    coefficient = dde.Variable(random.random())
    w_coef.append(coefficient)
    T_coef.append(coefficient)

r = dde.Variable(0.04)
k = dde.Variable(51.0)

def polynomial(x:float,coef:list[float])->float:
    result , n = 0 , 0
    for c in coef:
        result += c * (x ** n)
    return result

def ode(t, y):
    dy_dt = dde.grad.jacobian(y, t, i=0, j=0)
    func = dy_dt - r * y * (1 - y / k) - polynomial(x=w,coef=w_coef) - polynomial(x=T_period,coef=T_coef)
    return func


def exact_solution(t, y0, r_val, k_val,):
    return (y0 * k_val) / (y0 + (k_val - y0) * np.exp(-r_val * t))



# ============================================================================== #
#                            PREPARACIÓN DE DATOS
# ============================================================================== #

print("Cargando datos...")
data = pd.read_csv(FILE_PATH)

# ---------- Condición inicial ---------- #

t0 = data["tiempo(h)"].iloc[0]
tf = data["tiempo(h)"].iloc[-1]
y0 = data["concentracion(g/cm3)"].iloc[0]


t = data["tiempo(h)"].to_numpy().reshape(-1, 1)
y = data["concentracion(g/cm3)"].to_numpy().reshape(-1, 1)


# ------------- 80/20 split ------------- #

split = int(0.8 * len(t))
t_train, y_train = t[:split], y[:split]
t_test, y_test = t[split:], y[split:]




# ============================================================================== #
#                                  RED NEURONAL
# ============================================================================== #
os.makedirs(MODEL_PATH,exist_ok=True)

geom = dde.geometry.TimeDomain(t0, tf)
ic = dde.icbc.IC(
    geom,
    lambda t: y0,
    lambda _, on_initial: on_initial,
)
observe_y = dde.icbc.PointSetBC(t_train, y_train)

data_pinn = dde.data.PDE(
    geometry=geom,
    pde=ode,
    bcs=[ic,observe_y],
    num_domain=200,
    num_boundary=2,
    num_test=100,
    anchors=t_train,
)


neurons = [50, 50, 50]
layer_size = [1] + neurons + [1]

net = dde.nn.FNN(layer_size, 
                 "tanh", 
                 "Glorot uniform")

model = dde.Model(data_pinn, net)

model.compile(
    optimizer="adam",
    lr=LEARNING_RATE,
    external_trainable_variables=[r, k]+ w_coef+ T_coef
)


early_stop = dde.callbacks.EarlyStopping(monitor="loss_train",
                                         baseline=1e-4,
                                         start_from_epoch=20000)
variable = dde.callbacks.VariableValue(
                                        var_list=[r,k]+w_coef+T_coef, 
                                        period=600, 
                                        filename=MODEL_PATH/f'parameters_{TRATAMIENTO}.dat'
                                    )
callbacks = [
    early_stop,
    variable
]





# ============================================================================== #
#                               ENTRENAMIENTO
# ============================================================================== #


print("Entrenando PINN...")

losshistory, train_state = model.train(
                                        iterations=ITERATIONS,
                                        callbacks=callbacks,
                                        model_save_path=MODEL_PATH/f"T{TRATAMIENTO}"
                                    )

dde.saveplot(losshistory, train_state, isplot=False)

with open(file=MODEL_PATH/f'parameters_{TRATAMIENTO}.dat',mode='r') as f:
    for line in f:
        pass
last_line = line.strip()


epoch_str, values_str = last_line.split(" ", 1)
epoch = int(epoch_str)

params = [float(x) for x in values_str.strip("[]").split(",")]

print("\nRESULTADOS:")
print("r learned =", params[0])
print("k learned =", params[1])

# ============================================================================== #
#                               GRÁFICAS
# ============================================================================== #


print("\nGraficando...")
os.makedirs(FIGURES_PATH,exist_ok=True)
dde.utils.plot_loss_history(losshistory)

plt.savefig(FIGURES_PATH / f"loss_history_T{TRATAMIENTO}_{ITERATIONS}.png", dpi=300, bbox_inches="tight")
plt.close()  

T = np.linspace(t0, tf, 200).reshape(-1, 1)
pred = model.predict(T)
real = exact_solution(T, y0, params[0], params[1])

plt.figure(figsize=(8, 5))
plt.plot(T, real, label=f"Solución exacta (con parámetros aprendidos) T{TRATAMIENTO}", linewidth=4)
plt.plot(T, pred, "--", label="Predicción PINN", linewidth=4)
plt.scatter(t_train, y_train, color="black", label="Datos de entrenamiento")
plt.scatter(t_test, y_test, color="red", label="Datos test")

plt.xlabel("Tiempo de Fermentación(h)")
plt.ylabel("Concentración (g/cm³)")
plt.legend()
plt.grid()
plt.savefig(FIGURES_PATH/f"verhulst_T{TRATAMIENTO}_{ITERATIONS}.pdf")
