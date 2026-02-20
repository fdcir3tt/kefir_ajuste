import os 
import pandas as pd
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================== #
#                               CONFIGURACIÓN
# ============================================================================== #

os.environ["DDE_BACKEND"] = "pytorch"
dde.backend.set_default_backend("pytorch")
FILE_PATH = Path('data') / 'processed' / 'tratamiento_1.csv'
ITERATIONS = 20000

r = dde.Variable(0.04)
k = dde.Variable(51.0)

def ode(t, y):
    dy_dt = dde.grad.jacobian(y, t, i=0, j=0)
    return dy_dt - r * y * (1 - y / k)


def exact_solution(t, y0, r_val, k_val):
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

geom = dde.geometry.TimeDomain(t0, tf)
observe_y_bc = dde.icbc.PointSetBC(t_train, y_train)

data_pinn = dde.data.PDE(
    geometry=geom,
    pde=ode,
    bcs=[observe_y_bc],
    num_domain=200,
    num_boundary=7,
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
    lr=0.001,
    external_trainable_variables=[r, k]
)


early_stop = dde.callbacks.EarlyStopping(monitor="loss_train",
                                         baseline=1e-4,
                                         start_from_epoch=20000)
variable = dde.callbacks.VariableValue(
                                        var_list=[r,k], 
                                        period=600, 
                                        filename="parameters.dat"
                                    )
callbacks = [
    early_stop,
    variable
]





# ============================================================================== #
#                               ENTRENAMIENTO
# ============================================================================== #


print("Entrenando PINN...")
os.makedirs('models/verhulst',exist_ok=True)
losshistory, train_state = model.train(
                                        iterations=ITERATIONS,
                                        callbacks=callbacks,
                                        model_save_path="models/verhulst/cp"
                                    )

dde.saveplot(losshistory, train_state, isplot=True, issave=False)

with open(file='parameters.dat',mode='r') as f:
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
os.makedirs('figures',exist_ok=True)

T = np.linspace(t0, tf, 200).reshape(-1, 1)
pred = model.predict(T)
real = exact_solution(T, y0, params[0], params[1])

plt.figure(figsize=(8, 5))
plt.plot(T, real, label="Solución exacta (con parámetros aprendidos)", linewidth=4)
plt.plot(T, pred, "--", label="Predicción PINN", linewidth=4)
plt.scatter(t_train, y_train, color="black", label="Datos de entrenamiento")
plt.scatter(t_test, y_test, color="red", label="Datos test")

plt.xlabel("Tiempo de Fermentación(h)")
plt.ylabel("Concentración (g/cm³)")
plt.legend()
plt.grid()
plt.savefig("figures/verhulst.pdf")
plt.show()