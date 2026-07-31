# Problema inverso
En el modelado matemático del crecimiento microbiano, dijimos que un problema directo consiste en determinar la evolución temporal de la población $y(t)$ a partir de una ecuación diferencial conocida y un conjunto de parámetros biológicos $\phi$ previamente establecidos. En el contexto del kéfir de agua, esto implicaría resolver ecuaciones de crecimiento —como los modelos logístico o de Gompertz— asumiendo valores fijos para parámetros como la tasa de crecimiento $r$, la capacidad de carga $m$ y condiciones iniciales $y(t=0)$.

Sin embargo, en escenarios experimentales reales, muchos de estos parámetros no son directamente observables o pueden variar en función de tratamientos externos, como el pretratamiento por ultrasonido. En estos casos, el interés principal no radica únicamente en predecir la dinámica poblacional, sino en inferir los parámetros biológicos efectivos que gobiernan el crecimiento microbiano bajo distintas condiciones. Este planteamiento da lugar a un problema inverso, siendo que partimos de una curva y lo que se quiere inferir son los parámetros que generan esta curva dada una ecuación diferencial.

Las Redes Neuronales Informadas por Modelos Físicos (PINNs) han demostrado ser particularmente eficaces para la formulación y resolución de **problemas inversos**, ya que permiten tratar los parámetros desconocidos de la ecuación diferencial como variables adicionales a optimizar durante el entrenamiento. En este marco, tanto la solución $\hat{y}(t)$ como los parámetros biológicos $\phi$ (por ejemplo, $r$, $m$ o parámetros dependientes del ultrasonido como la intensidad $I$ y periodo de exposición $T$) se parametrizan mediante la red neuronal.

El procedimiento general consiste en imponer las ecuaciones gobernantes del crecimiento microbiano dentro de la función de pérdida, de modo que el residuo físico dependa explícitamente de los parámetros desconocidos. Entonces, definimos una función de residuos $R(\hat y,x;\phi)$:

$$
R(\hat y,x;\phi) = \frac{d\hat y}{dt} -{f}(\hat y,x;\phi) ,
$$

donde $f( y,t;\phi) $ representa la función del lado derecho la ecuación diferencial ordinaria {eq}`general_ode`.


A partir de datos experimentales parciales —las series de tiempo del crecimiento de los gránulos de kéfir—, la red se entrena para encontrar simultáneamente una solución consistente con los datos y un conjunto de parámetros que satisfagan la estructura física del sistema.

## Componentes de la función de perdida

La principal diferencia entre la formulación de problemas directos e inversos dentro del marco PINN radica en la función de pérdida, que en el caso inverso incorpora explícitamente términos asociados a los datos observados. Una formulación típica es {eq}`total_loss_function`, donde $\theta$ son los parámetros de la red neuronal y $\phi$ representa los parámetros físicos o biológicos a estimar. 

Este enfoque resulta especialmente atractivo para el estudio del crecimiento microbiano del kéfir de agua, ya que permite trabajar con datos escasos e incompletos sin recurrir a discretizaciones finas ni simulaciones numéricas tradicionales. Además, posibilita la inferencia de parámetros biológicos difíciles de medir experimentalmente, proporcionando información cuantitativa sobre el efecto del pretratamiento de ultrasonido en la dinámica de crecimiento.

En el caso específico que trabajamos,{eq}`total_loss_function` toma la forma: 

$$
 \mathcal{L}(\theta,r,m;X=\{C,D\})=\mathcal{L}_F+\mathcal{L}_D,
$$

donde se asume en el caso de modelo Verhulst:

$$
\mathcal{L}_F  = \sum_{t_i \in C_n}\frac{1}{2}\Big|\Big|R(\hat y_i,t_i;r,m)\Big|\Big|^2
$$

$$
\mathcal{L}_D  =\sum_{t_i\in \mathcal{D}}\frac{1}{2}\Big|\Big|y_i-\hat y_i\Big|\Big|^2
$$

$$
R(\hat y,t;r,m) = \frac{d\hat y}{dt} -r\hat y\Big( 1- \frac{1}{m}\hat y \Big) ,
$$


y la derivada de $\hat y$ se cálcula utilizando un método númerico de diferenciación automátizada. Aquí, queremos estimar valores puntuales de los parametros $\kappa$ y $L$.

En resumen, las PINNs ofrecen un marco unificado para abordar problemas directos e inversos de manera coherente, integrando datos experimentales y conocimiento físico-biológico. En este proyecto, el interés principal se centra en la resolución del problema inverso, orientado a la identificación de parámetros y dinámicas ocultas asociadas al crecimiento microbiano del kéfir de agua bajo distintos pretratamientos.

## Implementación de DeepXDE 
Con el apoyo de la paquetería [DeepXDE](https://deepxde.readthedocs.io/en/latest/index.html), podemos lograr esto definiendo las variables `kappa`  y `L` como variables entrenables partiendo de un valor inicial que queramos de cual partir: 

```python
import deepxde as dde


kappa = dde.Variable(0.04)
L = dde.Variable(51.0)

def verhulst_eq(x, y):
        dy_dt = dde.grad.jacobian(y, x, i=0, j=0)
    return (dy_dt- kappa * y * (1 - y / L))
```
De esta manera, al momento de compilar y entrenar el PINN, se aprenden tanto los parametros $\theta$ de la red neuronal como los parametros $\kappa$ y $L$.

```python
from kefir_ajuste.trainers import verhulst

model_equation = verhulst
run_name = f"{model_equation.__name__}_{LEARNING_RATE}_{EPOCHS}"
with mlflow.start_run(run_name=run_name):
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("learning_rate", LEARNING_RATE)

    model, loss_history,learned_parameters, y_true, y_pred = model_equation(dataset=dataset,
                                                                            epochs=EPOCHS,lr=LEARNING_RATE,          collocation_method=collocation_method)
                                            
```

donde primero definimos el espacio de datos de entrenamiento que se utilizaran para la PINN:

```python

geom = dde.geometry.TimeDomain(t0, tf)
ic = dde.icbc.IC(geom,
                 lambda t: y0,
                 lambda _, on_initial: on_initial)

data_pinn = dde.data.PDE(geometry=geom,
                         pde=ode,
                         bcs=[ observe_bc],
                         num_domain=200,
                         num_boundary=2,
                         num_test=100,
                         anchors=anchor_t)
```

Después, definimos la arquitectura de nuestra red neuronal de dirección directa, pero especificando qué parametros queremos aprender aparte de los pesos de las funciones de activación utilizando la variable `external_trainable_variables`:

```python
layer_size = [1, 50, 50, 50, 1]
net = dde.nn.FNN(layer_size,
                 activation="tanh",
                 kernel_initializer="Glorot uniform")

model = dde.Model(data_pinn, net)
model.compile(optimizer="adam",
              loss="MSE",
              lr=lr,
              external_trainable_variables=[kappa, L])
```

Finalmente, definimos cómo guardar los valores aprendidos durante el entrenamiento de nuestra red, utilizando el método `dde.callbacks.VariableValue`:
```python

variable = dde.callbacks.VariableValue(var_list=[kappa,L], 
                                       period=600, 
                                       filename=VARIABLES_PATH)
callbacks = [variable]

loss_history, _ = model.train(iterations=epochs,
                              callbacks=callbacks)
```


