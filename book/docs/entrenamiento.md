# Definición del modelo físico y dominio

El primer paso consiste en definir el modelo de crecimiento poblacional que describe el
fenómeno de estudio, típicamente mediante ecuaciones diferenciales ordinarias de tipo 
logístico o Gompertz. Estas ecuaciones representan una aproximación física del 
crecimiento microbiano bajo condiciones controladas de temperatura y presión, y 
constituyen la restricción principal que se impondrá durante el entrenamiento. Se define 
el dominio temporal del problema, correspondiente al intervalo experimental de 175 horas, 
junto con las condiciones iniciales asociadas a la concentración inicial de gránulos de 
kéfir.

# Construcción de la red neuronal

Después, se construye una red neuronal profunda que recibe como entrada el tiempo $t$ 
y produce como salida una aproximación continua de la población microbiana $\hat{u}(t)$. 
En el caso del problema inverso, los parámetros biológicos desconocidos del modelo de 
crecimiento se incorporan explícitamente como variables entrenables adicionales. Mediante 
diferenciación automática, se calculan las derivadas temporales de la salida de la red, 
necesarias para evaluar el residuo de la ecuación diferencial que gobierna el crecimiento 
microbiano.

El tipo de red neuronal que se utiliza frecuentemente en el campo de aprendizaje 
automatizado científico suele ser una red neuronal profunda (Deep Neural Network, DNN)
de  2-3 capas ocultas unidireccionales de 50-100 neuronas . Las funciones de activación 
que  han mostrado buen desempeño en mayoría de casos de uso de problemas inversos con 
 ecuaciones diferenciales ordinarias han sido  $\tanh$, $\text{ReLu}$  ,$\text{Softmax}$ 
{cite}`Cuomo2022,Pappu2025,Farea2025` .

# Optimización
El entrenamiento de la PINN se realiza mediante algoritmos de optimización basados en 
 pasos de gradiente, ajustando tanto los pesos de la red como  los parámetros del modelo. 
 Durante este proceso, se busca minimizar la función de pérdida global, equilibrando el 
 ajuste a datos y el cumplimiento de las ecuaciones gobernantes.

La correcta ponderación de los términos de la pérdida es un aspecto clave del flujo 
de trabajo, ya que un desbalance puede afectar la estabilidad del entrenamiento o sesgar 
la inferencia de parámetros.

# Evaluación y análisis de resultados

Una vez entrenada la red, se evalúa su desempeño comparando la solución
reconstruida con los datos experimentales disponibles. En el caso del problema inverso, 
los parámetros inferidos se analizan e interpretan desde un punto de vista biológico, 
evaluando su relación con el pretratamiento de ultrasonido aplicado a los gránulos de 
kéfir.

Finalmente, los resultados obtenidos con distintas configuraciones (tanto modelo 
matemático como red ) se comparan utilizando métricas de desempeño, lo que nos permitirá 
seleccionar el modelo más adecuado para describir la dinámica de crecimiento microbiano 
en función del tratamiento experimental.


