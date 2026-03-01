# Entrenamiento

El entrenamiento de las (PINNs) en este proyecto sigue un flujo de trabajo estructurado que permite abordar tanto problemas directos como inversos asociados al crecimiento microbiano de los gránulos de kéfir de agua. 

## Definición del modelo físico y del dominio
El primer paso consiste en definir el modelo de crecimiento poblacional que describe el fenómeno de estudio, típicamente mediante ecuaciones diferenciales ordinarias de tipo logístico o Gompertz. Estas ecuaciones representan una aproximación física del crecimiento microbiano bajo condiciones controladas de temperatura y presión, y constituyen la restricción principal que se impondrá durante el entrenamiento.

Asimismo, se define el dominio temporal del problema, correspondiente al intervalo experimental de 175 horas, junto con las condiciones iniciales asociadas a la concentración inicial de gránulos de kéfir.

## Construcción de la PINN

A continuación, se construye una red neuronal profunda que recibe como entrada el tiempo $t$ y produce como salida una aproximación continua de la población microbiana $\hat{P}(t)$. En el caso del problema inverso, los parámetros biológicos desconocidos del modelo de crecimiento se incorporan explícitamente como variables entrenables adicionales.

Mediante diferenciación automática, se calculan las derivadas temporales de la salida de la red, necesarias para evaluar el residuo de la ecuación diferencial que gobierna el crecimiento microbiano.

## Formulación de la función de pérdida
La función de pérdida se define como una combinación ponderada de distintos términos que reflejan los objetivos del entrenamiento. Para el problema directo, la pérdida incluye principalmente:
- El residuo de la ecuación diferencial de crecimiento, evaluado en un conjunto de puntos de colocación en el dominio temporal.


- El cumplimiento de las condiciones iniciales, garantizando la unicidad de la solución.


En el problema inverso, la función de pérdida se amplía para incluir:
- El residuo físico de la ecuación diferencial, dependiente de los parámetros a inferir.


- Los términos asociados a las condiciones iniciales.


Un término adicional de ajuste a los datos experimentales, correspondiente a las mediciones discretas del crecimiento de los gránulos de kéfir para cada pretratamiento de ultrasonido.


Esta formulación permite entrenar la red de manera que la solución reconstruida sea simultáneamente consistente con la dinámica física y con las observaciones experimentales.

## Entrenamiento y optimización
El entrenamiento de la PINN se realiza mediante algoritmos de optimización basados en gradiente, ajustando tanto los pesos de la red como, en el caso inverso, los parámetros biológicos del modelo. Durante este proceso, se busca minimizar la función de pérdida global, equilibrando el ajuste a datos y el cumplimiento de las ecuaciones gobernantes.

La correcta ponderación de los términos de la pérdida es un aspecto clave del flujo de trabajo, ya que un desbalance puede afectar la estabilidad del entrenamiento o sesgar la inferencia de parámetros.

## Evaluación y análisis de resultados
Una vez entrenada la PINN, se evalúa su desempeño comparando la solución reconstruida con los datos experimentales disponibles. En el caso del problema inverso, los parámetros inferidos se analizan e interpretan desde un punto de vista biológico, evaluando su relación con el pretratamiento de ultrasonido aplicado a los gránulos de kéfir.
Finalmente, los resultados obtenidos con distintas configuraciones de PINNs se comparan utilizando métricas como el coeficiente de determinación ($R^2$) y criterios de información (AIC y BIC), lo que permite seleccionar el modelo más adecuado para describir la dinámica de crecimiento microbiano en función del tratamiento experimental.
