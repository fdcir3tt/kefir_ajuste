# Construcción de la red neuronal

El tipo de red neuronal que se utiliza frecuentemente en el campo de aprendizaje 
automatizado científico suele ser una red neuronal profunda (Deep Neural Network, DNN)
de  2-3 capas ocultas unidireccionales de 50-100 neuronas . Las funciones de activación 
que  han mostrado buen desempeño en mayoría de casos de uso de problemas inversos con 
 ecuaciones diferenciales ordinarias han sido  $\tanh$, $\text{ReLu}$  ,$\text{Softmax}$ 
{cite}`Cuomo2022,Pappu2025,Farea2025` .

Después, se construye una red neuronal profunda que recibe como entrada el tiempo $t$ 
y produce como salida una aproximación continua de la población microbiana $\hat{u}(t)$. 
En el caso del problema inverso, los parámetros biológicos desconocidos del modelo de 
crecimiento se incorporan explícitamente como variables entrenables adicionales. Mediante 
diferenciación automática, se calculan las derivadas temporales de la salida de la red, 
necesarias para evaluar el residuo de la ecuación diferencial que gobierna el crecimiento 
microbiano. 

La arquitectura de las redes utilizadas consisten de:

**Modelo utilizado para problema inverso** 
- **Capa de entrada**: Capa de una sola neurona con función de activación $\sigma = \tanh$.

- **Capas ocultas**: 3 capas ocultas densas de 50 neuronas cada una con función de activación $\sigma = \tanh$.

- **Capa de salida**: Capa de una sola neurona con función de activación $\sigma = \tanh$.


- **Representación matemática**:

$$
NN(t) = \sigma(W_4\hspace{1mm}...\hspace{1mm}\sigma(W_1t+\mathbf{b}_1)+\mathbf{b}_4).
$$

- **Pesos**:

```{math}
\begin{align}
dim(W_1) &= 1\times50 = 50 \\
dim(W_2) &= 50\times50 = 2500\\
dim(W_3) &= 50\times50 = 2500\\
dim(W_4) &= 50\times1 = 50
\end{align}
```

- **Sesgos**:

```{math}
\begin{align}
dim(\mathbf{b}_1) &= 50 \\
dim(\mathbf{b}_2) &= 50\\
dim(\mathbf{b}_3) &= 50\\
dim(\mathbf{b}_4) &= 1
\end{align}
```

- **Parámetros entrenables**: Total de 5251 parámetros 


**Modelo utilizado en descubrimiento físico** 
- **Capa de entrada**: Capa de una sola neurona con función de activación $\sigma = \tanh$.

- **Capas ocultas**: 3 capas ocultas densas de 50 neuronas cada una con función de activación $\sigma = \tanh$.

- **Capa de salida**: Capa de una sola neurona con función de activación $\sigma = \tanh$.

- **Representación matemática**:

$$
NN(\mathbf{x}) = \sigma(W_4\hspace{1mm}...\hspace{1mm}\sigma(W_1\mathbf{x}+\mathbf{b}_1)+\mathbf{b}_4).
$$

- **Pesos**:

```{math}
\begin{align}
dim(W_1) &= 3\times50  = 150 \\
dim(W_2) &= 50\times50 = 2500\\
dim(W_3) &= 50\times50 = 2500\\
dim(W_4) &= 50\times1  = 50
\end{align}
```

- **Sesgos**:

```{math}
\begin{align}
dim(\mathbf{b}_1) &= 50 \\
dim(\mathbf{b}_2) &= 50 \\
dim(\mathbf{b}_3) &= 50 \\
dim(\mathbf{b}_4) &= 1
\end{align}
```

- **Parámetros entrenables**: Total de 5351 parámetros


## Definición del modelo físico y dominio

Definimos el modelo de crecimiento poblacional que describe el
fenómeno de estudio, típicamente mediante ecuaciones diferenciales ordinarias de tipo 
logístico o Gompertz que constituyen la restricción principal que se impondrá durante el entrenamiento. Se define 
el dominio temporal del problema, correspondiente al intervalo experimental de 168 horas, 
junto con las condiciones iniciales asociadas a la concentración inicial de gránulos de 
kéfir.

## Optimización

El entrenamiento de la PINN se realiza mediante el algoritmo de optimización de ADAM ,ya que resulta ser eficiente para problemas inversos {cite}`Pappu2025,Farea2025` y en general en el entrenamiento de perceptrones de multiples capas {cite}`kingma2017`. ADAM consiste de un algoritmo de descenso de gradiente que hace uso de estimaciones del promedio y la varianza del gradiente para actualizar mejor los pasos de gradiente {cite}`kingma2017`.

En el proceso de entrenamiento se van ajustando tanto los pesos de la red como  los parámetros del modelo. Durante este proceso, se busca minimizar la función de pérdida global, equilibrando el ajuste a datos y el cumplimiento de las ecuaciones gobernantes.

La correcta ponderación de los términos de la función de pérdida constituye un aspecto fundamental del flujo de trabajo, ya que un desbalance entre ellos puede comprometer la estabilidad del entrenamiento o introducir sesgos en la inferencia de parámetros. En este trabajo se empleará la función de pérdida definida en {eq}`total_loss_function`, cuyos términos se describen en detalle según el proceso considerado en las secciones correspondientes (véanse las páginas _Problema inverso_ y _Descubrimiento de dinámicas ocultas con PINNs_)

## Evaluación y análisis de resultados

Una vez entrenada la red, se evalúa su desempeño comparando la solución reconstruida con los datos experimentales disponibles. Los resultados obtenidos con las distintas configuraciones (tanto de modelo matemático como de red) se comparan utilizando métricas de desempeño, lo que permite seleccionar el modelo más adecuado para describir la dinámica de crecimiento microbiano en función del tratamiento experimental.

Con el objetivo de evaluar y comparar los resultados obtenidos al resolver problemas inversos mediante distintas configuraciones de Redes Neuronales Informadas por Modelos Físicos (PINNs), se emplean métricas estadísticas ampliamente utilizadas en el análisis de modelos de regresión. En particular, se utilizan las métricas error cuadrático medio (RMSE), error absoluto medio (MAE) y error porcentual absoluto medio (MAPE), en ese orden de prioridad, para evaluar el desempeño. Posteriormente, se van a comparar el coeficiente de determinación ($R^2$), el criterio de información de Akaike (AIC) y el criterio de información Bayesiano (BIC), los cuales permiten cuantificar tanto la calidad del ajuste a los datos experimentales como la complejidad del modelo inferido.

En el contexto de este proyecto, estas medidas se aplican para comparar las diferentes PINNs utilizadas en la estimación de parámetros del modelo de crecimiento microbiano a partir de las series de tiempo del crecimiento de gránulos de kéfir de agua sometidos a distintos pretratamientos de ultrasonido.

### Métrica: Root Mean Square Error (RMSE)

El RMSE es una métrica que mide la diferencia promedio entre los valores predichos por un modelo y los valores reales, penalizando más los errores grandes debido a que se elevan al cuadrado antes de promediar. Se calcula como la raíz cuadrada del promedio de los cuadrados de los errores. Esta métrica es especialmente útil cuando los errores grandes son más críticos y se desea que el modelo evite desviaciones significativas. Un gran error puede tener un impacto mucho mayor que varios errores pequeños ya que estamos trabajando con muy pocos datos, por lo que RMSE ayuda a capturar esa sensibilidad. Se calcula de la siguiente forma{cite}`DraperSmith2014`,(ver {cite}`Willmott2005`,pp 80, ec. 2):
```{math}
:label: rmse_formula
\begin{equation*}
\text{RMSE} = \Bigg[ \frac{1}{\Big|X\Big|} \sum_{x_i \in X} \Big(y_i - \hat{y}_i \Big)^2 \Bigg]^{-\frac{1}{2}}
\end{equation*}
```

### Métrica: Mean Absolute Error (MAE)

El MAE mide el promedio de las diferencias absolutas entre los valores predichos y los reales, sin importar si el error es positivo o negativo. A diferencia del RMSE, no penaliza de forma desproporcionada los errores grandes, por lo que ofrece una visión más “lineal” del desempeño del modelo. Es útil cuando se busca entender el error promedio de manera directa y sencilla, sin que los valores atípicos distorsionen la evaluación. Por ejemplo,nos permite saber, en promedio, cuán lejos se está de la predicción esperada en las mismas unidades de la variable medida. Se calcula (ver {cite}`Willmott2005`,pp 80, ec. 3):

```{math}
:label: mae_formula
\begin{equation*}
\text{MAE} = \frac{1}{\Big| X \Big|} \sum_{x_i \in X} \Big | y_i - \hat{y}_i \Big |
\end{equation*}
```

### Métrica: Mean Absolute Percentage Error (MAPE)

El MAPE expresa el error promedio como un porcentaje de los valores reales, calculando la magnitud del error relativo. Esto permite interpretar los errores de manera proporcional, lo que resulta útil cuando se comparan predicciones en diferentes escalas o se quiere entender el desempeño del modelo en términos porcentuales. Ayuda a evaluar la precisión relativa de las predicciones, evitando que los valores altos dominen la evaluación de todo el modelo. Se calcula (ver {cite}`ArmstrongCollopy1992`, pp 78):

```{math}
:label: mape_formula
\begin{equation*}
\text{MAPE} = \frac{1}{\Big| X \Big|} \sum_{x_i \in X}\Big | \frac{\hat{y}_i-y_i}{y_i} \Big |
\end{equation*}
```

### Criterio de Información de Akaike (AIC)

El criterio de información de Akaike (AIC) evalúa el equilibrio entre la bondad del ajuste del modelo y su complejidad, penalizando explícitamente el número de parámetros estimados. 

En problemas inversos resueltos con PINNs, esta métrica resulta especialmente útil para comparar configuraciones que ajustan los parámetros del modelo de crecimiento bajo distintas arquitecturas de red, funciones de activación o estrategias de ponderación de la función de pérdida{cite}`Baltazar-Larios2025`. Se calcula de la siguiente forma: (ver {cite}`Akaike1974`,pp 719)

```{math}
:label: akaike_information_criteria_formula
\begin{equation*}
\text{AIC} = 2k - 2ln\Big(\text{max}(\mathcal{L}(\theta;X))\Big)
\end{equation*}
```

Un valor menor de AIC indica un modelo más parsimonioso, es decir, aquel que logra describir adecuadamente la dinámica de crecimiento microbiano del kéfir utilizando el menor número efectivo de parámetros. Por ello, las PINNs con valores de AIC más bajos se consideran preferibles desde un punto de vista estadístico.

### Criterio de Información Bayesiano (BIC)

El BIC es conceptualmente similar al AIC, pero introduce una penalización más severa por el número de parámetros, lo que lo hace particularmente adecuado en escenarios con conjuntos de datos pequeños, como las series de tiempo disponibles en este estudio. 
{cite}`Schwarz1978`

```{math}
:label: bayesian_information_criteria_formula
\begin{equation*}
\text{BIC} = 2ln\Big(\text{max}(\mathcal{L}(\theta;X))\Big) - kln(|X|)
\end{equation*}
```

Esta métrica favorece modelos más simples y robustos, reduciendo el riesgo de sobreajuste al resolver problemas inversos. El BIC permite identificar qué configuraciones de PINNs logran capturar el crecimiento de los gránulos de kéfir de agua de manera consistente, manteniendo al mismo tiempo una estructura de modelo lo más sencilla posible. Al igual que en el caso del AIC, valores menores de BIC indican un mejor desempeño global.

### Coeficiente de determinación ($R^2$)

El coeficiente de determinación ($R^2$) mide la proporción de la variabilidad de los datos experimentales que es explicada por el modelo ajustado. Dentro del marco de las PINNs aplicadas a problemas inversos, esta métrica permite evaluar directamente la capacidad de la solución inferida para reproducir las curvas de crecimiento observadas en los gránulos de kéfir de agua.
(ver {cite}`SeberLee2003` Cap. 4.4 ,pp 111,teorema 4.2)

```{math}
:label: multiple_correlation_coefficient_formula
\begin{equation*}
R^2 = 1-\frac{\text{RSS}}{\sum_{x_i\in X}{(y_i-\bar{y})^2}}
\end{equation*} , RSS = \sum_{x_i\in X}(y_i-\hat{y}_i)^2
```

Valores de $R^2$ cercanos a 1 indican un alto grado de concordancia entre las predicciones del modelo de crecimiento —con parámetros inferidos por la PINN— y las mediciones experimentales. No obstante, dado que esta métrica no penaliza explícitamente la complejidad del modelo, su interpretación debe complementarse con criterios de información como el AIC y el BIC.
