# Problema inverso
En el modelado matemático del crecimiento microbiano, un problema directo consiste en determinar la evolución temporal de la población $P(t)$ a partir de una ecuación diferencial conocida y un conjunto de parámetros biológicos $\lambda$ previamente establecidos. En el contexto del kéfir de agua, esto implicaría resolver ecuaciones de crecimiento —como los modelos logístico o de Gompertz— asumiendo valores fijos para parámetros como la tasa de crecimiento $k$, la capacidad de carga $m$ y condiciones iniciales $P(t=0)$.

Sin embargo, en escenarios experimentales reales, muchos de estos parámetros no son directamente observables o pueden variar en función de tratamientos externos, como el pretratamiento por ultrasonido. En estos casos, el interés principal no radica únicamente en predecir la dinámica poblacional, sino en inferir los parámetros biológicos efectivos que gobiernan el crecimiento microbiano bajo distintas condiciones. Este planteamiento da lugar a un problema inverso, siendo que partimos de una curva y lo que se quiere inferir son los parámetros que generan esta curva dada una ecuación diferencial.

Las Redes Neuronales Informadas por Modelos Físicos (PINNs) han demostrado ser particularmente eficaces para la formulación y resolución de problemas inversos, ya que permiten tratar los parámetros desconocidos de la ecuación diferencial como variables adicionales a optimizar durante el entrenamiento. En este marco, tanto la solución $\hat{P}(t)$ como los parámetros biológicos $\lambda$ (por ejemplo, $r$, $m$ o parámetros dependientes del ultrasonido) se parametrizan mediante la red neuronal.

El procedimiento general consiste en imponer las ecuaciones gobernantes del crecimiento microbiano dentro de la función de pérdida, de modo que el residuo físico dependa explícitamente de los parámetros desconocidos. A partir de datos experimentales parciales —las series de tiempo del crecimiento de los gránulos de kéfir—, la red se entrena para encontrar simultáneamente una solución consistente con los datos y un conjunto de parámetros que satisfagan la estructura física del sistema.

La principal diferencia entre la formulación de problemas directos e inversos dentro del marco PINN radica en la función de pérdida, que en el caso inverso incorpora explícitamente términos asociados a los datos observados. Una formulación típica es

```{math}
 \mathcal{L}(\theta,\lambda;T)=w_f\mathcal{L}_f(\theta,\lambda;T_f)+w_b\mathcal{L}_b(\theta,\lambda;T_b)+w_i\mathcal{L}_i(\theta,\lambda;T_i)
```



donde $\theta$ son los parámetros de la red neuronal y $\lambda$ representa los parámetros físicos o biológicos a estimar. El término de información experimental se define como

```{math}
\mathcal{L}_i(\theta,\lambda;T_i)=\frac{1}{|T_i|}\sum_{x\in T_i}||\mathcal{I}(\hat{u},x)||^2_2
```

asegurando la compatibilidad entre la solución reconstruida y las mediciones experimentales disponibles.
Este enfoque resulta especialmente atractivo para el estudio del crecimiento microbiano del kéfir de agua, ya que permite trabajar con datos escasos e incompletos sin recurrir a discretizaciones finas ni simulaciones numéricas tradicionales. Además, posibilita la inferencia de parámetros biológicos difíciles de medir experimentalmente, proporcionando información cuantitativa sobre el efecto del pretratamiento de ultrasonido en la dinámica de crecimiento.
En resumen, las PINNs ofrecen un marco unificado para abordar problemas directos e inversos de manera coherente, integrando datos experimentales y conocimiento físico-biológico. En este proyecto, el interés principal se centra en la resolución del problema inverso, orientado a la identificación de parámetros y dinámicas ocultas asociadas al crecimiento microbiano del kéfir de agua bajo distintos pretratamientos.
