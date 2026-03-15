# Vistazo dentro de la caja negra
Redes Neuronales Informadas por Modelos Físicos (Physics-Informed Neural Networks, PINNs) constituyen una metodología de modelado computacional que integra explícitamente el conocimiento físico de un sistema dentro del proceso de entrenamiento de una red neuronal. A diferencia de los enfoques puramente basados en datos, las PINNs incorporan ecuaciones diferenciales ordinarias (ODEs), parciales (PDEs) u otras restricciones físicas directamente en la función de pérdida, forzando a la red a aprender soluciones que no solo se ajusten a los datos experimentales, sino que también respeten las ecuaciones gobernantes del fenómeno de estudio.{cite}`Cuomo2022,Lu2021`

En el contexto de este proyecto, PINNs se emplean para modelar el crecimiento microbiano de los gránulos de kéfir de agua, donde la dinámica poblacional ideal puede describirse mediante ecuaciones diferenciales de tipo logístico / Gompertz.

La motivación principal para el uso de PINNs radica en su capacidad para combinar la flexibilidad de aproximación de las redes neuronales con la interpretabilidad de los modelos clásicos. El uso de PINNs resulta particularmente adecuado en escenarios con datos escasos, incompletos o ruidosos —como las series de tiempo disponibles—, permitiendo inferir parámetros efectivos y dinámicas ocultas sin imponer formas funcionales rígidas.



El mecanismo de las redes neuronales informadas por física se puede seccionar en tres elementos: la red neuronal, diferenciación automática , retropropagación . En nuestro caso particular, estamos lidiando con un problema gobernado por una ecuación ordinaria de forma:

$$
\begin{equation*}
    \frac{du}{dt} = f(t,u;\lambda) \hspace{10mm};  B(t,u) = g(t)
\end{equation*}
$$


donde $u$ representa la población en un instante dado, y $\lambda$ parametros que caracterizan el sistema físico. 

Entonces, si quisieramos resolver este sistema, la red neuronal $\hat u$ vendría siendo una red con arquitectura de elección propia. Esta se somete bajo un entrenamiento usual de aprendizaje automático, el cambio rádicando en el uso de diferenciación automática para evaluar residuos de las ecuaciones diferenciales. Esto evita discretizaciones explícitas y permite trabajar con dominios continuos {cite}`Cuomo2022` y es necesario para evaluar componentes clave de la función de perdida. La forma de esta función de onda dependerá del tipo de problema estemos atacando: directo o inverso. 

Un problema directo consistiría en determinar la evolución temporal de la población $u(t)$ a partir de la ecuación diferencial conocida junto con los parámetros $\lambda$ previamente establecidos. En nuestro caso , esto implicaría resolver ecuaciones de crecimiento —como los modelos logístico o de Gompertz— asumiendo valores fijos para parámetros como la tasa de crecimiento $k$, la capacidad de carga $m$ con condición inicial $u(t=0)$. En escenarios experimentales reales, muchos de estos parámetros no son directamente observables o pueden variar en función de fuerzas externas. Este planteamiento da lugar a un problema inverso, siendo que partimos de una curva y lo que se quiere inferir son los parámetros que generan esta curva partiendo de una ecuación diferencial.

PINNs han demostrado ser particularmente eficaces para la formulación y resolución de problemas inversos, ya que permiten tratar los parámetros desconocidos de una ecuación diferencial como variables adicionales a optimizar durante el entrenamiento. En este marco, tanto la solución $\hat{u}(t)$ como los parámetros $\lambda$ (por ejemplo, $r$, $m$ ) se parametrizan mediante la red neuronal. El procedimiento general consiste en imponer las ecuaciones gobernantes del crecimiento microbiano dentro de la función de pérdida, de modo que el residuo físico dependa explícitamente de los parámetros desconocidos. A partir de datos experimentales parciales —las series de tiempo del crecimiento de los gránulos de kéfir—, la red se entrena para encontrar simultáneamente una solución consistente con los datos y un conjunto de parámetros que satisfagan la estructura física del sistema.



# Construcción de función de pérdida 
La idea central de las PINNs se basa en la construcción de una función de pérdida compuesta, diseñada para equilibrar simultáneamente el ajuste a los datos experimentales y el cumplimiento de las ecuaciones diferenciales que gobiernan el crecimiento microbiano. La formulación típica de la función de perdida para problemas inversos viene siendo:

$$
\begin{equation*}
\mathcal{L}(\theta,\lambda)=w_D\mathcal{L}_D(\theta)+w_F\mathcal{L}_F(\theta,\lambda)+w_B\mathcal{L}_B(\theta)    
\end{equation*}
$$

donde $\theta$ representa los parámetros entrenables de la red,$\mathcal{L}_F$ mientras que $\mathcal{L}_F$ ,$\mathcal{L}_B$ las perdidas asociadas al fenómeno físico  , y los pesos $w$ controlan la contribución relativa de cada término. 

El término asociado a las ecuaciones gobernantes se define como

$$
\begin{equation*}
\mathcal{L}_F(\theta,\lambda)=\frac{1}{|T_F|}\sum_{t\in T_F}||\frac{d\hat{u}_\theta}{dt}-f(t,\hat{u};\lambda)||^2
\end{equation*}
$$

donde $\hat{u}$ es la salida de la red neuronal y$\lambda$ corresponde a parámetros desconocidos que pueden ser inferidos durante el entrenamiento. Es decir, son variables que se estimaran junto con los parámetros de la red. De manera análoga, el término de condiciones iniciales y de frontera se expresa como

$$
\begin{equation*}
\mathcal{L}_B(\theta)=\frac{1}{|T_B|}\sum_{t_i\in T_B}||\mathcal{B}(\hat{u}_\theta,t_i)-g(t_i)||^2
\end{equation*}
$$

asegurando que la solución aprendida sea consistente con las condiciones experimentales del sistema, como la biomasa inicial de los gránulos de kéfir. Finalmente, se agrega el término asociado a que tan bien se ajusta la red a los datos observados y la clave de la resolución del problema inverso :

$$
\begin{equation*}
    \mathcal{L}_D(\theta)=\frac{1}{|T_D|}\sum_{t_i\in T_D}||\hat{u}_\theta(t_i)-x_i||^2
\end{equation*}
$$

donde $x_i$ es el dato observado en el instante $t_i$.

Este enfoque permite utilizar modelos poblacionales clásicos como guías físicas, mientras se exploran y cuantifican los efectos del pretratamiento de ultrasonido sobre la dinámica de crecimiento microbiano del kéfir de agua.

