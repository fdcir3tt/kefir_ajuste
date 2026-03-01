# Modelos de crecimiento microbiano 

El crecimiento de comunidades microbianas, como las presentes en los gránulos de kéfir de agua, puede describirse mediante modelos matemáticos de tipo poblacional. Estos modelos permiten representar la evolución temporal de la biomasa microbiana $P(t)$  bajo ciertas hipótesis biológicas y ambientales, y han sido ampliamente utilizados en el estudio de fermentaciones y sistemas probióticos.

El modelo Malthusiano asume que la tasa de crecimiento es proporcional al tamaño poblacional, lo que conduce a un crecimiento exponencial. Si bien este modelo puede aproximar fases iniciales de crecimiento microbiano, resulta insuficiente para describir sistemas reales como el kéfir, donde existen limitaciones de nutrientes, competencia entre microorganismos y efectos inducidos por el procesamiento.

La premisa en forma matemática, de las tazas de nacimiento $b(t)$ y de muertes $d(t)$ . Se propone que tanto $b(t)$ como $d(t)$ comparten una relación de proporcionalidad con el tamaño de población $P(t)$ :

```{math}
:label: rates
b(t)=\beta P(t), \quad d(t)=\delta P(t)
```


De aquí, se plantea que la taza en la que cambia el tamaño de población se da por la diferencia entre las tazas de nacimiento y muerte:

```{math}
\frac{dP}{dt}=b(t)-d(t)
```


Teniendo en mente las expresiones de {eq}`rates` , obtenemos la ecuación 

```{math}
:label: eq-logistic
\frac{dP}{dt}=rP(t)
```


Donde $r=\beta-\delta$ es una constante real. El modelo Veltusiano también considera las mismas premisas y la forma de {eq}`eq-logistic`, solo cambian  $b(t)$ y $d(t)$. Con este modelo se considera que el cociente $b(t)/P(t)$ en vez de mantenerse constante, se considera que este decrese linealmente con respecto a la población: $b(t)/P(t)=\beta - k_\beta P(t)$ . De manera similar con la taza de muerte. Como consecuencia, obtenemos:

$\begin{equation}
    b(t)=(\beta - k_{\beta}P(t)) P(t) ,\hspace{5mm} d(t)=(\delta + k_{\delta}P(t)) P(t)
\end{equation}$

Teniendo esto en mente, a partir de $(2)$ podemos 
$$
\frac{dP}{dt}=(\beta - k_{\beta}P) P-(\delta + k_{\delta}P) P
$$

$$
=(\beta - k_{\beta}P-\delta -k_{\delta}P) P
$$

$$
=(\beta -\delta -(k_{\beta}+k_{\delta})P) P
$$

$\begin{equation}
=(\beta -\delta)\left(1 -\frac{k_{\beta}+k_{\delta}}{\beta -\delta}P\right) P
\end{equation}$


Para incorporar estas restricciones, el modelo logístico introduce una capacidad de carga $m$, asociada a la disponibilidad de recursos y al entorno físico-químico del medio de cultivo.

Podemos renombrar los términos de (5) en forma compacta para por fin llegar a la ecuación logística:

$\begin{equation}
    \frac{dP}{dt}=rP(1-\frac{P}{m})
\end{equation}$


$$
    r=\beta -\delta,\hspace{5mm}m=\frac{\beta -\delta}{k_{\beta}+k_{\delta}}
$$



Donde $k_\beta$ y $k_\delta$ son constantes reales positivas.


 
Este modelo describe adecuadamente la dinámica sigmoidal observada en muchos procesos fermentativos, permitiendo además definir parámetros biológicos relevantes, como la rapidez máxima de crecimiento $\mu$ y el tiempo de retraso $\lambda$. Sin embargo, el modelo logístico supone una respuesta simétrica alrededor del punto de máxima rapidez, lo cual no siempre se observa en cultivos microbianos complejos.


El modelo de Gompertz ofrece una alternativa más flexible para describir el crecimiento microbiano asimétrico, característica común en fermentaciones reales. Sus parámetros permiten capturar de manera más realista las fases de adaptación, crecimiento acelerado y saturación. En su forma matemática tenemos:

$$
\begin{equation}
\frac{dP}{dt}=rP\hspace{0.5mm}ln(\frac{m}{P})
\end{equation}
$$


Aun así, tanto el modelo logístico como el de Gompertz dependen de supuestos funcionales específicos y parámetros constantes, lo que limita su capacidad para representar fenómenos no lineales inducidos por tratamientos externos. Sin embargo, nos serán útiles para la etapa de ajuste de PINNs con la serie de tiempo testigo ya que esta sigue un comportamiento ideal. 

En el caso particular de los demas experimentos, el crecimiento microbiano puede verse afectado por pretratamientos de ultrasonido, los cuales introducen perturbaciones físicas que modifican la estructura del consorcio microbiano y, potencialmente, los mecanismos de crecimiento. Estos efectos no siempre pueden ser capturados adecuadamente por modelos clásicos con parámetros fijos o formas funcionales predefinidas.

Es por eso que, las redes neuronales informadas por modelos físicos (Physics-Informed Neural Networks, PINNs) representan una alternativa robusta para el análisis del crecimiento microbiano. Las PINNs permiten integrar ecuaciones diferenciales poblacionales —como las del modelo logístico o de Gompertz— directamente en el proceso de entrenamiento de la red, al mismo tiempo que aprenden dinámicas no observables a partir de datos experimentales. De esta forma, es posible identificar parámetros efectivos dependientes del pretratamiento, modelar dinámicas ocultas y capturar desviaciones respecto a los modelos clásicos, incluso en escenarios con datos escasos, como las series de tiempo disponibles para el crecimiento de gránulos de kéfir de agua.

Este enfoque híbrido combina la interpretabilidad de los modelos biológicos tradicionales con la flexibilidad de las técnicas de aprendizaje profundo, ofreciendo una herramienta adecuada para caracterizar y comparar el efecto del ultrasonido sobre el crecimiento microbiano del kéfir.
