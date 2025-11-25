# Modelos Maltusiano y Veltusiano

El modelo Maltusiano es un modelo matemático que tiene de objetivo describir el comportamiento de una cierta población cual denotaremos con $P(t)$.

Parte de la premisa de que esta solo depende de las tazas de nacimiento $b(t)$ y de muertes $d(t)$ . Se propone que tanto $b(t)$ como $d(t)$ comparten una relación de proporcionalidad con el tamaño de población $P(t)$ :

$\begin{equation}
    b(t)=\beta P(t) ,\hspace{5mm} d(t)=\delta P(t)
\end{equation}$

De aquí, se plantea que la taza en la que cambia el tamaño de población se da por la diferencia entre las tazas de nacimiento y muerte:
$\begin{equation}
    \frac{dP}{dt}=b(t)-d(t)
\end{equation}$

Teniendo en mente las expresiones de $(1)$ , obtenemos la ecuación 

$\begin{equation}
    \frac{dP}{dt}=rP(t)
\end{equation}$

Donde $r=\beta-\delta$ es una constante real. El modelo Veltusiano también considera las mismas premisas y la forma de $(2)$, solo cambian  $b(t)$ y $d(t)$. Con este modelo se considera que el cociente $b(t)/P(t)$ en vez de mantenerse constante, se considera que este cociente decrese linealmente con respecto a la población $b(t)/P(t)=\beta - k_\beta P(t)$ . De manera similar con la taza de muerte. Como consecuencia, obtenemos:

$\begin{equation}
    b(t)=(\beta - k_{\beta}P(t)) P(t) ,\hspace{5mm} d(t)=(\delta + k_{\delta}P(t)) P(t)
\end{equation}$

Teniendo esto en mente, a partir de $(2)$ podemos 
$$
\frac{dP}{dt}=(\beta - k_{\beta}P(t)) P(t)-(\delta + k_{\delta}P(t)) P(t)
$$

$$
=(\beta - k_{\beta}P(t)-\delta -k_{\delta}P(t)) P(t)
$$

$$
=(\beta -\delta -(k_{\beta}+k_{\delta})P(t)) P(t)
$$

$\begin{equation}
=(\beta -\delta)\left(1 -\frac{k_{\beta}+k_{\delta}}{\beta -\delta}P(t)\right) P(t)
\end{equation}$

Podemos renombrar los términos en forma compacta para por fin llegar a la ecuación logística:

$$
    r=\beta -\delta,\hspace{5mm}m=\frac{\beta -\delta}{k_{\beta}+k_{\delta}}
$$

Donde $k_\beta$ y $k_\delta$ son constantes reales positivas.

# Ecuación Logística

De esta, obtenemos la ecuación logística:

$\begin{equation}
    \frac{dP}{dt}=rP(t)(1-\frac{P(t)}{m})
\end{equation}$

Se trata de una ecuación diferencial ordinaria separable. 

$$
\int \frac{dP}{P(t)(1-\frac{P(t)}{m})}=\int rdt
$$

Por el método de fracciones parciales podemos reescribir el lado izquierdo de la ecuación de forma más sencilla :

$$
\frac{m}{P(t)(m-P(t))}= \frac{1}{m-P}+\frac{1}{P}
$$

Tal que 

$$
\int \frac{dP}{P(t)(1-\frac{P(t)}{m})}=\int \frac{dP}{m-P}+\int\frac{dP}{P}
$$

$$
\hspace{28mm}=-ln|m-P|+ln|P|
$$