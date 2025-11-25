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

Donde $r=\beta-\delta$ es una constante real. El modelo Veltusiano también considera las mismas premisas y la forma de $(2)$, solo cambian  $b(t)$ y $d(t)$. Con este modelo se considera que el cociente $b(t)/P(t)$ en vez de mantenerse constante, se considera que este decrese linealmente con respecto a la población: $b(t)/P(t)=\beta - k_\beta P(t)$ . De manera similar con la taza de muerte. Como consecuencia, obtenemos:

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
\int \frac{dP}{P(1-\frac{P}{m})}=\int rdt
$$

Por el método de fracciones parciales podemos reescribir el lado izquierdo de la ecuación de forma más sencilla :

$$
\frac{m}{P(m-P)}= \frac{1}{m-P}+\frac{1}{P}
$$

Tal que 

$$
\int \frac{dP}{P(1-\frac{P}{m})}=\int \frac{dP}{m-P}+\int\frac{dP}{P}
$$

$$
\hspace{28mm}=-ln|m-P|+ln|P|
$$

$$
\hspace{14mm}=ln(\frac{P}{m-P})
$$

Sustituyendo se obtiene

$$
ln(\frac{P}{m-P})=rt+C
$$
$$
P=(m-P)e^{rt+C}
$$
$$
P(1+e^{rt+C})=me^{rt+C}
$$

$$
P=\frac{m}{1+e^{-(rt+C)}}
$$

Finalmente, vemos que la curva de población se describe por:

$$
\begin{equation}
P(t)=\frac{m}{1+e^{-(rt-k)}}
\end{equation}
$$

Donde $k>0$, ya que $C=ln(\frac{P_0}{m-P_0})<0$

# Modelo de Gompertz

El modelo de Gompertz es una función exponencial que busca describir también el crecimiento poblacional que tiene la forma:

$$
\begin{equation}
P(t)=ae^{-e^{-(bt+c)}}
\end{equation}
$$

Donde $a,b,c $ son constantes reales. Pero esta forma no nos sirve mucho, nos vendría mejor expresarlo en forma de ecuación diferencial. 


$$
\frac{dP}{dt}=(ae^{-e^{-(bt+c)}})\frac{d}{dt}(-e^{-(bt+c)})
$$

$$
\hspace{1mm}=P[be^{-(bt+c)}]
$$

Vemos que de la expresión (8) tenemos:
$$
lnP=ln(a)-e^{-(bt+c)}
$$

Por lo tanto,podemos escribir la ecuación diferencial de la siguiente forma.

$$
\begin{equation}
\frac{dP}{dt}=rP\hspace{0.5mm}ln(\frac{m}{P})
\end{equation}
$$

Donde

$$
r=b ,\hspace{5mm} ln(\frac{m}{P})=ln(a)-ln(P) \iff m=a 
$$

Si consideramos $P(0)=P_0 $, de (8) obtenemos:

$$
c=-ln(ln(\frac{m}{P_0}))
$$

Y como $ln(\frac{m}{P}) \geq 0$, sabemos que $c<0$. Entonces, podemos reescribir (8) como:

$$
\begin{equation}
P(t)=me^{-e^{-(rt-k)}}\end{equation}

$$

Tener $P(t)$ de esta forma nos convendrá para poder interpretar de forma más intuitiva el rol de cada paramétro en el crecimento de la población. 


# Calculando parámetros biológicos

A continuación, calcularemos los parametros biológicos $\mu$ (rápidez de crecimiento máxima) y $\lambda$ (retraso de crecimiento). 

## Modelo Logístico

Comenzamos calculando $\mu$ primero, ya que lo necesitaremos para lograr calcular $\lambda$ con la recta tangente $T(t)$ que pasa por el punto de máxima rápidez. Con esto, tenemos las siguientes condiciones:

$$
\begin{equation}
\mu=max(\frac{dP}{dt}) ,\hspace{5mm} \lambda=T^{-1}(0)
\end{equation}
$$

Entonces, calculamos la segunda y tercera derivada de $P(t)$ para encontrar $\mu$:

$$
\frac{dP^2}{dt^2}=r\frac{dP}{dt}(1-\frac{P}{m})+rP(-\frac{1}{m}\frac{dP}{dt})
$$

$$
\hspace{-14mm}=r\frac{dP}{dt}(1-\frac{2P}{m})
$$

Ahora, denominaremos $t_c$ como el instante en cuál la rápidez de crecimiento es máxima. Observamos que solo hay dos casos en cuál la segunda derivada es igual a cero. A nosotros nos interesa el caso no trivial ($\frac{dP}{dt}\neq0$):

$$
(1-\frac{2P_c}{m})=0\to P(t_c)=P_c=\frac{m}{2}
$$

Evaluando $\frac{dP}{dt}$ en este instante conseguimos el valor de $\mu$:

$$
\mu=rP_c (1- \frac{P_c}{m})=r\frac{m}{4}
$$

Seguimos con $\lambda$ teniendo en cuenta:

$$
T(t)=\mu t+ T_0 ,\hspace{5mm} T(t_c)=P_c
$$

Con esto

$$
\frac{dP^3}{dt^3}=
$$


## Modelo Gompertz





