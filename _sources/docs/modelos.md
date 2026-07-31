# Modelos de crecimiento bacteriano

## Formulación matemática


El crecimiento de comunidades microbianas, como el presente en los gránulos de kéfir de agua, puede describirse mediante modelos matemáticos de tipo poblacional {cite}`Zwietering1990`. Estos modelos permiten representar la evolución temporal de la biomasa microbiana $y(t)$  bajo ciertas hipótesis biológicas y ambientales, y han sido ampliamente utilizados en el estudio de fermentaciones y sistemas probióticos.

El modelo Malthusiano asume que la tasa de crecimiento es proporcional al tamaño poblacional{cite}`AAdkinsWilliams2012`, lo que conduce a un crecimiento exponencial. En forma matemática, partimos de las tazas de nacimiento $b(t)$ y de muertes $d(t)$ . Se propone que tanto $b(t)$ como $d(t)$ comparten una relación de proporcionalidad con el tamaño de población $y(t)$ :


```{math}
:label: rates
b(t)=\beta y(t), \quad d(t)=\delta y(t).
```

De aquí, se plantea que la taza en la que cambia el tamaño de población se da por la diferencia entre las tazas de nacimiento y muerte:

$$
\begin{equation*}
\frac{dy}{dt}=b(t)-d(t).
\end{equation*}
$$

Teniendo en mente las expresiones de {eq}`rates` , obtenemos la ecuación 

```{math}
:label: exponential
\frac{dy}{dt}=ry(t).
```

Donde $r=\beta-\delta$ es una constante real. El modelo Veltusiano también considera las mismas premisas y la forma de {eq}`exponential`, solo cambian  $b(t)$ y $d(t)$. Con este modelo se considera que el cociente $b(t)/y(t)$ en vez de mantenerse constante, se considera que este decrese linealmente con respecto a la población: $b(t)/y(t)=\beta - k_\beta y(t)$ . De manera similar con la taza de muerte. Como consecuencia, obtenemos:

$$
\begin{equation*}
b(t)=(\beta - k_{\beta}y) y ,\hspace{5mm} d(t)=(\delta + k_{\delta}y) y.
\end{equation*}
$$

Teniendo esto en mente, obtenemos

```{math}
:label: proc-logistic
\begin{align}
\frac{dy}{dt} &= (\beta - k_{\beta}y) y - (\delta + k_{\delta}y) y \\
&= (\beta - \delta - (k_{\beta}+k_{\delta})y) y \\
&= (\beta - \delta)\left(1 - \frac{k_{\beta}+k_{\delta}}{\beta - \delta} y \right) y.
\end{align}
```


Para incorporar estas restricciones, el modelo logístico introduce una capacidad de carga $m$, asociada a la disponibilidad de recursos y al entorno físico-químico del medio de cultivo.Podemos renombrar los términos de {eq}`proc-logistic` en forma compacta para por fin llegar a la ecuación logística:

```{math}
:label: verhulst_eq

\begin{equation*}
\frac{dy}{dt}=ry(1-\frac{y}{m}),
\end{equation*}

```

$$
\begin{equation*}
    r=\beta -\delta,\hspace{5mm}m=\frac{\beta -\delta}{k_{\beta}+k_{\delta}},
\end{equation*}
$$

donde $k_\beta$ y $k_\delta$ son constantes reales positivas. Este modelo describe adecuadamente la dinámica sigmoide observada en muchos procesos fermentativos, incluyendo el de granulos de kéfir {cite}`Zajek2010,Baltazar-Larios2025`, permitiendo además definir parámetros biológicos interpretables, como la rapidez máxima de crecimiento $\mu_m$ y el tiempo de retraso $\lambda$ {cite}`Zwietering1990`. 

## Ecuaciones interpretables

A continuación, veremos cómo obtener dichos parámetros de 3 modelos típicos
en la literatura de epidemología. Nos interesan las reparametrizaciones de estos modelos para comprender con mayor facilidad. A continuación se muestra cómo podemos llegar a estas reparametrizaciones interpretables.  
 
Primero, $\mu_m$,$\lambda$ se definen como "máxima taza de crecimiento específico" y "tiempo de retraso" respectivamente {cite}`Zwietering1990`, y se pueden representar estas definiciones con las expresiones:  

$$
\mu_m := \text{arg}\Bigg[\text{max}\Bigg( \frac{dy}{dt}\Bigg)\Bigg] 

$$

$$
\lambda :=\text{arg} \Bigg[ L \cap\Bigg \{(t,y = 0): t\in\mathbb{R}\Bigg\} \Bigg] ,
$$

donde $L$ es la recta tangencial a la curva en el punto de inflección.

Tomando como ejemplo el modelo de Verhulst, cuya ecuación diferencial tiene la forma de {eq}`verhulst_eq` para cálcular primero $\mu_m$, partimos de:

```{math}
:label: mu_proc

\begin{align*}
\text{max}\Big(\frac{dy}{dt}\Big)\iff \frac{d^2y}{dt^2}=0,\\
\frac{d}{dt}\Big[ry\Big(1-\frac{y}{m}\Big) \Big]= 0 \\
\frac{d}{dt}[ry]\Big(1-\frac{y}{m}\Big) + ry\frac{d}{dt}\Big(1-\frac{y}{m}\Big) = 0 \\
r\frac{dy}{dt}\Big(1-\frac{y}{m}\Big)-\frac{r}{m}y\frac{dy}{dt}= 0\\
r\frac{dy}{dt}\Bigg[1-\frac{2y}{m}\Bigg] = 0.
\end{align*}

```

De {eq}`mu_proc` obtenemos la solución
no trivial : 

$$
1-\frac{2y}{m} = 0 \to y(t_i) = \frac{m}{2}
$$

$$
\begin{aligned}
\frac{dy}{dt}\Big|_{t_i} 
&= r y(t_i) \Bigl( 1 - \frac{y(t_i)}{m} \Bigr) \\
&= r \frac{m}{2} \Bigl( 1 - \frac{1}{2} \Bigr) \\
&= r \frac{m}{4}
\end{aligned}
$$

$$
\begin{aligned}
\mu_m =\frac{dy}{dt}\Bigg|_{t_i} = r\frac{m}{4}
\end{aligned}
$$

Ahora solo nos falta conocer $t_i$ despejando de la solución explícita $P(t)$: 

$$
y(t_i) = \frac{my_0e^{rt_i}}{m+y_0(e^{rt_i}-1)} \\


\frac{m}{2} = \frac{my_0e^{rt_i}}{m+y_0(e^{rt_i}-1)} \\ 
m+y_0(e^{rt_i}-1) = 2y_0e^{rt_i}\\

m-(y_0e^{rt_i}+1)= 0 \\ 
\frac{m-1}{y_0} = e^{rt_i}\\

t_i = ln\Bigg[\Big(\frac{m-1}{ry_0}\Big)\Bigg]


$$

Una vez tenemos $\mu_m$ y el punto de inflección $t_i$, podemos cálcular $\lambda$:

$$
\frac{m}{2} = \mu_m t_i + y_0 \to y_0 =\frac{m}{2} -\mu_m t_i\\

0=\mu_m\lambda+y_0\\
\lambda = -\frac{y_0}{\mu_m}=t_i-\frac{m}{2\mu_m}\\
\lambda = ln\Bigg[\Big(\frac{m-1}{ry_0}\Big)\Bigg] - \frac{2}{r}
$$


De similar forma podemos reescribir los modelos típicos en términos de los parámetros interpretables{cite}`Zwietering1990`:


| Nombre   |          Ecuación típica            |      Ecuación con términos interpretables    |
|----------|-------------------------------------|----------------------------------------|
| Verhulst | $y(x) =\frac{a}{1+e^{(b-cx)}}$   |  $y(t)=\frac{A}{1+e^{(\frac{4\mu_m}{A}(\lambda-t)+2)}}$  |
| Gompertz | $y(x)=a e^{-e^{(b-cx)}}$ | $y(t)=Ae^{-e^{\frac{4\mu_m}{A}(\lambda-t)+1}}$  |
| Richards | $y(x)=a(1+\nu  e^{k(\tau-x)})^{-\frac{1}{\nu}}$   | $y(t)=A(1+\nu e^{(1+\nu)} \cdot e^{\frac{\mu_m}{A}\cdot(1+\nu)^{(1+\frac{1}{\nu})}\cdot(\lambda - t)})^{-\frac{1}{\nu}}$  |
                                        
## Límites de modelos 

Estos modelos suponen una respuesta simétrica alrededor del punto de máxima rapidez, lo cual no se observa en la figura {figure}``. Si bien estos modelos pueden aproximar fases iniciales de crecimiento microbiano, resultan insuficiente para describir sistemas reales como el abordado por las series de tratamiento, ya que se presentan efectos inducidos. Tanto el modelo logístico como el de Gompertz dependen de supuestos funcionales específicos y parámetros constantes como se tocó previamente, lo que limita su capacidad para representar fenómenos no lineales como el que tenemos a la mano. En el caso particular de los experimentos con tratamiento, los efectos no siempre pueden ser capturados adecuadamente por modelos clásicos con parámetros fijos o formas funcionales predefinidas.

Es por eso que, las redes neuronales informadas por modelos físicos (Physics-Informed Neural Networks, PINNs) representan una alternativa robusta para el análisis del crecimiento microbiano. PINNs permiten integrar ecuaciones diferenciales ordinarias —como las del modelo de Verhulst, Gompertz o de Richards directamente en el proceso de entrenamiento de la red, al mismo tiempo que aprenden dinámicas a partir de datos experimentales {cite}`Cuomo2022,Pappu2025`. 

De esta forma, es posible identificar parámetros efectivos dependientes del pretratamiento, modelar dinámicas ocultas y capturar desviaciones respecto a los modelos clásicos, incluso en escenarios con datos escasos $(n_{obs}<200)$, como las series de tiempo disponibles para el crecimiento de gránulos de kéfir de agua ($n_{obs}=75$).

Este enfoque híbrido combina la interpretabilidad de los modelos tradicionales con la flexibilidad de las técnicas de aprendizaje profundo, ofreciendo una herramienta adecuada para caracterizar y comparar el efecto del ultrasonido sobre el crecimiento microbiano del kéfir.

