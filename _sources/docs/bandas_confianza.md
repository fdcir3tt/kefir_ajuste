
Esta componente, básicamente define las distribuciones que asumimos acerca de las variables relevantes al problema:

- Los parámetros del modelo;   $\theta = \{W_k,b_k\}\to$ $\theta$ ~ $\mathcal{N}(0,0.5)$ 
  
- Los parámetros físicos ; $\lambda=\{ c_j \} \to$  $log(c_j)$ ~ $\mathcal{N}(\bar{c_j},0.5)$
  
- El ruido, $\sigma$ ~ $\text{HalfNormal}(1)$
  
- Los datos, $y_i$ ~ $\mathcal{N}(\hat{u_i},\sigma)$

Con esto, podemos construir la función de probabilidad $\pi(x)$, donde el estado  se define como $x\in\{ (t,I,T)|  0\geq t \leq 168 , I \in \mathbb{R},T \in \mathbb{R}\}$.  Construimos $\pi(x)$ con inferencia Bayesiana: 

$$\begin{equation*}
 \pi(x) =  P(\Theta|\mathcal{D})\propto P(\mathcal{D}|\Theta)P(\Theta)
\end{equation*} 
$$
Donde 
- $P(\mathcal{D}|\Theta)$ Es la **verosimilitud** de los datos
- $P(\Theta)$ Es la distribución a **priori** de los parámetros
Entonces, tenemos finalmente:
$$
P(X|\theta)=\prod_{y_i\in X} f(y_i|\theta) ;\hspace{5mm} f(x_i|\theta)=\frac{1}{\sqrt{2\pi\sigma}}e^{-\frac{(y_i-\hat{y_i})^2}{2\sigma^2}} 
$$
$$
P(R|\theta)=\prod_{r_i\in R} f(r_i|\theta) ;\hspace{5mm} f(r_i|\theta)=\frac{1}{\sqrt{2\pi\sigma}}e^{-\frac{r_i^2}{2\sigma^2}} 
$$
siendo que $r_i = \frac{d\hat{y_i}}{dt}-\mathcal{F}(x_i,\hat{y_i})$, también:
$$P(\theta)=\prod_{\theta_i\in\theta} f(\theta_i) ;\hspace{5mm} f(\theta_i)=\frac{1}{\sqrt{2\pi\sigma_\theta}}e^{\frac{-(\mu_\theta-\theta_i)^2}{2\sigma^2}} $$
$$P(\lambda)=\prod_{\lambda_i\in\lambda} f(\lambda_i) ;\hspace{5mm} f(\lambda_i)=\frac{1}{\lambda_i\sqrt{2\pi\sigma_\lambda}}e^{\frac{-(\mu_\lambda-log(\lambda_i))^2}{2\sigma_\lambda^2}} $$


```{figure} /images/mcmc.png
:width: 72%
```
```{figure} /images/posterior_distributions.png
:width: 72%
```