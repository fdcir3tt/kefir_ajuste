# Physics Informed Neural Networks (PINNs)

## Concepto
Las Redes Neuronales informadas por modelos físicos (PINNs) constituyen una metodología de modelado computacional que integra explícitamente las leyes físicas en el proceso de entrenamiento de una red neuronal. A diferencia de los enfoques puramente centrados en datos, las PINNs incorporan ecuaciones diferenciales ordinarias (ODEs), parciales (PDEs) u otras restricciones físicas como parte del mecanismo de optimización. Esto permite que la red aprenda una solución que no solo se ajuste a los datos disponibles, sino que también satisfaga, en el sentido de residuos mínimos, las ecuaciones gobernantes del sistema.

La motivación detrás de las PINNs es aprovechar la capacidad de aproximación universal de las redes neuronales mientras se restringe su espacio de soluciones mediante las leyes físicas conocidas. De esta manera, el modelo actúa como un solucionador continuo que puede operar incluso con datos escasos, incompletos o ruidosos, garantizando mayor estabilidad y generalización que los métodos puramente empíricos.

## Funciones de perdida

La idea central matemática de las PINNs se encuentra en la formulación de una función de pérdida compuesta, diseñada para equilibrar simultáneamente:

- El ajuste a datos observados, cuando estos existen.

- El cumplimiento de las ecuaciones diferenciales gobernantes, expresado como el residuo de la PDE.

- El respeto de condiciones iniciales y de frontera, necesarias para garantizar la unicidad y estabilidad de la solución.

Una formulación típica de la pérdida es:

$$

 \mathcal{L}(\theta;T)=w_f\mathcal{L}_f(\theta;T_f)+w_b\mathcal{L}_b(\theta;T_b)

$$

donde:

$$
\mathcal{L}_f(\theta;T_f)=\frac{1}{|T_f|}\sum_{x\in T_f}||f(x;\partial_i\hat{u};\partial_{ij}^2\hat{u}),\lambda||^2_2
$$

$$
\mathcal{L}_i(\theta;T_b)=\frac{1}{|T_b|}\sum_{x\in T_b}||\mathcal{B}(\hat{u},x)||^2_2
$$


La eficiencia de las PINNs se debe a que los residuos de las PDEs se calculan mediante diferenciación automática, lo que evita discretizaciones explícitas y facilita trabajar en dominios de alta dimensionalidad. No obstante, un desafío activo de investigación es la correcta ponderación de los términos de la pérdida, pues el entrenamiento puede volverse rígido o inestable si una de las componentes domina a las demás.

## Problemas inversos

Las PINNs han demostrado ser particularmente eficaces en la formulación de problemas inversos, donde el objetivo no es únicamente resolver por la variable de estado, sino estimar parámetros desconocidos dentro de la ecuación diferencial. En estos escenarios, parámetros físicos tales como coeficientes de difusión, viscosidades, términos de reacción o condiciones internas del dominio se tratan como variables adicionales a optimizar.

El marco general consiste en parametrizar la solución y los parámetros desconocidos mediante la red neuronal,aplicar las ecuaciones gobernantes para que la pérdida física dependa de dichos parámetros y finalmente entrenar la red de forma que la solución reconstruida y la estructura física sean compatibles con los datos.

Este enfoque resulta atractivo porque permite manejar datos escasos o incompletos sin requerir mallas finas o simulaciones tradicionales. También se pueden inferir parámetros difíciles de medir en experimentos (por ejemplo, permeabilidad, módulo elástico o coeficientes cinéticos). La diferencia entre resolviendo problemas directos a los inversos es la función de perdida:

$$

 \mathcal{L}(\theta,\lambda;T)=w_f\mathcal{L}_f(\theta,\lambda;T_f)+w_b\mathcal{L}_b(\theta,\lambda;T_b)+w_i\mathcal{L}_i(\theta,\lambda;T_i)

$$

Donde:

$$
\mathcal{L}_i(\theta,\lambda;T_i)=\frac{1}{|T_i|}\sum_{x\in T_i}||\mathcal{I}(\hat{u},x)||^2_2
$$

En resumen, las PINNs proporcionan un marco unificado para resolver simultáneamente problemas directos e inversos, integrando datos experimentales y conocimiento físico de manera coherente y eficiente. En nuestro caso, nos interesa resolver el problema inverso presentado.