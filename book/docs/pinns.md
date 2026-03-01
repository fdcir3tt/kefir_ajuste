# Physics Informed Neural Networks (PINNs)

## Concepto
Redes Neuronales Informadas por Modelos Físicos (Physics-Informed Neural Networks, PINNs) constituyen una metodología de modelado computacional que integra explícitamente el conocimiento físico de un sistema dentro del proceso de entrenamiento de una red neuronal. A diferencia de los enfoques puramente basados en datos, las PINNs incorporan ecuaciones diferenciales ordinarias (ODEs), parciales (PDEs) u otras restricciones físicas directamente en la función de pérdida, forzando a la red a aprender soluciones que no solo se ajusten a los datos experimentales, sino que también respeten las ecuaciones gobernantes del fenómeno de estudio.

En el contexto de este proyecto, las PINNs se emplean para modelar el crecimiento microbiano de los gránulos de kéfir de agua, donde la dinámica poblacional puede describirse mediante ecuaciones diferenciales de tipo logístico o Gompertz. Estas ecuaciones representan una aproximación física del comportamiento global del sistema, mientras que la red neuronal captura dinámicas no observadas explícitamente, tales como los efectos inducidos por el pretratamiento de ultrasonido sobre la estructura y actividad microbiana.

La motivación principal para el uso de PINNs radica en su capacidad para combinar la flexibilidad de aproximación de las redes neuronales con la interpretabilidad de los modelos biológicos clásicos. Este enfoque híbrido resulta particularmente adecuado en escenarios con datos escasos, incompletos o ruidosos —como las series de tiempo disponibles para el crecimiento de kéfir—, permitiendo inferir parámetros efectivos y dinámicas ocultas sin imponer formas funcionales rígidas.

## Función de pérdida y formulación matemática
La idea central de las PINNs se basa en la construcción de una función de pérdida compuesta, diseñada para equilibrar simultáneamente el ajuste a los datos experimentales y el cumplimiento de las ecuaciones diferenciales que gobiernan el crecimiento microbiano. Una formulación típica de la función de pérdida es
$$

 \mathcal{L}(\theta;T)=w_f\mathcal{L}_f(\theta;T_f)+w_b\mathcal{L}_b(\theta;T_b)

$$

donde $\theta$ representa los parámetros entrenables de la red, y los pesos $w_f$ y $w_b$​ controlan la contribución relativa de cada término.
El término asociado a las ecuaciones gobernantes se define como
$$
\mathcal{L}_f(\theta;T_f)=\frac{1}{|T_f|}\sum_{x\in T_f}||f(x;\partial_i\hat{u};\partial_{ij}^2\hat{u}),\lambda||^2_2
$$

donde $f(\cdot)$ representa el residuo de la ecuación diferencial que modela el crecimiento poblacional, $\hat{u}$ es la salida de la red neuronal y$\lambda$ corresponde a parámetros biológicos desconocidos que pueden ser inferidos durante el entrenamiento.
De manera análoga, el término de condiciones iniciales y de frontera se expresa como
$$
\mathcal{L}_i(\theta;T_b)=\frac{1}{|T_b|}\sum_{x\in T_b}||\mathcal{B}(\hat{u},x)||^2_2
$$
asegurando que la solución aprendida sea consistente con las condiciones experimentales del sistema, como la biomasa inicial de los gránulos de kéfir.

La eficiencia computacional de las PINNs proviene del uso de diferenciación automática para evaluar los residuos de las ecuaciones diferenciales, evitando discretizaciones explícitas y permitiendo trabajar con dominios continuos. No obstante, un desafío relevante en su implementación es la adecuada ponderación de los términos de la función de pérdida, ya que un desbalance entre el ajuste a datos y las restricciones físicas puede afectar la estabilidad y convergencia del entrenamiento.
En conjunto, este enfoque permite utilizar modelos poblacionales clásicos como guías físicas, mientras se exploran y cuantifican los efectos del pretratamiento de ultrasonido sobre la dinámica de crecimiento microbiano del kéfir de agua.
