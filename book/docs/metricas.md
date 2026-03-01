# Métricas de evaluación para la comparación de modelos PINNs

Con el objetivo de evaluar y comparar los resultados obtenidos al resolver problemas inversos mediante distintas configuraciones de Redes Neuronales Informadas por Modelos Físicos (PINNs), se emplearon métricas estadísticas ampliamente utilizadas en el análisis de modelos. En particular, se utilizaran el Criterio de Información de Akaike (AIC), el Criterio de Información Bayesiano (BIC) y el coeficiente de determinación ($R^2$), las cuales permiten cuantificar tanto la calidad del ajuste a los datos experimentales como la complejidad del modelo inferido.

En el contexto de este proyecto, estas métricas se aplican para comparar diferentes PINNs utilizadas en la estimación de parámetros biológicos del modelo de crecimiento microbiano (por ejemplo, el modelo de Gompertz) a partir de las series de tiempo del crecimiento de gránulos de kéfir de agua sometidos a distintos pretratamientos de ultrasonido.

## Criterio de Información de Akaike (AIC)
El AIC evalúa el equilibrio entre la bondad del ajuste del modelo y su complejidad, penalizando explícitamente el número de parámetros estimados. En problemas inversos resueltos con PINNs, esta métrica resulta especialmente útil para comparar configuraciones que ajustan los parámetros del modelo de crecimiento bajo distintas arquitecturas de red, funciones de activación o estrategias de ponderación de la función de pérdida.
Un valor menor de AIC indica un modelo más parsimonioso, es decir, aquel que logra describir adecuadamente la dinámica de crecimiento microbiano del kéfir utilizando el menor número efectivo de parámetros. Por ello, las PINNs con valores de AIC más bajos se consideran preferibles desde un punto de vista estadístico.

## Criterio de Información Bayesiano (BIC)
El BIC es conceptualmente similar al AIC, pero introduce una penalización más severa por el número de parámetros, lo que lo hace particularmente adecuado en escenarios con conjuntos de datos pequeños, como las series de tiempo disponibles en este estudio. Esta métrica favorece modelos más simples y robustos, reduciendo el riesgo de sobreajuste al resolver problemas inversos.

En este trabajo, el BIC permite identificar qué configuraciones de PINNs logran capturar el crecimiento de los gránulos de kéfir de agua de manera consistente, manteniendo al mismo tiempo una estructura de modelo lo más sencilla posible. Al igual que en el caso del AIC, valores menores de BIC indican un mejor desempeño global.

## Coeficiente de determinación ($R^2$)
El coeficiente de determinación ($R^2$) mide la proporción de la variabilidad de los datos experimentales que es explicada por el modelo ajustado. Dentro del marco de las PINNs aplicadas a problemas inversos, esta métrica permite evaluar directamente la capacidad de la solución inferida para reproducir las curvas de crecimiento observadas en los gránulos de kéfir de agua.
Valores de $R^2$ cercanos a 1 indican un alto grado de concordancia entre las predicciones del modelo de crecimiento —con parámetros inferidos por la PINN— y las mediciones experimentales. No obstante, dado que esta métrica no penaliza explícitamente la complejidad del modelo, su interpretación debe complementarse con criterios de información como el AIC y el BIC.

## Comparación global de resultados
El uso conjunto de AIC, BIC y $R^2$ proporciona una base objetiva y complementaria para la comparación de los resultados obtenidos mediante distintas formulaciones de problemas inversos con PINNs. Mientras que el $R^2$ ofrece una medida directa de la calidad del ajuste a los datos, el AIC y el BIC aportan criterios adicionales para evaluar la parsimonia, eficiencia y robustez de cada modelo, facilitando la selección del enfoque más adecuado para describir el crecimiento microbiano del kéfir de agua bajo diferentes pretratamientos de ultrasonido.