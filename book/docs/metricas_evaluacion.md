## Métricas de Evaluación para la Comparación de Problemas Inversos con PINNs

Con el objetivo de **comparar y evaluar los resultados obtenidos a partir de los problemas inversos resueltos mediante distintas PINNs**, se emplearon métricas estadísticas ampliamente utilizadas en el análisis y selección de modelos: el **Criterio de Información de Akaike (AIC)**, el **Criterio de Información Bayesiano (BIC)** y el **coeficiente de determinación \($R^2$\)**. Estas métricas permiten cuantificar tanto la calidad del ajuste como la complejidad del modelo, proporcionando una base objetiva para la comparación entre enfoques.

### Criterio de Información de Akaike (AIC)

El **AIC** evalúa el compromiso entre la bondad del ajuste del modelo y su complejidad, penalizando el número de parámetros estimados. En el contexto de los problemas inversos abordados con PINNs, el AIC resulta particularmente útil para comparar modelos que ajustan los parámetros del modelo de Gompertz bajo distintas configuraciones de red o estrategias de entrenamiento.



Un valor menor de AIC indica un modelo más parsimonioso, es decir, un mejor equilibrio entre precisión y complejidad. Por lo tanto, al comparar distintas PINNs, aquellas con valores de AIC más bajos se consideran preferibles desde un punto de vista estadístico.

### Criterio de Información Bayesiano (BIC)

El **BIC** es similar al AIC, pero introduce una penalización más severa por el número de parámetros, especialmente cuando el tamaño del conjunto de datos es grande. Esta métrica favorece modelos más simples y es particularmente adecuada cuando se busca evitar el sobreajuste en los problemas inversos.

En este estudio, el BIC permitió identificar qué configuraciones de PINNs logran describir adecuadamente el crecimiento de los gránulos de kéfir de agua utilizando el modelo de Gompertz, manteniendo al mismo tiempo una estructura lo más simple posible. Al igual que el AIC, valores menores de BIC indican un mejor modelo.

### Coeficiente de Determinación ($R^2$)

El **coeficiente de determinación \($R^2$\)** mide la proporción de la variabilidad de los datos experimentales que es explicada por el modelo ajustado. En el marco de las PINNs aplicadas a problemas inversos, esta métrica permite evaluar directamente la capacidad del modelo entrenado para reproducir los datos de crecimiento observados.

Valores de \($R^2$\) cercanos a 1 indican un excelente ajuste entre las predicciones del modelo de Gompertz —con parámetros inferidos por la PINN— y los datos experimentales. A diferencia de AIC y BIC, el \($R^2$\) no penaliza explícitamente la complejidad del modelo, por lo que su interpretación se complementa con las métricas de información.

### Comparación Global de Resultados

La utilización conjunta de **AIC, BIC y \(R^2\)** permitirá realizar una comparación medible entre los resultados obtenidos con cada modelo de crecimiento que escojamos. Mientras que el \(R^2\) proporciona una medida directa de la calidad del ajuste, el AIC y el BIC aportaron criterios adicionales para evaluar la eficiencia y robustez de cada modelo.