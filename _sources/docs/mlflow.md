# Reproducibilidad utilizando mlflow

Con el objetivo de garantizar la reproducibilidad de los resultados obtenidos en este proyecto, se optó por utilizar la herramienta [mlflow](https://mlflow.org/docs/latest/ml/) como solución de seguimiento y gestión de experimentos. La meta de reproducibilidad es que,al partir de una configuración de experimento dada, se pueda obtener consistentemente el mismo resultado.

Se eligió MLflow como la herramienta más adecuada para este caso de uso, ya que cumple con las características requeridas, entre ellas:

- **Código abierto**
- **Seguimiento de experimentos** : registra métricas, hiperparámetros, artefactos y configuraciones de entrenamiento.

- **Model Registry**: permite versionar modelos y gestionar su ciclo de vida.

- **Gestión de artefactos**: almacena modelos, gráficos, logs y otros archivos generados durante el entrenamiento.

- **Reproducibilidad**: registra información del entorno y dependencias para facilitar la repetición de experimentos.

- **Compatibilidad amplia**: funciona con PyTorch, TensorFlow, Scikit-learn, XGBoost, Hugging Face y muchas otras herramientas.

- **Despliegue flexible**: puede ejecutarse localmente o utilizando servicios de almacenamiento externos.

- **Valor currícular**: Se utiliza mucho en roles corporativos de ciencia de datos por estar integrado a la herramienta de [DataBricks](https://www.databricks.com/)

Otras herramientas que se pudieron haber utilizado son:

| Nombre         | Descripción|Características  |Página de recurso|
|----------------|------------|-----------------|------|
| **Weights&Biases** |Plataforma de MLOps enfocada en el seguimiento y análisis de experimentos de machine learning. Permite registrar automáticamente métricas, hiperparámetros, modelos, datasets y artefactos durante el entrenamiento.|- Seguimiento de experimentos<br>- Visualización avanzada<br>- Gestión de artefactos<br>- Colaboración<br>- Integraciones<br>- Hyperparameter Sweeps | [enlace](https://wandb.ai/site) |
| **ClearML**        |Plataforma de MLOps de código abierto que va más allá del seguimiento de experimentos. Además de registrar métricas y modelos, incluye capacidades de orquestación, automatización y gestión de infraestructura | -Seguimiento de experimentos <br>-Ejecución remota<br>-Gestión de datasets<br>-Pipelines automatizadas<br>-Orquestación de recursos<br>-Model Registry| [enlace](https://wandb.ai/site) |  

## Demo

Continuando con el enfoque de reproducibilidad, se registraron de manera sistemática y detallada las **métricas**, los **parámetros** y otras **configuraciones** relevantes asociadas a cada corrida de entrenamiento de los modelos PINN, con el fin de asegurar un control riguroso sobre las condiciones de entrenamiento, comparación y análisis de los distintos experimentos realizados. 

Para ilustrar cómo se lleva a cabo la comparación de experimentos en la interfaz de usuario, se presenta una breve demostración. En ella se muestran diferentes experimentos y ejecuciones realizadas con diversas configuraciones, así como las herramientas disponibles para compararlos a partir de las métricas obtenidas en cada corrida.

```{figure} /images/mlflow_demo_2.gif
:alt: mlflow Demo
:width: 1000px
:align: center
```


## Configuración de experimentos

En concordancia con el uso de la herramienta, para el seguimiento de experimentos se diseñó una estrategia de configuración orientada a garantizar la trazabilidad, organización y reproducibilidad de los modelos PINN generados. Proponemos una nomenclatura estructurada para identificar cada experimento, definida como: <ecuación diferencial>\_<función de ajuste>\_<asignación de puntos de entrenamiento>\_<objetivo de experimento>. Esta convención permite asociar de manera directa cada experimento con su propósito.

Asimismo, cada corrida de entrenamiento dentro de un experimento fue identificada mediante el formato: <etiqueta de tratamiento de ultrasonido>\_<grado de polinomio>\_<número de épocas>. Esta estructura facilita el reconocimiento preciso de las condiciones específicas bajo las cuales se ejecutó cada entrenamiento.

```python
import mlflow 
from mlflow.data.pandas_dataset import PandasDataset

ensure_experiment_active(EXPERIMENT_NAME)
mlflow.set_experiment(EXPERIMENT_NAME)

# Carga y registro de datos 
dataset = load_data(DATA_FILE_NAME)
dataset_source_url = f"data/processed/{DATA_FILE_NAME}"
mlflow_dataset: PandasDataset = mlflow.data.from_pandas(dataset, 
                                                        source=dataset_source_url,
                                                        targets="concentracion(g/cm3)",
                                                        name=DATA_FILE_NAME)
```



```python
run_name = delta.__name__
with mlflow.start_run(run_name=run_name):
    mlflow.log_input(mlflow_dataset, context=CONTEXT)
    mlflow.log_param("epochs", EPOCHS)

    # =================
    #      >Código<
    # =================


    log_run(experiment=EXPERIMENT,
            dataset=dataset,
            model=model,
            model_name=model_name,
            collocation_method = COLLOCATION_METHOD,
            loss_history=loss_history,
            learned_params=learned_parameters,
            y_true=y_true,
            y_pred=y_pred)
```



Los experimentos fueron concebidos para evaluar el desempeño del entrenamiento de redes neuronales informadas por física (PINNs), explorando distintas combinaciones de variables relevantes, tales como: funciones de ajuste, número de épocas, cantidad de puntos de entrenamiento, métodos de selección de puntos y semillas de inicialización. La integración de nomenclatura clara con el uso sistemático de mlflow nos permite un buen manejo de las condiciones experimentales.


## Registro de parámetros y métricas

Para la gestión de experimentos, se definieron de manera explícita los valores, parámetros y métricas a registrar en cada corrida, con el objetivo de asegurar un seguimiento detallado del desempeño de los modelos PINN y facilitar su análisis comparativo. En primer lugar, se registraron variables asociadas a la configuración de cada corrida, las cuales permiten contextualizar los resultados obtenidos. Luego, se registraron también los parámetros estimados por el modelo, los cuales constituyen el resultado directo del proceso de entrenamiento:


```python

print(f"Logging Model {model_name}...")
# Log modelo
mlflow.pytorch.log_model(model.net, 
                        name=model_name)

```


```python

n_params = len(learned_params)
print("Logging learned parameters...")
# Log parametros aprendidos
mlflow.log_params(params=learned_params)
    
if collocation_method:
    mlflow.log_param("collocation_method",collocation_method.__name__)
else:
    mlflow.log_param("collocation_method",collocation_method)
```






| Parámetro                           | Descripción                                                                                                                                          |
|-------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| Grado                               | Corresponde al grado del polinomio cuyos coeficientes se buscan estimar.                                                                             |
| Puntos de entrenamiento             | Indica si el esquema de colocación de los puntos de entrenamiento, lo cual impacta en la capacidad de generalización del modelo.                     |
| Semilla                             | Valor utilizado para la inicialización de procesos estocásticos, garantizando la reproducibilidad de los resultados.                                 |
| Cantidad de puntos de entrenamiento | Número total de puntos utilizados durante el entrenamiento.                                                                                          |
| Épocas                              | Número de iteraciones de entrenamiento ejecutadas                                                                                                    | 
| Coeficientes de polinomio           | Valores ajustados por el modelo que definen el polinomio de interés                                                                                  |
| Dataset                             | Conjunto de datos utilizado en cada experimento                                                                                                      |

Y para evaluar el desempeño de los modelos, se consideró un conjunto de métricas complementarias que permiten analizar tanto la precisión como la complejidad del modelo.

```python
import deepxde as dde
import matplotlib.pyplot as plt

from kefir_ajuste.utils import plot_solution

dde.utils.plot_loss_history(loss_history)
mlflow.log_figure(plt.gcf(), "loss_plot.png")
plt.close() 

figures = plot_solution(model=model, data=dataset,experiment=experiment)
for fig, name in figures:
        mlflow.log_figure(fig, f"{name}.png")
        plt.close(fig)
    

```

```python
from kefir_ajuste.utils import compute_regression_metrics

n_params = len(learned_params)
metrics = compute_regression_metrics(
        y_true=y_true,
        y_pred=y_pred,
        n_params=n_params,
    )

print("Logging metrics...")
for metric,value in metrics.items():
        mlflow.log_metric(metric, value)

```

| Métrica                               | Descripción                                                                                                                                          |Fórmula
|---------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------|
| Root Mean Square Error (RMSE)         | Mide la magnitud promedio del error cuadrático, siendo sensible a errores grandes.                                                                   |$\Bigg[\frac{1}{n} \sum_{x_i \in X}\Big(y_i - \hat{y}_i \Big)^2 \Bigg]^{-\frac{1}{2}}$ |
| Mean Absolute Error (MAE)             | Calcula el promedio de los errores absolutos, proporcionando una medida robusta del error medio.                                                     |$\frac{1}{n} \sum_{x_i \in X}\text{abs}\Big[\hat{y}_i-y_i \Big]$  |
| Mean Absolute Percentage Error (MAPE) | Expresa el error en términos relativos, facilitando su interpretación porcentual.                                                                    | $\frac{1}{n} \sum_{x_i \in X}\text{abs}\Big[\frac{\hat{y}_i-y_i}{y_i} \Big]$ | 
| Akaike Information Criterion (AIC)    | Evalúa la calidad del modelo penalizando su complejidad, favoreciendo modelos parsimoniosos                                                          |$2k - 2ln\Big( \frac{\text{RSS}}{n}\Big)$  | 
| Bayesian Information Criterion (BIC)  | Similar al AIC, pero con una penalización más estricta, útil para la selección de modelos.                                                           |$2ln\Big(\frac{\text{RSS}}{n}\Big)-kln(n)$ |
| Coeficiente de determinación ($R^2$)  | Indica la proporción de la varianza de los datos explicada por el modelo                                                                             |$1-\frac{\text{RSS}}{\sum_{x_i\in X}(y_i-\bar{y})^2}$  |
 


