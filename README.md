<div style="display: flex; align-items: center; justify-content: space-between;">
  <img src="https://imgs.search.brave.com/cyPPm2FEbLyCvnJYMrRM0rin9lNl4ye5NJi1oOIYR2k/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly9taXJv/Lm1lZGl1bS5jb20v/djIvMSpPbmtYUEpN/bU5QbE1VRmo5S3Fl/N3BBLnBuZw" alt="mcd_logo" width="100">
  <img src="https://imgs.search.brave.com/zClzZulw4XA8_jofzD_0A05zkvIZ5WUquznMkQCGx24/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9pZGVu/dGlkYWRidWhvLnVu/aXNvbi5teC93cC1j/b250ZW50L3VwbG9h/ZHMvMjAyNC8wMy9F/U0NVRE8tQ09MT1Iu/cG5n" alt="unison_logo" width="100">
</div>

# Ajuste de curvas poblacionales de granos de Kefir utilizando PINNs


[![Python][python-shield]][python-url]
[![Markdown][md-shield]][md-url]
[![Git][git-shield]][git-url]
[![Github][github-shield]][github-url]

[![Jupyter Notebooks][jupyter-shield]][jupyter-url]
[![PyTorch][pytorch-shield]][pytorch-url]
[![Scikit-learn][sklearn-shield]][sklearn-url]


[python-shield]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[python-url]: https://www.python.org/

[md-shield]: https://img.shields.io/badge/Markdown-000?style=for-the-badge&logo=markdown
[md-url]: https://www.markdownguide.org/

[git-shield]: https://img.shields.io/badge/GIT-E44C30?style=for-the-badge&logo=git&logoColor=white
[git-url]: https://git-scm.com/

[github-shield]: https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white
[github-url]: https://github.com/

[jupyter-shield]: https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter&logoColor=white
[jupyter-url]: https://jupyter.org/




[pytorch-shield]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white
[pytorch-url]: https://pytorch.org/



[sklearn-shield]: https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white
[sklearn-url]: https://scikit-learn.org/


## Descripción del proyecto

Este proyecto tiene como objetivo ajustar curvas de crecimiento de granos de kéfir utilizando Physics-Informed Neural Networks (PINNs) junto con modelos matemáticos comúnmente empleados en este campo.

Se consideran modelos matemáticos que aparecen con frecuencia en la literatura científica relacionada con el crecimiento de granos de kéfir.

Para la implementación de las PINNs, se emplearán las bibliotecas DeepXD y NeuralUQ
## Estructura 

````
├── LICENSE # Licencia del proyecto
├── Makefile # Automatización de tareas (construcción de docs, tests, etc.)
├── config.yaml # Archivo de configuración general
├── data/ # Datos de entrada
│ └── raw/ # Datos crudos sin modificar
│ 
├── docs/ # Documentación generada con Sphinx
│ ├── src/ # Archivos fuente de la documentación
│ ├── build/ # Documentación HTML compilada
│ └── examples/ # Ejemplos ejecutables
├── figures/ # Figuras y resultados gráficos
├── models/ # Modelos entrenados (checkpoints)
├── notebooks/ # Notebooks de Jupyter
├── plots/ # Gráficos generados automáticamente
├── scripts/ # Scripts de utilidad
│ ├── graphs.py
│ └── pinn.py
├── src/ # Código fuente principal
│ ├── log_eq.py # Implementación de la ecuación logística
│ └── solve_eq.py # Solución numérica de la ecuación
├── tests/ # Pruebas unitarias con pytest
│ ├── test_log_eq.py
│ └── test_solve_eq.py
└── requirements.txt # Dependencias del entorno

```