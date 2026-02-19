import pandas as pd

data_path= f"../../data/raw/fermentacionKefirdeAguaTestigo.xlsx"

def load_data():
    """
    Carga y transforma los datos experimentales de un archivo Excel en formato largo.

    La función lee un archivo Excel especificado por la variable global `data_path`,
    omite las primeras 8 filas, elimina las columnas no nombradas (por ejemplo, "Unnamed: 0" a "Unnamed: 3")
    y reorganiza los datos en formato largo para facilitar su análisis o visualización.

    Returns
    -------
    pandas.DataFrame
        Un DataFrame en formato largo con las siguientes columnas:
        - 'Tiempo de Fermentacón (h)': tiempo de fermentación en horas.
        - 'Tratamiento': nombre del tratamiento experimental.
        - 'Concentración (g/cm3)': valor de concentración correspondiente.
    """
    data=pd.read_excel(data_path, skiprows=8)
    data=data.drop(columns=[f'Unnamed: {k}' for k in range(4)])
    data_long = pd.melt(data, id_vars='Tiempo de Fermentacón (h)', var_name='Tratamiento', value_name='Concentración (g/cm3)')
    return data_long