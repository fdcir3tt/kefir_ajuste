import os 
import pandas as pd
from pathlib import Path

# ============================================================================== #
#                               CONFIGURACIÓN
# ============================================================================== #

DATA_PATH = Path('data')
RAW_FILE = 'fermentacionKefirdeAguaTestigo.xlsx'
ROWS_SKIPPED= 8  # Si no hay filas que saltar, configurar como 'None'
TREATMENT_DICT = {'Testigo (T1) Kéfir sin ultrasonicar':'tratamiento_1',
                  '15 seg. 20 W/cm2 (T2)':'tratamiento_2',
                  '1 min. 20 W/cm2 (T3)':'tratamiento_3',
                  '15 seg. 34 W/cm2 (T4)':'tratamiento_4',
                  '1 min. 34 W/cm2 (T5)':'tratamiento_5'}
INTENSITY_DICT = {
            'tratamiento_1': {"intensity": 0.0, "period": 0.0},
            'tratamiento_2': {"intensity": 20.0, "period": 15.0},
            'tratamiento_3': {"intensity": 20.0, "period": 60.0},
            'tratamiento_4': {"intensity": 34.0, "period": 15.0},
            'tratamiento_5': {"intensity": 34.0, "period": 60.0},
        }

# ============================================================================== #
#                               CARGA DE DATOS
# ============================================================================== #

def extract( file_path:Path,skip_rows:int | None =None)->pd.DataFrame:
    """
    Extract raw data from a file into a pandas DataFrame.

    This function reads experimental data from an input file and converts it
    into a structured pandas DataFrame. Only Excel files are currently supported.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the input data file.
    skip_rows : int or None, optional
        Number of rows to skip when reading the file. Passed directly to
        ``pandas.read_excel``.

    Returns
    -------
    pandas.DataFrame
        Extracted dataset.

    Raises
    ------
    ValueError
        If the file format is not supported.

    Notes
    -----
    Currently only ``.xlsx`` files are supported.
    """
    if file_path.suffix == ".xlsx":
        extracted_data = pd.read_excel(file_path, skiprows=skip_rows)
    else:
        raise ValueError(f"Formato no soportado: {file_path.suffix}")
    
    print(f"Datos extraídos correctamente!")
    return extracted_data

def transform( data_frame:pd.DataFrame )->pd.DataFrame:
    """
    Clean and restructure the raw dataset.

    This function reshapes the dataset from wide to long format, removes
    unnecessary columns, and adds experimental condition variables such as
    intensity and exposure period.

    Parameters
    ----------
    data_frame : pandas.DataFrame
        Raw dataset containing treatment columns and fermentation time.

    Returns
    -------
    pandas.DataFrame
        Transformed dataset in long format with additional features:
        - ``intensidad(W/cm^2)``
        - ``periodo de exposición(s)``

    Notes
    -----
    This function relies on external dictionaries:
    ``INTENSITY_DICT`` and ``TREATMENT_DICT``.
    """

    transformed_data = ( data_frame.drop( columns= [f"Unnamed: {k}" for k in range(4)])
                                   .melt( id_vars= 'Tiempo de Fermentacón (h)',
                                          var_name= 'tratamiento',
                                          value_name= 'concentracion(g/cm3)'
                                        ) 
                        )
    transformed_data["intensidad(W/cm^2)"] = transformed_data["tratamiento"].apply(lambda x:INTENSITY_DICT[TREATMENT_DICT[x]]["intensity"])
    transformed_data["periodo de exposición(s)"] = transformed_data["tratamiento"].apply(lambda x:INTENSITY_DICT[TREATMENT_DICT[x]]["period"])
    print(f"Datos transformados correctamente!")

    return transformed_data


def load( data_frame:pd.DataFrame,directory:Path ):
    """
    Save transformed dataset into structured CSV files.

    This function splits the dataset by treatment group and writes each
    subset to a separate CSV file. It also generates a combined control
    dataset.

    Parameters
    ----------
    data_frame : pandas.DataFrame
        Transformed dataset containing treatment information.
    directory : pathlib.Path
        Base directory where processed files will be stored.

    Returns
    -------
    None

    Notes
    -----
    - Files are saved under ``directory/processed``.
    - File names are determined using ``TREATMENT_DICT``.
    - A combined dataset is saved as ``control_dataset.csv``.
    """
    os.makedirs(name = directory / 'processed',
                exist_ok = True )
    
    treatments = data_frame['tratamiento'].unique()
    
    for t in treatments:
        file_name = TREATMENT_DICT[t] + '.csv'
        file_path = directory / 'processed' / file_name
        mask = data_frame['tratamiento']== t
        df = (data_frame[mask]
              .rename(columns={'Tiempo de Fermentacón (h)':'tiempo(h)'})
              .drop(columns='tratamiento')
        )
        
        df.to_csv(file_path,index=False)
        print(f"Datos '{file_name}' cargados a '{directory}' correctamente!")
    df = data_frame.rename(columns={'Tiempo de Fermentacón (h)':'tiempo(h)'})
    df.to_csv("data/processed/control_dataset.csv")
# ============================================================================== #
#                               FLUJO PRINCIPAL
# ============================================================================== #

def main():

    print("Extrayendo datos de archivo crudo...")
    file_path= DATA_PATH / 'raw' / RAW_FILE
    extracted_data = extract(file_path=file_path,
                             skip_rows=ROWS_SKIPPED)

    print("Transformando datos crudos...")
    transformed_data = transform(data_frame=extracted_data)
    
    print("Cargando datos...")
    load(data_frame=transformed_data,
         directory=DATA_PATH)

if __name__=="__main__":
    main()