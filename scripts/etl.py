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

# ============================================================================== #
#                               CARGA DE DATOS
# ============================================================================== #

def extract( file_path:Path,skip_rows:int | None =None)->pd.DataFrame:
    """ Recibe ubicación de archivo de datos, y los convierte en un DataFrame de pandas"""
    if file_path.suffix == ".xlsx":
        extracted_data = pd.read_excel(file_path, skiprows=skip_rows)
    else:
        raise ValueError(f"Formato no soportado: {file_path.suffix}")
    
    print(f"Datos extraídos correctamente!")
    return extracted_data

def transform( data_frame:pd.DataFrame )->pd.DataFrame:
    """ Limpia y reestructura el DataFrame """

    transformed_data = ( data_frame.drop( columns= [f"Unnamed: {k}" for k in range(4)])
                                   .melt( id_vars= 'Tiempo de Fermentacón (h)',
                                          var_name= 'tratamiento',
                                          value_name= 'concentracion(g/cm3)'
                                        ) 
                        )
    print(f"Datos transformados correctamente!")

    return transformed_data


def load( data_frame:pd.DataFrame,directory:Path ):
    """ Guarda dataframe con el nombre específicado"""
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