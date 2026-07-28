import pandas as pd
from pathlib import Path

data_source = Path("data/raw/fermentacionKefirdeAguatestigo.xlsx")

df = pd.read_excel(data_source,sheet_name="Repeticiones",skiprows=9)
df = df.drop(columns=["Unnamed: 0","Unnamed: 1","Unnamed: 2"])
df = df.melt(id_vars= 'Tiempo de Fermentacón (h)',
               var_name= 'tratamiento',
               value_name= 'biomasa(g)'
                                        )
df["repetición"] = df["tratamiento"].apply(lambda x:x.split("Repetición ")[1])
df["tratamiento"]= df["tratamiento"].apply(lambda x:x.split("Repetición ")[1])


df["repetición"]=df["repetición"].astype(float)
df["repetición"]=df["repetición"].astype(str)
df["repetición"]=df["repetición"].apply(lambda x:x.split('.')[0])

df["tratamiento"]=df["tratamiento"].astype(float)
df["tratamiento"]=df["tratamiento"].astype(str)
df["tratamiento"]=df["tratamiento"].apply(lambda x:x.split('.')[1])

# Nuevas columnas

treatment_dict={0:"Testigo (T1) Kéfir sin ultrasonicar",
                1:"15 seg. 20 W/cm2 (T2)",
                2:"1 min. 20 W/cm2 (T3)",
                3:"15 seg. 34 W/cm2 (T4)",
                4:"1 min. 34 W/cm2 (T5)"}

intensity_dict = {
         "Testigo (T1) Kéfir sin ultrasonicar":[0,0],
         "15 seg. 20 W/cm2 (T2)": [15,20],
         "1 min. 20 W/cm2 (T3)": [60,20],
         "15 seg. 34 W/cm2 (T4)":[15,34],
         "1 min. 34 W/cm2 (T5)":[60,34],

}
repetition_col = "repetición"
treatment_col  = "tratamiento"
intensity_col  = "intensidad(W/cm^2)"
exposure_col   = "periodo de exposición(s)"

df[treatment_col]=df[treatment_col].apply(lambda x:treatment_dict[int(x)])
df[intensity_col]=df[treatment_col].apply(lambda x:intensity_dict[x][1])
df[exposure_col]=df[treatment_col].apply(lambda x:intensity_dict[x][0])

# Guardado

df["biomasa_prom(g)"] = df.groupby(["Tiempo de Fermentacón (h)","tratamiento"])[["biomasa(g)"]].transform("mean")
df = df[["Tiempo de Fermentacón (h)","tratamiento","biomasa_prom(g)","intensidad(W/cm^2)","periodo de exposición(s)"]]
df = df.rename(columns={"biomasa_prom(g)":"biomasa(g)"}).drop_duplicates()


df.to_csv("data/processed/fixed_controlset.csv")