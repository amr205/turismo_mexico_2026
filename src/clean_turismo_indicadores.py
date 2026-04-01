import pandas as pd

# -------------------------
# 1. Archivos
# -------------------------
input_file = "data/raw/turismo_indicadores_inegi.csv"
output_file = "data/processed/turismo_indicadores_inegi_clean.csv"

# -------------------------
# 2. Cargar datos
# -------------------------
df = pd.read_csv(input_file)

# -------------------------
# 3. Parsear columna indicador
# -------------------------
def parse_indicador(indicador):
    try:
        partes = indicador.split(">")
        
        if len(partes) < 3:
            return pd.Series([None, None, None, None])
        
        texto = partes[0].strip()
        tipo = partes[1].strip()
        movilidad = partes[2].replace(". Absoluto", "").strip()
        
        # Flujo
        flujo = "entrada" if "ingresaron" in texto else "salida"
        
        # Variable
        if "Número de visitantes" in texto:
            variable = "num_visitantes"
        elif "Gasto total" in texto:
            variable = "gasto_total"
        elif "Gasto medio" in texto:
            variable = "gasto_medio"
        else:
            variable = "otro"
        
        return pd.Series([variable, tipo, movilidad, flujo])
    
    except Exception:
        return pd.Series([None, None, None, None])

df[["variable", "tipo", "movilidad", "flujo"]] = df["indicador"].apply(parse_indicador)

# -------------------------
# 4. Eliminar columnas redundantes
# -------------------------
df = df.drop(columns=[
    "cve_entidad",
    "desc_entidad",
    "cve_municipio",
    "desc_municipio"
])

# (Opcional pero recomendado)
df["nivel"] = "nacional"

# -------------------------
# 5. Identificar columnas de tiempo
# -------------------------
time_cols = [col for col in df.columns if "/m" in col]

# -------------------------
# 6. Convertir a formato largo
# -------------------------
df_long = df.melt(
    id_vars=[
        "variable", "tipo", "movilidad", "flujo",
        "unidad_medida", "nivel"
    ],
    value_vars=time_cols,
    var_name="fecha",
    value_name="valor"
)

# -------------------------
# 7. Limpiar fecha
# -------------------------
df_long["fecha"] = pd.to_datetime(
    df_long["fecha"].str.replace("/m", "-"),
    format="%Y-%m"
)

# -------------------------
# 8. Convertir valor a numérico
# -------------------------
df_long["valor"] = pd.to_numeric(df_long["valor"], errors="coerce")

# -------------------------
# 9. Eliminar filas inválidas
# -------------------------
df_long = df_long.dropna(subset=["valor"])

# -------------------------
# 10. Ordenar datos
# -------------------------
df_long = df_long.sort_values(["variable", "tipo", "fecha"])

# -------------------------
# 11. Guardar resultado
# -------------------------
df_long.to_csv(output_file, index=False)

print(f"Archivo limpio guardado en: {output_file}")