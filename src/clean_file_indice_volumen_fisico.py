import pandas as pd

# ---------------------------
# 📁 ARCHIVOS
# ---------------------------

input_file = "data/raw/indice_volumen_fisico_inegi.csv"
output_file = "data/processed/indice_volumen_fisico_inegi_clean.csv"

# ---------------------------
# 📥 CARGAR CSV
# ---------------------------

df = pd.read_csv(input_file)

# ---------------------------
# 🧹 ELIMINAR COLUMNAS VACÍAS
# ---------------------------

df = df.dropna(axis=1, how="all")
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# ---------------------------
# 🔧 LIMPIEZA YEAR
# ---------------------------

df["year"] = df["year"].astype(str).str.strip()
df["year"] = df["year"].replace("", pd.NA)
df["year"] = pd.to_numeric(df["year"], errors="coerce")

# Propagar año
df["year"] = df["year"].ffill()

# ---------------------------
# 🔧 LIMPIEZA QUARTER
# ---------------------------

df["quarter"] = df["quarter"].astype(str).str.strip().str.upper()
df["quarter"] = df["quarter"].str.replace(r"\s+", "", regex=True)

# Mantener solo T1–T4
valid_quarters = ["T1", "T2", "T3", "T4"]
df = df[df["quarter"].isin(valid_quarters)]

# Convertir a número
df["quarter_num"] = df["quarter"].str.replace("T", "").astype(int)

# ---------------------------
# 🧹 LIMPIEZA FINAL
# ---------------------------

df = df[df["year"].notna()]
df["year"] = df["year"].astype(int)

# ---------------------------
# 🔢 CONVERTIR COLUMNAS NUMÉRICAS
# ---------------------------

cols_numericas = [
    "ivf_total_nacional",
    "ivf_turistico_total",
    "ivf_turistico_bienes",
    "ivf_turistico_servicios"
]

for col in cols_numericas:
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False)
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ---------------------------
# 🧹 ELIMINAR DUPLICADOS
# ---------------------------

df = df.drop_duplicates(subset=["year", "quarter"], keep="first")

# ---------------------------
# 📅 CREAR FECHA
# ---------------------------

df["date"] = pd.PeriodIndex(
    year=df["year"],
    quarter=df["quarter_num"],
    freq="Q"
).to_timestamp()

# ---------------------------
# 📊 FORMATO FINAL
# ---------------------------

df = df.sort_values("date")
df = df.set_index("date")

# Mantener columnas clave
df = df[[
    "year",
    "quarter",
    "ivf_total_nacional",
    "ivf_turistico_total",
    "ivf_turistico_bienes",
    "ivf_turistico_servicios"
]]

# ---------------------------
# ✅ VALIDACIÓN
# ---------------------------

print(df.head())
print("\nTipos de datos:\n", df.dtypes)
print("\nDuplicados en índice:", df.index.duplicated().sum())
print("\nValores nulos:\n", df.isna().sum())

# ---------------------------
# 💾 GUARDAR CSV
# ---------------------------

df.reset_index().to_csv(output_file, index=False)

print(f"\nCSV limpio guardado en: {output_file}")