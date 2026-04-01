import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------------------
# 📁 ARCHIVOS
# ---------------------------

file_1 = "data/processed/consumo_turistico_inegi_clean.csv"
file_2 = "data/processed/indice_volumen_fisico_inegi_clean.csv"

output_dir = "plots"

# Crear carpeta si no existe
os.makedirs(output_dir, exist_ok=True)

# ---------------------------
# 📊 FUNCIÓN PARA GRAFICAR
# ---------------------------

def plot_timeseries(file_path, output_name):
    df = pd.read_csv(file_path, parse_dates=["date"])

    df = df.set_index("date")

    # Crear figura
    plt.figure()

    for col in df.columns:
        if col not in ["year", "quarter"]:
            plt.plot(df[col], label=col)

    plt.title(f"Series de tiempo - {output_name}")
    plt.xlabel("Fecha")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid()

    # Guardar imagen
    save_path = os.path.join(output_dir, f"{output_name}.png")
    plt.savefig(save_path, bbox_inches="tight")

    print(f"Figura guardada en: {save_path}")

    plt.close()

# ---------------------------
# 🚀 GENERAR GRÁFICAS
# ---------------------------

plot_timeseries(file_1, "consumo_turistico")
plot_timeseries(file_2, "ivf_turismo")