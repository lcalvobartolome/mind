import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ruta al archivo
file = "/export/usuarios01/ivgomez/mind/pipeline_irina/1_questions_testing/qwen27b_merged.xlsx"

# Leer el Excel
df = pd.read_excel(file)

# Quedarse solo con columnas numéricas
df_num = df.select_dtypes(include="number")

# Calcular correlación (Pearson)
corr = df_num.corr(method="pearson")

# Guardar la matriz
#corr.to_excel("/export/usuarios01/ivgomez/mind/pipeline_irina/1_questions_testing/correlation_matrix.xlsx")

# Dibujar heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=0.5
)

plt.title("Correlation Matrix")
plt.tight_layout()

plt.savefig(
    "/export/usuarios01/ivgomez/mind/pipeline_irina/1_questions_testing/correlation_matrix_final.jpg",
    dpi=300
)
plt.show()