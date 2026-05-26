import matplotlib.pyplot as plt
import numpy as np

models = ["deepseek-r1:8b", "gemma3:4b", "mistral:7b"]

# Accuracies
f1_en = [0.28379920189743185,0.21910925648752944, 0.3055566740802904]
f1_es = [0.2749960510767486, 0.27219926211460077, 0.3046591454644409]
f1_it = [0.3020859082770035, 0.3055547844604851, 0.32027815377103264]
 
# P-values
p_en_es = [
    0.7492586247608418,
    0.7139286071392861,
    0.962903709629037
]

p_en_it = [
    0.31050465907901537,
    0.000999900009999,
    0.5371462853714628
]

n_models = len(models)
base_x = np.arange(n_models)

bar_width = 0.18
exp_gap = 0.5  # espacio REAL entre EN–ES y EN–IT

plt.figure(figsize=(12, 5))

for i in range(n_models):
    # Centros de experimentos
    x_es = base_x[i] - exp_gap / 2
    x_it = base_x[i] + exp_gap / 2

    # --- EN–ES ---
    plt.bar(x_es - bar_width / 2, f1_en[i], bar_width,
            color="#1f77b4", label="English" if i == 0 else "")
    plt.bar(x_es + bar_width / 2, f1_es[i], bar_width,
            color="#ff7f0e", label="Other language" if i == 0 else "")

    # --- EN–IT ---
    plt.bar(x_it - bar_width / 2, f1_en[i], bar_width,
            color="#1f77b4")
    plt.bar(x_it + bar_width / 2, f1_it[i], bar_width,
            color="#ff7f0e")

    # P-values
    plt.text(x_es,
             max(f1_en[i], f1_es[i]) + 0.015,
             f"p={p_en_es[i]:.3g}",
             ha="center", va="bottom", rotation=90, fontsize=12)

    plt.text(x_it,
             max(f1_en[i], f1_it[i]) + 0.015,
             f"p={p_en_it[i]:.3g}",
             ha="center", va="bottom", rotation=90, fontsize=12)

# --- EJE X PRINCIPAL ---
plt.xticks(base_x, [])
plt.ylabel("F1-Score", fontsize=15)


# --- SUBETIQUETAS EN–ES / EN–IT ---
for i in range(n_models):
    plt.text(
        base_x[i] - exp_gap / 2,
        -0.035,
        "EN–SP",
        ha="center",
        va="top",
        fontsize=14,
        transform=plt.gca().get_xaxis_transform()
    )

    plt.text(
        base_x[i] + exp_gap / 2,
        -0.035,
        "EN–IT",
        ha="center",
        va="top",
        fontsize=14,
        transform=plt.gca().get_xaxis_transform()
    )

for i, model in enumerate(models):
    plt.text(
        base_x[i],
        -0.085,
        model,
        ha="center",
        va="top",
        fontsize=15,
        transform=plt.gca().get_xaxis_transform()
    )

# Margen eje Y
plt.ylim(0, max(f1_en + f1_es + f1_it) + 0.14)

plt.legend()
plt.tight_layout()

plt.savefig(
    "/export/usuarios01/ivgomez/mind/t_data/outputs/cm/f1score.jpg",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
