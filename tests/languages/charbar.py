import matplotlib.pyplot as plt
import numpy as np

models = ["deepseek-r1:8b", "gemma3:4b", "mistral:7b"]

# Accuracies
acc_en = [0.3429, 0.3013, 0.3462]
acc_es = [0.3333, 0.3333, 0.3462]
acc_it = [0.3654, 0.3622, 0.3558]

# P-values
p_en_es = [
    0.7492586247608418,
    0.19341265286193737,
    1.0
]

p_en_it = [
    0.31050465907901537,
    0.00432400493446039,
    0.7754496546815659
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
    plt.bar(x_es - bar_width / 2, acc_en[i], bar_width,
            color="#1f77b4", label="English" if i == 0 else "")
    plt.bar(x_es + bar_width / 2, acc_es[i], bar_width,
            color="#ff7f0e", label="Other language" if i == 0 else "")

    # --- EN–IT ---
    plt.bar(x_it - bar_width / 2, acc_en[i], bar_width,
            color="#1f77b4")
    plt.bar(x_it + bar_width / 2, acc_it[i], bar_width,
            color="#ff7f0e")

    # P-values
    plt.text(x_es,
             max(acc_en[i], acc_es[i]) + 0.015,
             f"p={p_en_es[i]:.3g}",
             ha="center", va="bottom", rotation=90, fontsize=12)

    plt.text(x_it,
             max(acc_en[i], acc_it[i]) + 0.015,
             f"p={p_en_it[i]:.3g}",
             ha="center", va="bottom", rotation=90, fontsize=12)

# --- EJE X PRINCIPAL ---
plt.xticks(base_x, [])
plt.ylabel("Accuracy", fontsize=15)

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

# --- NOMBRE DEL MODELO (MÁS ABAJO) ---
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
plt.ylim(0, max(acc_en + acc_es + acc_it) + 0.14)

plt.legend()
plt.tight_layout()

plt.savefig(
    "/export/usuarios01/ivgomez/mind/t_data/outputs/cm/accuracy_mcnemar_FINAL2_correct.jpg",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
