"""
Avaliação Visual do Modelo Final - ROC, Matriz de Confusão e Importância das Features
=========================================================================================
Requer que `treino_final.py` já tenha sido executado na mesma pasta (gera os .pkl).

Saídas:
  - curva_roc.png
  - matriz_confusao.png       (versão com contagens e percentuais)
  - importancia_features.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.fft import rfft, rfftfreq
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                              roc_curve, auc)
from sklearn.preprocessing import label_binarize
import joblib
import warnings
warnings.filterwarnings("ignore")

# === CONFIGURAÇÕES (iguais ao treino_final.py) ================================

FS = 59
WINDOW_SIZE = 128
STEP_SIZE = 64
RANDOM_STATE = 42

ARQUIVOS = {
    "motor_limpo":         "../dados/motor_balanceado.csv",
    "desbaceado_parafuso": "../dados/motor_desbalanceado.csv",
    "parafuso_aruela":     "coleta-parauso-aruela.csv",
}

NOMES_CLASSES = {
    "motor_limpo":         "Motor Limpo",
    "desbaceado_parafuso": "Desbalanceado (1 parafuso)",
    "parafuso_aruela":     "Parafuso c/ Aruela",
}

CANAIS = ["x1", "y1", "z1", "x2", "y2", "z2", "x3", "y3", "z3"]

CORES_CLASSES = ["#2196F3", "#FF9800", "#4CAF50"]

# === EXTRAÇÃO DE FEATURES (mesma lógica do treino) ============================

def features_tempo(s):
    rms = np.sqrt(np.mean(s ** 2))
    return {
        "mean": np.mean(s), "std": np.std(s), "rms": rms,
        "variance": np.var(s), "kurtosis": stats.kurtosis(s),
        "skewness": stats.skew(s), "peak_to_peak": np.ptp(s),
        "crest_factor": np.max(np.abs(s)) / (rms + 1e-10),
    }

def features_frequencia(s, fs=FS):
    N = len(s)
    freqs = rfftfreq(N, 1. / fs)
    mag = np.abs(rfft(s)) / N
    freqs, mag = freqs[1:], mag[1:]
    e = np.sum(mag ** 2) + 1e-10
    en = mag ** 2 / e
    return {
        "dom_freq": freqs[np.argmax(mag)],
        "spectral_centroid": np.sum(freqs * mag) / (np.sum(mag) + 1e-10),
        "spectral_energy": np.sum(mag ** 2),
        "spectral_entropy": -np.sum(en * np.log(en + 1e-10)),
    }

def extrair_features(janela, canais):
    feat = {}
    for i, canal in enumerate(canais):
        s = janela[:, i]
        prefixo = f"s{canal[-1]}_{canal[:-1]}"
        for k, v in features_tempo(s).items():
            feat[f"{prefixo}_{k}"] = v
        for k, v in features_frequencia(s).items():
            feat[f"{prefixo}_{k}"] = v
    return feat

# === 1. CARREGAR MODELO E OBJETOS AUXILIARES ==================================

print("Carregando modelo treinado...")
rf   = joblib.load("../modelos/modelo_rf.pkl")
le   = joblib.load("../modelos/label_encoder.pkl")
feats_selecionadas = joblib.load("../modelos/features_selecionadas.pkl")

# === 2. RECONSTRUIR O CONJUNTO DE TESTE (mesma divisão do treino) =============

print("Recriando o conjunto de teste (mesmo split do treino_final.py)...")

registros = []
for classe, caminho in ARQUIVOS.items():
    df = pd.read_csv(caminho)
    dados = df[CANAIS].values
    for start in range(0, len(dados) - WINDOW_SIZE + 1, STEP_SIZE):
        janela = dados[start: start + WINDOW_SIZE]
        feat = extrair_features(janela, CANAIS)
        feat["classe"] = classe
        registros.append(feat)

df_feat = pd.DataFrame(registros)
X = df_feat[feats_selecionadas].values.astype(np.float32)
y = le.transform(df_feat["classe"].values)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

nomes_display = [NOMES_CLASSES[c] for c in le.classes_]
n_classes = len(le.classes_)

y_pred  = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)

# ════════════════════════════════════════════════════════════════════════════
# 1) MATRIZ DE CONFUSÃO (contagens + percentuais)
# ════════════════════════════════════════════════════════════════════════════

print("Gerando matriz de confusão...")

cm = confusion_matrix(y_test, y_pred)
cm_norm = confusion_matrix(y_test, y_pred, normalize="true")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

disp1 = ConfusionMatrixDisplay(cm, display_labels=nomes_display)
disp1.plot(ax=axes[0], cmap="Blues", colorbar=False, values_format="d")
axes[0].set_title("Matriz de Confusão — Contagens", fontsize=12, fontweight="bold")
axes[0].tick_params(axis="x", rotation=20)

disp2 = ConfusionMatrixDisplay(cm_norm, display_labels=nomes_display)
disp2.plot(ax=axes[1], cmap="Blues", colorbar=False, values_format=".1%")
axes[1].set_title("Matriz de Confusão — Normalizada (%)", fontsize=12, fontweight="bold")
axes[1].tick_params(axis="x", rotation=20)

fig.suptitle(
    f"Random Forest — {len(feats_selecionadas)} features | "
    f"Acurácia: {(y_pred == y_test).mean():.3f}",
    fontsize=13, fontweight="bold", y=1.02
)
plt.tight_layout()
plt.savefig("matriz_confusao.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ matriz_confusao.png")

# ════════════════════════════════════════════════════════════════════════════
# 2) CURVA ROC (One-vs-Rest, multiclasse)
# ════════════════════════════════════════════════════════════════════════════

print("Gerando curva ROC (One-vs-Rest)...")

y_test_bin = label_binarize(y_test, classes=range(n_classes))

fpr, tpr, roc_auc = {}, {}, {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Micro-average (agrega todas as classes)
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Macro-average
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= n_classes
fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

fig, ax = plt.subplots(figsize=(7, 6.5))

for i in range(n_classes):
    ax.plot(fpr[i], tpr[i], color=CORES_CLASSES[i % len(CORES_CLASSES)],
            linewidth=2.2,
            label=f"{nomes_display[i]} (AUC = {roc_auc[i]:.3f})")

ax.plot(fpr["macro"], tpr["macro"], color="#6A1B9A", linestyle="--",
        linewidth=2.4, label=f"Macro-média (AUC = {roc_auc['macro']:.3f})")

ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1.5,
        label="Aleatório (AUC = 0.500)")

ax.set_xlabel("Taxa de Falsos Positivos (FPR)", fontsize=11)
ax.set_ylabel("Taxa de Verdadeiros Positivos (TPR)", fontsize=11)
ax.set_title("Curva ROC — One-vs-Rest (Multiclasse)", fontsize=13, fontweight="bold")
ax.set_xlim(-0.01, 1.01)
ax.set_ylim(-0.01, 1.02)
ax.legend(fontsize=9, loc="lower right")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("curva_roc.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ curva_roc.png")

# ════════════════════════════════════════════════════════════════════════════
# 3) IMPORTÂNCIA DAS FEATURES
# ════════════════════════════════════════════════════════════════════════════

print("Gerando gráfico de importância das features...")

importancias = rf.feature_importances_
ordem = np.argsort(importancias)[::-1]
feats_ordenadas = [feats_selecionadas[i] for i in ordem]
imp_ordenadas = importancias[ordem]

# Erro estimado (desvio entre as árvores)
std_imp = np.std(
    [tree.feature_importances_ for tree in rf.estimators_], axis=0
)[ordem]

# Cor por sensor (s1, s2, s3 -> primeiro caractere após "s")
CORES_SENSOR = {"1": "#2196F3", "2": "#FF9800", "3": "#4CAF50"}
cores_barras = [CORES_SENSOR.get(f[1], "#9E9E9E") for f in feats_ordenadas]

fig, ax = plt.subplots(figsize=(9, 0.4 * len(feats_ordenadas) + 1.5))

y_pos = np.arange(len(feats_ordenadas))
ax.barh(y_pos, imp_ordenadas[::-1], xerr=std_imp[::-1],
        color=cores_barras[::-1], alpha=0.85, edgecolor="white",
        error_kw=dict(elinewidth=1, ecolor="gray", capsize=2))

ax.set_yticks(y_pos)
ax.set_yticklabels([f.replace("_", " ") for f in feats_ordenadas[::-1]], fontsize=9)
ax.set_xlabel("Importância (Gini)", fontsize=11)
ax.set_title(f"Importância das Features — Random Forest ({len(feats_selecionadas)} features)",
              fontsize=13, fontweight="bold")
ax.grid(axis="x", alpha=0.3)

from matplotlib.patches import Patch
legenda = [Patch(facecolor=CORES_SENSOR[s], label=f"Sensor {s}") for s in ["1", "2", "3"]]
ax.legend(handles=legenda, fontsize=10, loc="lower right")

plt.tight_layout()
plt.savefig("importancia_features.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ importancia_features.png")

print("\nConcluído! 3 figuras geradas.")
