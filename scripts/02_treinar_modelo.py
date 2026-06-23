"""
Treino do Modelo Final - Random Forest com Subconjunto de 21 Features
========================================================================
Subconjunto selecionado pela Seleção Gulosa (Sequential Forward Selection),
correspondente ao pico de acurácia da curva (passo 21, acc≈0.885 em CV 2-fold).

Como usar:
  1. Coloque os 3 CSVs na mesma pasta deste script (ou ajuste ARQUIVOS abaixo)
  2. Rode: python treino_final.py
  3. Saídas geradas:
     - modelo_final_rf.pkl          (modelo treinado)
     - label_encoder.pkl            (mapeamento classe <-> número)
     - matriz_confusao.png
     - relatorio_classificacao.txt
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.fft import rfft, rfftfreq
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings("ignore")

# === CONFIGURAÇÕES ============================================================

FS = 59            # Frequência de amostragem (Hz)
WINDOW_SIZE = 256  # Amostras por janela (~2.2 s)
STEP_SIZE = 256     # Passo (50% de overlap)
RANDOM_STATE = 42

ARQUIVOS = {
    "motor_limpo":         "../dados/motor_balanceado.csv",
    "desbalanceado": "../dados/motor_desbalanceado.csv",
}

NOMES_CLASSES = {
    "motor_limpo":         "Motor Limpo",
    "desbalanceado": "Desbalanceado",
}

CANAIS = ["x1", "y1", "z1", "x2", "y2", "z2", "x3", "y3", "z3"]

# === SUBCONJUNTO SELECIONADO (Seleção Gulosa - top 21, acc=0.885 em CV) =======

FEATURES_SELECIONADAS = [
    's2_z_dom_freq',
    's2_x_dom_freq',
    's3_y_dom_freq',
    's2_y_dom_freq',
    's2_x_std',
    's3_y_spectral_entropy',
    's1_y_spectral_centroid',
    's2_x_spectral_energy',
    's3_z_spectral_entropy',
    's3_x_crest_factor',
    's2_x_skewness',
    's2_x_spectral_entropy',
    's1_x_dom_freq',
    's2_z_std',
    's2_y_peak_to_peak',
    's2_z_spectral_energy',
    's2_z_crest_factor',
    's1_y_kurtosis',
    's1_x_std',
    's3_x_mean',
    's1_y_skewness',
]

# === EXTRAÇÃO DE FEATURES (mesma lógica da seleção gulosa) ====================

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
    freqs, mag = freqs[1:], mag[1:]   # remove componente DC
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

# === 1. CARREGAR DADOS E EXTRAIR FEATURES =====================================

print("=" * 60)
print("  TREINO FINAL — Random Forest (21 features selecionadas)")
print("=" * 60)
print("\n[1/4] Extraindo features dos sinais...")

registros = []
for classe, caminho in ARQUIVOS.items():
    df = pd.read_csv(caminho)
    dados = df[CANAIS].values
    n_janelas = 0
    for start in range(0, len(dados) - WINDOW_SIZE + 1, STEP_SIZE):
        janela = dados[start: start + WINDOW_SIZE]
        feat = extrair_features(janela, CANAIS)
        feat["classe"] = classe
        registros.append(feat)
        n_janelas += 1
    print(f"  • {NOMES_CLASSES[classe]:28s} → {n_janelas} janelas")

df_feat = pd.DataFrame(registros)

# Selecionar apenas as 21 features escolhidas
X = df_feat[FEATURES_SELECIONADAS].values.astype(np.float32)

le = LabelEncoder()
y = le.fit_transform(df_feat["classe"].values)

print(f"\n  Total de janelas: {len(df_feat)}")
print(f"  Features utilizadas: {len(FEATURES_SELECIONADAS)}")
print(f"  Classes: {list(le.classes_)}")

# === 2. DIVISÃO TREINO / TESTE ================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# === 3. TREINO DO MODELO FINAL ================================================

print("\n[2/4] Treinando Random Forest (200 árvores)...")

rf = RandomForestClassifier(
    n_estimators=300,
    criterion="gini",      # medida de impureza
    max_features="sqrt",   # √(n_features)
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
rf.fit(X_train, y_train)

# === 4. AVALIAÇÃO ==============================================================

print("\n[3/4] Avaliando o modelo...\n")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(rf, X, y, cv=cv, scoring="accuracy", n_jobs=-1)

y_pred = rf.predict(X_test)
nomes_display = [NOMES_CLASSES[c] for c in le.classes_]

acc_teste = rf.score(X_test, y_test)
relatorio = classification_report(y_test, y_pred, target_names=nomes_display)

print(f"  Acurácia (teste 20%)     : {acc_teste:.4f}")
print(f"  Cross-validation 5-fold  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print()
print(relatorio)

# === 5. MATRIZ DE CONFUSÃO =====================================================

print("[4/4] Gerando matriz de confusão e salvando artefatos...")

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6.5, 5.5))
disp = ConfusionMatrixDisplay(cm, display_labels=nomes_display)
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(
    f"Matriz de Confusão — RF (21 features)\n"
    f"Acc. teste: {acc_teste:.3f} | CV 5-fold: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}",
    fontsize=11, fontweight="bold", pad=12,
)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("matriz_confusao.png", dpi=150, bbox_inches="tight")
plt.close()

# === 6. SALVAR MODELO E RELATÓRIO ==============================================

joblib.dump(rf, "../modelos/modelo_rf.pkl")
joblib.dump(le, "../modelos/label_encoder.pkl")
joblib.dump(FEATURES_SELECIONADAS, "../modelos/features_selecionadas.pkl")

with open("relatorio_classificacao.txt", "w", encoding="utf-8") as f:
    f.write("RELATÓRIO DO MODELO FINAL — Random Forest (21 features)\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Acurácia (teste 20%)    : {acc_teste:.4f}\n")
    f.write(f"Cross-validation 5-fold : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n\n")
    f.write("Features utilizadas:\n")
    for feat in FEATURES_SELECIONADAS:
        f.write(f"  - {feat}\n")
    f.write("\n" + relatorio)

print("  ✓ modelo_final_rf.pkl")
print("  ✓ label_encoder.pkl")
print("  ✓ features_selecionadas.pkl")
print("  ✓ matriz_confusao.png")
print("  ✓ relatorio_classificacao.txt")

print("\n" + "=" * 60)
print(f"  CONCLUÍDO — Acurácia final: {acc_teste*100:.2f}%")
print("=" * 60)

# === COMO USAR O MODELO DEPOIS (exemplo) ======================================
#
# import joblib
# rf = joblib.load("modelo_final_rf.pkl")
# le = joblib.load("label_encoder.pkl")
# feats = joblib.load("features_selecionadas.pkl")
#
# # X_novo deve ser um array (n_amostras, 21) na MESMA ORDEM de `feats`
# pred = rf.predict(X_novo)
# classes_previstas = le.inverse_transform(pred)