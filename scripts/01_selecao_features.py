"""
Seleção Gulosa de Features (Sequential Forward Selection)
==========================================================
Algoritmo:
  1. Começa com conjunto vazio
  2. A cada passo, testa adicionar cada feature restante ao conjunto atual
  3. Adiciona a que produz maior acurácia em CV
  4. Repete até incluir todas as 108 features
  5. Salva a curva completa para o usuário escolher o subconjunto ideal
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from scipy.fft import rfft, rfftfreq
import joblib, time, warnings
warnings.filterwarnings("ignore")

# FS = frequência de amostragem; WINDOW_SIZE/STEP_SIZE definem o janelamento dos sinais brutos
FS = 59; WINDOW_SIZE = 128; STEP_SIZE = 64; RANDOM_STATE = 42
CANAIS = ["x1","y1","z1","x2","y2","z2","x3","y3","z3"]
# RF_BUSCA: floresta pequena/rápida, só para comparar features durante a busca gulosa
RF_BUSCA = dict(n_estimators=10, max_depth=8,  random_state=RANDOM_STATE)
# RF_FINAL: floresta completa, usada só para validar o subconjunto ótimo ao final
RF_FINAL = dict(n_estimators=200, max_depth=None, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=1)
CV_BUSCA = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
CV_FINAL = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

ARQUIVOS = {
    "motor_limpo":         "motor-limpo.csv",
    "desbaceado_parafuso": "desbacenceado-um-parafuso.csv",
    "parafuso_aruela":     "coleta-parauso-aruela.csv",
}
NOMES_CLASSES = {
    "motor_limpo":         "Motor Limpo",
    "desbaceado_parafuso": "Desbalanceado (1 parafuso)",
    "parafuso_aruela":     "Parafuso c/ Aruela",
}

# 8 features estatísticas no domínio do tempo (energia, dispersão e forma do sinal)
def features_tempo(s):
    rms = np.sqrt(np.mean(s**2))
    return {"mean":np.mean(s),"std":np.std(s),"rms":rms,"variance":np.var(s),
            "kurtosis":stats.kurtosis(s),"skewness":stats.skew(s),
            "peak_to_peak":np.ptp(s),"crest_factor":np.max(np.abs(s))/(rms+1e-10)}

# 4 features no domínio da frequência, extraídas do espectro FFT da janela
def features_frequencia(s, fs=FS):
    N=len(s); freqs=rfftfreq(N,1./fs); mag=np.abs(rfft(s))/N
    freqs,mag=freqs[1:],mag[1:]   # remove o bin de frequência 0 (componente DC)
    e=np.sum(mag**2)+1e-10; en=mag**2/e   # energia total e energia normalizada por bin (p/ entropia)
    return {"dom_freq":freqs[np.argmax(mag)],
            "spectral_centroid":np.sum(freqs*mag)/(np.sum(mag)+1e-10),
            "spectral_energy":np.sum(mag**2),
            "spectral_entropy":-np.sum(en*np.log(en+1e-10))}

# Aplica as 12 features (8 tempo + 4 frequência) a cada um dos 9 canais (3 sensores x 3 eixos)
# da janela, gerando 108 features no total. Prefixo "sN_eixo" identifica sensor e eixo.
def extrair_features(janela, canais):
    feat={}
    for i,canal in enumerate(canais):
        s=janela[:,i]; p=f"s{canal[-1]}_{canal[:-1]}"
        for k,v in features_tempo(s).items(): feat[f"{p}_{k}"]=v
        for k,v in features_frequencia(s).items(): feat[f"{p}_{k}"]=v
    return feat

print("="*65)
print("  SELEÇÃO GULOSA DE FEATURES — Sequential Forward Selection")
print("="*65)
print("\n[1/4] Extraindo features...")

# Fatia cada CSV em janelas deslizantes de WINDOW_SIZE amostras (passo STEP_SIZE)
# e extrai as 108 features de cada janela, rotulando-a com a classe do arquivo.
registros=[]
for classe, caminho in ARQUIVOS.items():
    df=pd.read_csv(caminho); dados=df[CANAIS].values
    for start in range(0,len(dados)-WINDOW_SIZE+1,STEP_SIZE):
        j=extrair_features(dados[start:start+WINDOW_SIZE],CANAIS); j["classe"]=classe
        registros.append(j)

df_feat=pd.DataFrame(registros)
le=LabelEncoder()
X=df_feat.drop(columns=["classe"]).values.astype(np.float32)
y=le.fit_transform(df_feat["classe"].values)
feature_names=df_feat.drop(columns=["classe"]).columns.tolist()
N_FEAT=X.shape[1]
print(f"  {len(df_feat)} janelas | {N_FEAT} features | {len(le.classes_)} classes\n")

print(f"[2/4] Seleção Gulosa (RF={RF_BUSCA['n_estimators']} árvores, CV={CV_BUSCA.n_splits}-fold)")
print(f"      {N_FEAT*(N_FEAT+1)//2} avaliações — aguarde...\n")

rf_busca=RandomForestClassifier(**RF_BUSCA)
selecionadas_idx=[]; restantes_idx=list(range(N_FEAT)); resultados=[]
t_inicio=time.time()

# A cada passo, testa cada feature restante junto com as já selecionadas e
# mantém apenas a que melhorar mais a acurácia média em CV (busca gulosa).
for passo in range(1, N_FEAT+1):
    melhor_acc=-1; melhor_std=0; melhor_feat=None
    for fi in restantes_idx:
        conjunto=selecionadas_idx+[fi]
        scores=cross_val_score(rf_busca,X[:,conjunto],y,cv=CV_BUSCA,scoring="accuracy")
        if scores.mean()>melhor_acc:
            melhor_acc=scores.mean(); melhor_std=scores.std(); melhor_feat=fi
    selecionadas_idx.append(melhor_feat); restantes_idx.remove(melhor_feat)
    delta=melhor_acc-(resultados[-1]["acc_media"] if resultados else 0)
    resultados.append({"passo":passo,"feature":feature_names[melhor_feat],
                        "sensor":feature_names[melhor_feat][1],
                        "eixo":feature_names[melhor_feat][3],
                        "tipo":"_".join(feature_names[melhor_feat].split("_")[2:]),
                        "acc_media":melhor_acc,"acc_std":melhor_std,"delta_acc":delta})
    decorrido=time.time()-t_inicio
    eta=decorrido/passo*(N_FEAT-passo) if passo>1 else 0
    print(f"  [{passo:3d}/{N_FEAT}] +{feature_names[melhor_feat]:35s}  CV:{melhor_acc:.4f}±{melhor_std:.4f}  Δ:{delta:+.4f}  ETA:{eta:.0f}s")
    if passo%10==0 or passo==N_FEAT:
        pd.DataFrame(resultados).to_csv("selecao_gulosa_resultado.csv",index=False)

df_res=pd.DataFrame(resultados)
t_total=time.time()-t_inicio
print(f"\n  Concluído em {t_total:.0f}s ({t_total/60:.1f} min)")

# Três critérios para escolher quantas features usar:
# - idx_max: passo com a maior acurácia (pode incluir features quase irrelevantes)
# - idx_elbow: primeiro passo que já atinge 99% da acurácia máxima (bom equilíbrio)
# - idx_diminishing: primeiro passo cujo ganho marginal cai abaixo de 0.2%
accs=df_res["acc_media"].values
idx_max=int(np.argmax(accs))
idx_elbow=next(i for i,a in enumerate(accs) if a>=accs[idx_max]*0.99)
deltas_arr=df_res["delta_acc"].values
idx_diminishing=next((i for i in range(1,len(deltas_arr)) if abs(deltas_arr[i])<0.002),idx_max)

print(f"\n  Ponto máximo:          passo {idx_max+1:3d} | acc={accs[idx_max]:.4f}")
print(f"  Elbow (99% do máx):    passo {idx_elbow+1:3d} | acc={accs[idx_elbow]:.4f}")
print(f"  Retorno marginal<0.2%: passo {idx_diminishing+1:3d} | acc={accs[idx_diminishing]:.4f}")

print(f"\n[3/4] Validando subconjunto elbow ({idx_elbow+1} features) com RF completo...")
subset_idx=selecionadas_idx[:idx_elbow+1]
subset_features=[feature_names[i] for i in subset_idx]
X_sub=X[:,subset_idx]
rf_final=RandomForestClassifier(**RF_FINAL)
cv_scores_final=cross_val_score(rf_final,X_sub,y,cv=CV_FINAL,scoring="accuracy")
print(f"  CV 5-fold: {cv_scores_final.mean():.4f} ± {cv_scores_final.std():.4f}")
X_tr,X_te,y_tr,y_te=train_test_split(X_sub,y,test_size=0.2,stratify=y,random_state=RANDOM_STATE)
rf_final.fit(X_tr,y_tr)
nomes_display=[NOMES_CLASSES[c] for c in le.classes_]
print(f"  Acurácia teste: {rf_final.score(X_te,y_te):.4f}\n")
print(classification_report(y_te,rf_final.predict(X_te),target_names=nomes_display))

print("[4/4] Gerando visualizações...")
CORES={"1":"#2196F3","2":"#FF9800","3":"#4CAF50"}
TIPOS_FEAT={"mean":"Média","std":"Desvio Padrão","rms":"RMS","variance":"Variância",
            "kurtosis":"Curtose","skewness":"Assimetria","peak_to_peak":"Pico a Pico",
            "crest_factor":"Fator de Crista","dom_freq":"Freq. Dominante",
            "spectral_centroid":"Centróide Esp.","spectral_energy":"Energia Esp.",
            "spectral_entropy":"Entropia Esp."}

passos=df_res["passo"].values; accs_m=df_res["acc_media"].values; accs_s=df_res["acc_std"].values

fig=plt.figure(figsize=(18,16))
gs=gridspec.GridSpec(3,2,figure=fig,hspace=0.45,wspace=0.35,height_ratios=[2,1.2,1.2])

# A) Curva principal
ax_acc=fig.add_subplot(gs[0,:])
ax_acc.fill_between(passos,accs_m-accs_s,accs_m+accs_s,alpha=0.2,color="#2196F3",label="±1 std")
ax_acc.plot(passos,accs_m,color="#1565C0",linewidth=2,label="Acurácia média (CV 3-fold)")
for idx,label,cor,mk,sz in [(idx_elbow,"⬆ Elbow\n(99% do máx)","#E53935","^",120),
                              (idx_max,"★ Máximo","#6A1B9A","*",160),
                              (idx_diminishing,"▲ Retorno\nmarginal <0.2%","#2E7D32","D",80)]:
    ax_acc.scatter(idx+1,accs_m[idx],color=cor,s=sz,zorder=5,marker=mk)
    ax_acc.annotate(label,xy=(idx+1,accs_m[idx]),xytext=(idx+4,accs_m[idx]-0.025),
                    fontsize=9,color=cor,fontweight="bold",
                    arrowprops=dict(arrowstyle="->",color=cor,lw=1.2))
ax_acc.axvline(idx_elbow+1,color="#E53935",linestyle="--",alpha=0.5,linewidth=1.2)
ax_acc.set_xlabel("Número de features selecionadas",fontsize=11)
ax_acc.set_ylabel("Acurácia (CV 3-fold)",fontsize=11)
ax_acc.set_title("Curva de Acurácia — Seleção Gulosa Progressiva",fontsize=13,fontweight="bold")
ax_acc.set_xlim(0,N_FEAT+2); ax_acc.grid(alpha=0.35); ax_acc.legend(fontsize=10,loc="lower right")

# B) Ganho marginal
ax_delta=fig.add_subplot(gs[1,:])
cores_delta=["#4CAF50" if d>0 else "#E53935" for d in deltas_arr]
ax_delta.bar(passos,deltas_arr,color=cores_delta,alpha=0.75,edgecolor="white",linewidth=0.3)
ax_delta.axhline(0,color="black",linewidth=0.8)
ax_delta.axhline(0.002,color="#FF9800",linestyle=":",linewidth=1.2,label="Limiar 0.2%")
ax_delta.axhline(-0.002,color="#FF9800",linestyle=":",linewidth=1.2)
ax_delta.axvline(idx_elbow+1,color="#E53935",linestyle="--",alpha=0.5,linewidth=1.2)
ax_delta.set_xlabel("Passo",fontsize=11); ax_delta.set_ylabel("Δ Acurácia",fontsize=11)
ax_delta.set_title("Ganho Marginal por Passo",fontsize=12,fontweight="bold")
ax_delta.set_xlim(0,N_FEAT+2); ax_delta.legend(fontsize=9); ax_delta.grid(axis="y",alpha=0.3)

# C) Primeiras 40 features
ax_tipo=fig.add_subplot(gs[2,0])
top40=df_res.head(40)
cores_top=[CORES.get(s,"#9E9E9E") for s in top40["sensor"].values]
ax_tipo.barh(range(len(top40)),top40["acc_media"].values,color=cores_top,alpha=0.8,edgecolor="white")
ax_tipo.set_yticks(range(len(top40)))
ax_tipo.set_yticklabels([f"{r['passo']:3d}. {r['feature'].replace('_',' ')}" for _,r in top40.iterrows()],fontsize=7)
ax_tipo.set_xlabel("Acurácia acumulada",fontsize=10)
ax_tipo.set_title("Primeiras 40 Features Selecionadas",fontsize=12,fontweight="bold")
ax_tipo.grid(axis="x",alpha=0.3)
from matplotlib.patches import Patch
ax_tipo.legend(handles=[Patch(facecolor=CORES[s],label=f"Sensor {s}") for s in ["1","2","3"]],fontsize=9,loc="lower right")

# D) Distribuição de tipos no subset ótimo
ax_dist=fig.add_subplot(gs[2,1])
n_opt=idx_elbow+1
subset_df=df_res.head(n_opt)
contagem=subset_df["tipo"].map(lambda t:TIPOS_FEAT.get(t,t)).value_counts()
barras=ax_dist.barh(contagem.index,contagem.values,color="#5C6BC0",alpha=0.8,edgecolor="white")
ax_dist.set_xlabel("Qtd no subconjunto ótimo",fontsize=10)
ax_dist.set_title(f"Tipos de Feature no Subconjunto Ótimo ({n_opt} features)",fontsize=12,fontweight="bold")
ax_dist.grid(axis="x",alpha=0.3)
for bar,val in zip(barras,contagem.values):
    ax_dist.text(bar.get_width()+0.05,bar.get_y()+bar.get_height()/2,str(val),va="center",fontsize=10,fontweight="bold")

fig.suptitle("Seleção Gulosa de Features — Classificador de Vibração (Random Forest)",fontsize=14,fontweight="bold",y=0.99)
plt.savefig("selecao_gulosa_curva.png",dpi=150,bbox_inches="tight")
plt.close()
print("  ✓ selecao_gulosa_curva.png")

df_res.to_csv("/outputs/selecao_gulosa_resultado.csv",index=False)
subset_df2=df_res.head(n_opt)[["passo","feature","sensor","eixo","tipo","acc_media","acc_std","delta_acc"]].copy()
subset_df2["tipo"]=subset_df2["tipo"].map(lambda t:TIPOS_FEAT.get(t,t))
subset_df2.to_csv("subset_otimo.csv",index=False)
joblib.dump(rf_final,"modelo_rf_subset_otimo.pkl")
joblib.dump(le,"label_encoder.pkl")
joblib.dump(subset_features,"features_selecionadas.pkl")

print("  ✓ selecao_gulosa_resultado.csv")
print("  ✓ subset_otimo.csv")
print("  ✓ modelo_rf_subset_otimo.pkl")
print("  ✓ features_selecionadas.pkl")
print(f"""
{'='*65}
  RESUMO FINAL
{'='*65}
  Subconjunto ótimo (elbow): {n_opt} features de {N_FEAT}
  Acurácia máxima (busca):   {accs_m[idx_max]:.4f} — passo {idx_max+1}
  Acurácia elbow  (busca):   {accs_m[idx_elbow]:.4f} — passo {idx_elbow+1}
  Acurácia final  (5-fold):  {cv_scores_final.mean():.4f} ± {cv_scores_final.std():.4f}
  Redução de features:       {N_FEAT} → {n_opt} ({(1-n_opt/N_FEAT)*100:.0f}% menos)
{'='*65}
""")