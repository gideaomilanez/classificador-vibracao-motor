"""
app_vibracao.py - Monitor de Vibração
3 sensores | Abas por sensor | Tempo + FFT Stem por eixo
Suavização do espectro por média móvel
"""
import os
import sys

# Workaround para um erro comum do Qt/xcb em algumas distros Linux, onde o
# plugin de plataforma falha ao carregar libxkbcommon dinamicamente. Se as
# libs existirem no sistema, o processo se re-executa (os.execve) já com
# LD_PRELOAD configurado, antes de qualquer import do PyQt5/pyqtgraph.
if os.environ.get("VIBRACAO_ENV_FIXED") != "1":
    env = os.environ.copy()
    env["VIBRACAO_ENV_FIXED"] = "1"   # evita reexecução infinita
    env["PYQTGRAPH_QT_LIB"] = "PyQt5"
    preload = [p for p in (
        "/lib/x86_64-linux-gnu/libxkbcommon.so.0",
        "/lib/x86_64-linux-gnu/libxkbcommon-x11.so.0",
    ) if os.path.exists(p)]
    if preload:
        env["LD_PRELOAD"] = ":".join(preload)
    os.execve(sys.executable, [sys.executable] + sys.argv, env)

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import queue, threading, time, csv
from collections import deque
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QSpinBox, QGroupBox,
    QFileDialog, QStackedWidget, QProgressBar,
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont

try:
    import serial
except ImportError:
    print("pip install pyserial"); sys.exit(1)

# == Classificador (Random Forest) ==========================================
try:
    import joblib
    from scipy import stats as _stats
    from scipy.fft import rfft as _rfft, rfftfreq as _rfftfreq
    _CLF_LIBS_OK = True
except ImportError:
    _CLF_LIBS_OK = False

# == Constantes ================================================================
JANELA     = 2000
FFT_MIN    = 32
FFT_N      = 256   # nº de amostras usadas na FFT dos GRÁFICOS (menos pontos,
                   # mais legível; 256 amostras -> 128 bins no espectro)
TIMER_MS   = 50
N_SENSORES = 3

BG       = "#11111b"
PANEL    = "#181825"
BORDER   = "#313244"
TEXT     = "#cdd6f4"
SUBTEXT  = "#6c7086"
ACCENT   = "#89b4fa"
OK_CLR   = "#a6e3a1"
ERR_CLR  = "#f38ba8"
WARN_CLR = "#f9e2af"

SENSOR_NOMES = [
    "Sensor 1 — MPU-9250",
    "Sensor 2 — MPU-9250",
    "Sensor 3 — MPU-6050",
]
SENSOR_CORES = [
    ("#5B9FD4", "#7BBFE8", "#A8D8F0", "#E24B4A"),
    ("#EF9F27", "#F5B94A", "#FAD48A", "#F97316"),
    ("#4CAF50", "#66BB6A", "#A5D6A7", "#26A69A"),
]
EIXOS = ["X", "Y", "Z"]

# == Classificador - constantes ================================================
CLF_WINDOW = 256   # nº de amostras usadas para extrair as features (igual ao treino)
CLF_EVERY  = 15    # roda a predição a cada N ticks do timer (~15*50ms = 0.75s)
CLF_SMOOTH = 5     # média das últimas N predições (suaviza oscilações)

# IMPORTANTE: o modelo foi treinado com FS fixo. A frequência usada na extração
# de features em tempo real DEVE ser a mesma do treino, senão as features de
# frequência (dom_freq, centroid...) ficam deslocadas e o modelo erra.
FS_TREINO = 59     # mesmo valor do FS em treino_final.py

# Ordem de exibição no diagnóstico: somente 2 classes
CLASSES_ORDEM = ["balanceado", "desbalanceado"]

NOMES_CLASSES_DISPLAY = {
    "balanceado": "Balanceado",
    "desbalanceado": "Desbalanceado",
}

CORES_ESTADO = {
    "balanceado": OK_CLR,
    "desbalanceado": ERR_CLR,
}

ICONES_ESTADO = {
    "balanceado": "✅",
    "desbalanceado": "⚠️",
}

# Canais na MESMA ordem das colunas do buffer (x1,y1,z1,x2,y2,z2,x3,y3,z3)
CANAIS_CLF = ["x1", "y1", "z1", "x2", "y2", "z2", "x3", "y3", "z3"]

# == Carregar modelo treinado ================================================
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODELO = None
LABEL_ENCODER = None
FEATURES_SELECIONADAS = None
MODELO_OK = False
MODELO_ERRO = ""

if _CLF_LIBS_OK:
    try:
        MODELO = joblib.load(os.path.join(MODEL_DIR, "..", "modelos", "modelo_rf.pkl"))
        LABEL_ENCODER = joblib.load(os.path.join(MODEL_DIR, "..", "modelos", "label_encoder.pkl"))
        FEATURES_SELECIONADAS = joblib.load(os.path.join(MODEL_DIR, "..", "modelos", "features_selecionadas.pkl"))
        MODELO_OK = True
    except Exception as e:
        MODELO_ERRO = str(e)
else:
    MODELO_ERRO = "joblib/scipy não instalados (pip install joblib scipy)"


# == Extração de features (igual ao treino_final.py) ==========================
def _features_tempo(s):
    rms = np.sqrt(np.mean(s ** 2))
    return {
        "mean": np.mean(s), "std": np.std(s), "rms": rms,
        "variance": np.var(s), "kurtosis": _stats.kurtosis(s),
        "skewness": _stats.skew(s), "peak_to_peak": np.ptp(s),
        "crest_factor": np.max(np.abs(s)) / (rms + 1e-10),
    }

def _features_frequencia(s, fs):
    N = len(s)
    freqs = _rfftfreq(N, 1. / fs)
    mag = np.abs(_rfft(s)) / N
    freqs, mag = freqs[1:], mag[1:]
    e = np.sum(mag ** 2) + 1e-10
    en = mag ** 2 / e
    return {
        "dom_freq": freqs[np.argmax(mag)],
        "spectral_centroid": np.sum(freqs * mag) / (np.sum(mag) + 1e-10),
        "spectral_energy": np.sum(mag ** 2),
        "spectral_entropy": -np.sum(en * np.log(en + 1e-10)),
    }

def extrair_features_clf(janela, fs):
    """janela: array (CLF_WINDOW, 9) na ordem CANAIS_CLF."""
    feat = {}
    for i, canal in enumerate(CANAIS_CLF):
        s = janela[:, i]
        prefixo = f"s{canal[-1]}_{canal[:-1]}"
        for k, v in _features_tempo(s).items():
            feat[f"{prefixo}_{k}"] = v
        for k, v in _features_frequencia(s, fs).items():
            feat[f"{prefixo}_{k}"] = v
    return feat

# == Serial thread =============================================================
# Roda em thread separada para não travar a interface enquanto lê a porta serial.
# Os dados (ou erros) são passados para a UI via fila (queue), que é consumida
# no timer da janela principal (_update).
def serial_thread(porta, baud, q, stop):
    try:
        ser = serial.Serial(porta, baudrate=baud, timeout=1.0)
        time.sleep(2)   # tempo para o ESP32 reiniciar após abrir a porta serial
        ser.reset_input_buffer()
        q.put(("STATUS", "Conectado"))
        while not stop.is_set():
            try:
                raw = ser.readline().decode("ascii", errors="ignore").strip()
                parts = raw.split(",")
                if len(parts) == 9:   # ignora linhas incompletas/corrompidas
                    q.put((time.time(), tuple(float(v) for v in parts)))
            except (ValueError, serial.SerialException):
                continue
        ser.close()
    except serial.SerialException as e:
        q.put(("ERRO", str(e)))

# == Média móvel + FFT =========================================================
def suavizar_espectro(amp, janela=5):
    if janela <= 1:
        return amp
    return np.convolve(amp, np.ones(janela) / janela, mode="same")

def calcular_fft(buf, col, fs, suavizar=False, janela_mm=5):
    arr = np.array([r[col] for r in buf], dtype=float)
    if len(arr) < FFT_MIN:
        return np.zeros(1), np.zeros(1)
    # Usa apenas as últimas FFT_N amostras: o espectro fica com FFT_N/2 bins
    # (ex.: 256 amostras -> 128 pontos), muito mais legível do que usar o
    # buffer inteiro de 2000 amostras (que geraria ~1000 pontos no stem).
    if len(arr) > FFT_N:
        arr = arr[-FFT_N:]
    arr -= arr.mean()
    w = np.hanning(len(arr))
    spec = np.fft.rfft(arr * w)
    freqs = np.fft.rfftfreq(len(arr), d=1.0 / fs)
    amp = np.abs(spec) * 2.0 / np.sum(w)
    if suavizar:
        amp = suavizar_espectro(amp, janela_mm)
    return freqs, amp

# Monta os pontos de um gráfico "stem" (uma linha vertical por frequência,
# do eixo 0 até a amplitude) usando um único plot contínuo: cada tripla de
# pontos é (base, topo, NaN) -- o NaN quebra a linha entre uma haste e a próxima.
def stem_xy(freqs, amp):
    n = len(freqs)
    xs = np.empty(n * 3); ys = np.empty(n * 3)
    xs[0::3] = freqs; xs[1::3] = freqs; xs[2::3] = np.nan
    ys[0::3] = 0;     ys[1::3] = amp;   ys[2::3] = np.nan
    return xs, ys

# == Janela principal ==========================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Monitor de Vibração — 3 Sensores")
        self.resize(1450, 920)
        self.setStyleSheet(f"background:{BG};")

        self.monitoring   = False
        self.collecting   = False
        self.q            = queue.Queue()
        self.stop_event   = None
        self.buffer       = deque(maxlen=JANELA)
        self.ts_buf       = deque(maxlen=JANELA)
        self.col_buf      = []
        self.col_ts       = []
        self.n_recv       = 0
        self.current_page = 0
        self.pred_counter = 0
        self.proba_hist   = deque(maxlen=CLF_SMOOTH)  # histórico p/ suavizar predição

        pg.setConfigOption("background", BG)
        pg.setConfigOption("foreground", TEXT)

        self._build_ui()
        self._build_sensor_pages()
        self._build_compare_page()
        self._build_diag_page()
        self._show_page(0)

        self.timer = QTimer()
        self.timer.timeout.connect(self._update)
        self.timer.start(TIMER_MS)

    # == Estilos ===============================================================
    def _nav_style(self, active=False, color=ACCENT):
        bg = color if active else BORDER
        fg = "#1e1e2e" if active else TEXT
        return (
            f"QPushButton{{background:{bg};color:{fg};border-radius:7px;"
            f"font-weight:{'bold' if active else 'normal'};padding:9px;"
            f"font-size:12px;text-align:left;padding-left:14px;}}"
            f"QPushButton:hover{{background:{color};color:#1e1e2e;}}"
        )

    def _inp_style(self):
        return (f"background:#313244;color:{TEXT};border:1px solid {BORDER};"
                f"border-radius:4px;padding:5px;font-size:11px;")

    def _grp_style(self):
        return (f"QGroupBox{{color:{ACCENT};font-weight:bold;font-size:11px;"
                f"border:1px solid {BORDER};border-radius:6px;"
                f"margin-top:10px;padding-top:8px;}}")

    def _bar_style(self, cor):
        return (
            f"QProgressBar{{background:{BORDER};border-radius:7px;"
            f"min-height:20px;max-height:20px;text-align:center;"
            f"color:{TEXT};font-size:11px;font-weight:bold;}}"
            f"QProgressBar::chunk{{background:{cor};border-radius:7px;}}"
        )

    def _lbl(self, txt):
        l = QLabel(txt)
        l.setStyleSheet(f"color:{TEXT};font-size:11px;")
        return l

    # == UI principal ==========================================================
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setSpacing(6)
        root.setContentsMargins(6, 6, 6, 6)

        # == Sidebar =======================================================
        sidebar = QWidget()
        sidebar.setFixedWidth(225)
        sidebar.setStyleSheet(f"background:{PANEL};border-radius:10px;")
        vb = QVBoxLayout(sidebar)
        vb.setSpacing(5)
        vb.setContentsMargins(10, 12, 10, 12)

        nav = [
            ("🔵", "Sensor 1",    SENSOR_CORES[0][0]),
            ("🟠", "Sensor 2",    SENSOR_CORES[1][0]),
            ("🟢", "Sensor 3",    SENSOR_CORES[2][0]),
            ("📊", "Comparar",    ACCENT),
            ("🔧", "Diagnóstico", "#b4befe"),
        ]
        self.nav_btns = []
        for i, (icon, label, cor) in enumerate(nav):
            btn = QPushButton(f"{icon}  {label}")
            btn.setStyleSheet(self._nav_style(i == 0, cor))
            btn.clicked.connect(lambda _, idx=i, c=cor: self._show_page(idx, c))
            vb.addWidget(btn)
            self.nav_btns.append(btn)

        vb.addSpacing(8)

        # Conexão
        g_c = QGroupBox("⚡ Conexão")
        g_c.setStyleSheet(self._grp_style())
        gc = QVBoxLayout(g_c)
        gc.addWidget(self._lbl("Porta serial"))
        self.inp_porta = QLineEdit("/dev/ttyUSB1")
        self.inp_porta.setStyleSheet(self._inp_style())
        gc.addWidget(self.inp_porta)
        gc.addWidget(self._lbl("Baud rate"))
        self.inp_baud = QLineEdit("115200")
        self.inp_baud.setStyleSheet(self._inp_style())
        gc.addWidget(self.inp_baud)
        vb.addWidget(g_c)

        # Coleta
        g_col = QGroupBox("💾 Coleta")
        g_col.setStyleSheet(self._grp_style())
        gco = QVBoxLayout(g_col)
        gco.addWidget(self._lbl("Nº de amostras"))
        self.spin_n = QSpinBox()
        self.spin_n.setRange(100, 100_000)
        self.spin_n.setValue(1000)
        self.spin_n.setSingleStep(100)
        self.spin_n.setStyleSheet(self._inp_style())
        gco.addWidget(self.spin_n)

        bstyle = (
            f"QPushButton{{background:{ACCENT};color:#1e1e2e;border-radius:5px;"
            f"font-weight:bold;padding:6px;font-size:11px;}}"
            f"QPushButton:hover{{background:#74a8f7;}}"
            f"QPushButton:disabled{{background:{BORDER};color:{SUBTEXT};}}"
        )
        self.btn_coletar = QPushButton("⏺  Iniciar Coleta")
        self.btn_coletar.setStyleSheet(bstyle)
        self.btn_coletar.clicked.connect(self._iniciar_coleta)
        self.btn_coletar.setEnabled(False)
        gco.addWidget(self.btn_coletar)

        self.lbl_coleta = QLabel("—")
        self.lbl_coleta.setStyleSheet(f"color:{SUBTEXT};font-size:10px;")
        self.lbl_coleta.setAlignment(Qt.AlignCenter)
        gco.addWidget(self.lbl_coleta)

        self.btn_salvar = QPushButton("⬇  Salvar CSV")
        self.btn_salvar.setStyleSheet(bstyle)
        self.btn_salvar.clicked.connect(self._salvar_csv)
        self.btn_salvar.setEnabled(False)
        gco.addWidget(self.btn_salvar)
        vb.addWidget(g_col)

        # Suavização do espectro
        g_filt = QGroupBox("〰 Suavização do Espectro")
        g_filt.setStyleSheet(self._grp_style())
        gfl = QVBoxLayout(g_filt)

        self.btn_filtro = QPushButton("Suavização: DESLIGADA")
        self.btn_filtro.setCheckable(True)
        self.btn_filtro.setChecked(False)
        self.btn_filtro.setStyleSheet(
            f"QPushButton{{background:{BORDER};color:{TEXT};border-radius:5px;"
            f"font-size:11px;padding:5px;font-weight:bold;}}"
            f"QPushButton:checked{{background:#cba6f7;color:#1e1e2e;}}"
        )
        self.btn_filtro.toggled.connect(
            lambda on: self.btn_filtro.setText(
                "Suavização: LIGADA" if on else "Suavização: DESLIGADA"
            )
        )
        gfl.addWidget(self.btn_filtro)
        gfl.addWidget(self._lbl("Janela da média móvel"))
        self.spin_mm = QSpinBox()
        self.spin_mm.setRange(1, 21)
        self.spin_mm.setValue(5)
        self.spin_mm.setSingleStep(2)
        self.spin_mm.setSuffix(" bins")
        self.spin_mm.setStyleSheet(self._inp_style())
        gfl.addWidget(self.spin_mm)
        vb.addWidget(g_filt)

        vb.addStretch()

        self.lbl_status = QLabel("⚪  Desconectado")
        self.lbl_status.setStyleSheet(
            f"color:{SUBTEXT};font-size:10px;padding:5px;"
            f"background:#1e1e2e;border-radius:5px;")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setWordWrap(True)
        vb.addWidget(self.lbl_status)

        self.btn_ligar = QPushButton("▶  Ligar Monitor")
        self.btn_ligar.setStyleSheet(
            f"QPushButton{{background:{OK_CLR};color:#1e1e2e;border-radius:6px;"
            f"font-weight:bold;padding:8px;font-size:12px;}}"
            f"QPushButton:hover{{background:#94d889;}}")
        self.btn_ligar.clicked.connect(self._ligar)
        vb.addWidget(self.btn_ligar)

        self.btn_desligar = QPushButton("⏹  Desligar")
        self.btn_desligar.setStyleSheet(
            f"QPushButton{{background:{ERR_CLR};color:#1e1e2e;border-radius:6px;"
            f"font-weight:bold;padding:8px;font-size:12px;}}"
            f"QPushButton:hover{{background:#e07a96;}}")
        self.btn_desligar.clicked.connect(self._desligar)
        self.btn_desligar.setEnabled(False)
        vb.addWidget(self.btn_desligar)

        root.addWidget(sidebar)

        self.stack = QStackedWidget()
        self.stack.setStyleSheet(f"background:{BG};")
        root.addWidget(self.stack, stretch=1)

    # == Páginas dos sensores ==================================================
    def _build_sensor_pages(self):
        self.sensor_data = []
        for si in range(N_SENSORES):
            cx, cy, cz, cf = SENSOR_CORES[si]
            eixo_cores = [cx, cy, cz]

            w = QWidget()
            w.setStyleSheet(f"background:{BG};")
            vb = QVBoxLayout(w)
            vb.setContentsMargins(4, 4, 4, 4)
            vb.setSpacing(4)

            title = QLabel(
                f"<span style='color:{cx};font-size:14px;font-weight:bold;'>"
                f"● {SENSOR_NOMES[si]}</span>")
            title.setStyleSheet(f"background:{BG};padding:2px 4px;")
            vb.addWidget(title)

            glw = pg.GraphicsLayoutWidget()
            glw.setStyleSheet(f"background:{BG};")

            time_curves, fft_stems, fft_dots, fft_peaks, fft_plots = [], [], [], [], []

            for ei, (eixo, cor) in enumerate(zip(EIXOS, eixo_cores)):
                # Tempo
                pt = glw.addPlot(row=ei, col=0)
                pt.setTitle(f"<span style='color:{cor};font-size:11px;'>Tempo — {eixo}</span>")
                pt.setLabel("left", "g", size="9px")
                pt.showGrid(x=True, y=True, alpha=0.12)
                pt.setYRange(-15, 15)
                if ei < 2:
                    pt.getAxis("bottom").setStyle(showValues=False)
                else:
                    pt.setLabel("bottom", "amostras", size="9px")
                ct = pt.plot(pen=pg.mkPen(cor, width=1.3))
                time_curves.append(ct)

                # FFT Stem
                pf = glw.addPlot(row=ei, col=1)
                pf.setTitle(f"<span style='color:{cf};font-size:11px;'>FFT — {eixo}</span>")
                pf.setLabel("left", "Amp (g)", size="9px")
                pf.showGrid(x=True, y=True, alpha=0.12)
                if ei < 2:
                    pf.getAxis("bottom").setStyle(showValues=False)
                else:
                    pf.setLabel("bottom", "Hz", size="9px")
                pf.addLine(y=0, pen=pg.mkPen(SUBTEXT, width=0.5))

                stem = pf.plot(pen=pg.mkPen(cf, width=1.2))
                dots = pf.plot(pen=None, symbol="o",
                               symbolBrush=pg.mkBrush(cf),
                               symbolSize=4,
                               symbolPen=pg.mkPen(cf, width=0.5))
                peak = pf.plot(pen=None, symbol="o",
                               symbolBrush=pg.mkBrush("red"),
                               symbolSize=9,
                               symbolPen=pg.mkPen("red", width=1))

                fft_stems.append(stem)
                fft_dots.append(dots)
                fft_peaks.append(peak)
                fft_plots.append(pf)

            glw.ci.layout.setColumnStretchFactor(0, 1)
            glw.ci.layout.setColumnStretchFactor(1, 1)
            vb.addWidget(glw)
            self.stack.addWidget(w)
            self.sensor_data.append((time_curves, fft_stems, fft_dots, fft_peaks, fft_plots))

    # == Página comparar =======================================================
    def _build_compare_page(self):
        w = QWidget()
        w.setStyleSheet(f"background:{BG};")
        vb = QVBoxLayout(w)
        vb.setContentsMargins(4, 4, 4, 4)

        title = QLabel(
            f"<span style='color:{ACCENT};font-size:14px;font-weight:bold;'>"
            f"📊 Comparação — todos os sensores</span>")
        title.setStyleSheet(f"background:{BG};padding:2px 4px;")
        vb.addWidget(title)

        glw = pg.GraphicsLayoutWidget()
        glw.setStyleSheet(f"background:{BG};")
        self.cmp_curves = []

        for ei, eixo in enumerate(EIXOS):
            p = glw.addPlot(row=ei, col=0)
            p.setTitle(f"<span style='font-size:11px;'>Eixo {eixo} — todos os sensores</span>")
            p.setLabel("left", "g", size="9px")
            p.showGrid(x=True, y=True, alpha=0.12)
            p.setYRange(-15, 15)
            p.addLegend(offset=(10, 10))
            if ei < 2:
                p.getAxis("bottom").setStyle(showValues=False)
            else:
                p.setLabel("bottom", "amostras", size="9px")

            row_c = []
            for si in range(N_SENSORES):
                cor = SENSOR_CORES[si][ei]
                c = p.plot(pen=pg.mkPen(cor, width=1.3), name=f"S{si+1}")
                row_c.append(c)
            self.cmp_curves.append(row_c)

        vb.addWidget(glw)
        self.stack.addWidget(w)

    # == Página diagnóstico ====================================================
    def _build_diag_page(self):
        w = QWidget()
        w.setStyleSheet(f"background:{BG};")
        vb = QVBoxLayout(w)
        vb.setContentsMargins(16, 16, 16, 16)
        vb.setSpacing(12)

        title = QLabel(
            f"<span style='color:#b4befe;font-size:16px;font-weight:bold;'>"
            f"🔧 Diagnóstico do Motor</span>")
        title.setStyleSheet(f"background:{BG};padding:2px 4px;")
        vb.addWidget(title)

        # == Card principal: estado atual ====================================
        self.diag_card = QWidget()
        self.diag_card.setStyleSheet(
            f"background:{PANEL};border-radius:14px;"
            f"border:3px solid {BORDER};")
        card_vb = QVBoxLayout(self.diag_card)
        card_vb.setContentsMargins(24, 20, 24, 20)
        card_vb.setSpacing(4)

        self.diag_icon = QLabel("⏳")
        self.diag_icon.setAlignment(Qt.AlignCenter)
        self.diag_icon.setStyleSheet("font-size:56px;background:transparent;")
        card_vb.addWidget(self.diag_icon)

        self.diag_estado = QLabel("Aguardando dados...")
        self.diag_estado.setAlignment(Qt.AlignCenter)
        self.diag_estado.setStyleSheet(
            f"color:{TEXT};font-size:26px;font-weight:bold;background:transparent;")
        card_vb.addWidget(self.diag_estado)

        self.diag_confianca = QLabel(" ")
        self.diag_confianca.setAlignment(Qt.AlignCenter)
        self.diag_confianca.setStyleSheet(
            f"color:{SUBTEXT};font-size:15px;background:transparent;")
        card_vb.addWidget(self.diag_confianca)

        vb.addWidget(self.diag_card)

        # == Barras de probabilidade por classe ==============================
        g_prob = QGroupBox("📈 Probabilidade por Estado (média das últimas "
                           f"{CLF_SMOOTH} análises)")
        g_prob.setStyleSheet(self._grp_style())
        prob_vb = QVBoxLayout(g_prob)
        prob_vb.setSpacing(8)

        self.diag_bars = {}
        for classe in CLASSES_ORDEM:
            row = QHBoxLayout()
            row.setSpacing(10)

            icone_lbl = QLabel(ICONES_ESTADO[classe])
            icone_lbl.setStyleSheet("font-size:14px;")
            icone_lbl.setFixedWidth(22)
            row.addWidget(icone_lbl)

            nome_lbl = QLabel(NOMES_CLASSES_DISPLAY[classe])
            nome_lbl.setStyleSheet(f"color:{TEXT};font-size:12px;")
            nome_lbl.setFixedWidth(190)
            row.addWidget(nome_lbl)

            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(True)
            bar.setFormat("%p%")
            bar.setStyleSheet(self._bar_style(CORES_ESTADO[classe]))
            row.addWidget(bar)

            prob_vb.addLayout(row)
            self.diag_bars[classe] = bar

        vb.addWidget(g_prob)

        # == Informações técnicas =============================================
        g_info = QGroupBox("ℹ Informações Técnicas")
        g_info.setStyleSheet(self._grp_style())
        info_vb = QVBoxLayout(g_info)

        self.diag_text = QLabel("Aguardando conexão...")
        self.diag_text.setStyleSheet(
            f"background:#1e1e2e;color:{TEXT};font-size:11px;"
            f"font-family:monospace;padding:12px;border-radius:8px;")
        self.diag_text.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.diag_text.setWordWrap(True)
        info_vb.addWidget(self.diag_text)

        vb.addWidget(g_info)

        # == Aviso se o modelo não carregou ==================================
        if not MODELO_OK:
            aviso = QLabel(
                f"⚠ Modelo de classificação não carregado: {MODELO_ERRO}\n"
                f"Coloque modelo_final_rf.pkl, label_encoder.pkl e "
                f"features_selecionadas.pkl na mesma pasta deste script.")
            aviso.setStyleSheet(
                f"color:{WARN_CLR};font-size:11px;background:#1e1e2e;"
                f"border-radius:8px;padding:10px;")
            aviso.setWordWrap(True)
            vb.addWidget(aviso)

        vb.addStretch()
        self.stack.addWidget(w)

    # == Navegação =============================================================
    def _show_page(self, idx, cor=ACCENT):
        self.current_page = idx
        self.stack.setCurrentIndex(idx)
        nav_cores = [
            SENSOR_CORES[0][0], SENSOR_CORES[1][0], SENSOR_CORES[2][0],
            ACCENT, "#b4befe",
        ]
        for i, btn in enumerate(self.nav_btns):
            btn.setStyleSheet(self._nav_style(i == idx, nav_cores[i]))

    # == Controles =============================================================
    def _ligar(self):
        porta = self.inp_porta.text().strip()
        baud  = int(self.inp_baud.text().strip())
        self.q          = queue.Queue()
        self.stop_event = threading.Event()
        self.buffer     = deque(maxlen=JANELA)
        self.ts_buf     = deque(maxlen=JANELA)
        self.n_recv     = 0
        self.proba_hist.clear()   # zera o histórico de predições
        t = threading.Thread(
            target=serial_thread,
            args=(porta, baud, self.q, self.stop_event),
            daemon=True,
        )
        t.start()
        self.monitoring = True
        self.btn_ligar.setEnabled(False)
        self.btn_desligar.setEnabled(True)
        self.btn_coletar.setEnabled(True)
        self._set_status(WARN_CLR, "🟡  Conectando...")

    def _desligar(self):
        if self.stop_event:
            self.stop_event.set()
        self.monitoring = False
        self.collecting = False
        self.btn_ligar.setEnabled(True)
        self.btn_desligar.setEnabled(False)
        self.btn_coletar.setEnabled(False)
        self._set_status(SUBTEXT, "⚪  Desconectado")

    def _iniciar_coleta(self):
        self.col_buf = []
        self.col_ts  = []
        self.collecting = True
        self.btn_salvar.setEnabled(False)
        self.lbl_coleta.setText("Coletando...")
        self.lbl_coleta.setStyleSheet(f"color:{ACCENT};font-size:10px;")

    def _salvar_csv(self):
        if not self.col_buf:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Salvar CSV", "vibracao.csv", "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp","x1","y1","z1","x2","y2","z2","x3","y3","z3"])
            for ts, row in zip(self.col_ts, self.col_buf):
                w.writerow([ts, *row])
        self.statusBar().showMessage(f"✅  Salvo em {path}")

    def _set_status(self, color, txt):
        self.lbl_status.setText(txt)
        self.lbl_status.setStyleSheet(
            f"color:{color};font-size:10px;padding:5px;"
            f"background:#1e1e2e;border-radius:5px;")

    # == Loop de atualização ===================================================
    def _update(self):
        if not self.monitoring:
            return

        while not self.q.empty():
            item = self.q.get_nowait()
            if isinstance(item[0], str):
                tag, msg = item
                if tag == "STATUS":
                    self._set_status(OK_CLR, f"🟢  {msg}")
                elif tag == "ERRO":
                    self._set_status(ERR_CLR, f"❌  {msg}")
                    self._desligar()
                continue
            ts, vals = item
            self.buffer.append(vals)
            self.ts_buf.append(ts)
            self.n_recv += 1
            if self.collecting:
                self.col_buf.append(vals)
                self.col_ts.append(ts)
                n_col = self.spin_n.value()
                prog  = len(self.col_buf)
                self.lbl_coleta.setText(f"{prog}/{n_col}")
                if prog >= n_col:
                    self.collecting = False
                    self.lbl_coleta.setText(f"✅  {prog} amostras")
                    self.lbl_coleta.setStyleSheet(f"color:{OK_CLR};font-size:10px;")
                    self.btn_salvar.setEnabled(True)

        buf = list(self.buffer)
        if len(buf) < 2:
            return

        xs      = np.arange(len(buf))
        ts_list = list(self.ts_buf)
        fs      = ((len(ts_list)-1) / (ts_list[-1] - ts_list[0])
                   if len(ts_list) >= 2 else 100.0)

        suavizar  = self.btn_filtro.isChecked()
        janela_mm = self.spin_mm.value()

        # == Páginas dos sensores ==========================================
        for si in range(N_SENSORES):
            tc, fst, fdt, fpt, fpl = self.sensor_data[si]
            for ei in range(3):
                col = si * 3 + ei
                tc[ei].setData(xs, [r[col] for r in buf])
                if len(buf) >= FFT_MIN:
                    freqs, amp = calcular_fft(
                        buf, col, fs,
                        suavizar=suavizar, janela_mm=janela_mm,
                    )
                    sx, sy = stem_xy(freqs, amp)
                    fst[ei].setData(sx, sy)
                    fdt[ei].setData(freqs, amp)
                    fpl[ei].setXRange(0, fs / 2, padding=0)
                    if len(amp) > 1:
                        pk = int(np.argmax(amp[1:])) + 1
                        fpt[ei].setData([freqs[pk]], [amp[pk]])
                        cf = SENSOR_CORES[si][3]
                        fpl[ei].setTitle(
                            f"<span style='color:{cf};font-size:10px;'>"
                            f"FFT — {EIXOS[ei]}  |  pico: {freqs[pk]:.1f} Hz"
                            f"{'  〰' if suavizar else ''}</span>"
                        )

        # == Comparar ======================================================
        for ei in range(3):
            for si in range(N_SENSORES):
                col = si * 3 + ei
                self.cmp_curves[ei][si].setData(xs, [r[col] for r in buf])

        # == Diagnóstico ===================================================
        self.pred_counter += 1
        if MODELO_OK:
            if len(buf) >= CLF_WINDOW:
                if self.pred_counter % CLF_EVERY == 0:
                    self._atualizar_diagnostico(buf[-CLF_WINDOW:])
            else:
                faltam = CLF_WINDOW - len(buf)
                self.diag_icon.setText("⏳")
                self.diag_estado.setText("Coletando dados...")
                self.diag_estado.setStyleSheet(
                    f"color:{TEXT};font-size:26px;font-weight:bold;background:transparent;")
                self.diag_confianca.setText(
                    f"Aguardando {faltam} amostra(s) para a primeira análise")
                self.diag_card.setStyleSheet(
                    f"background:{PANEL};border-radius:14px;border:3px solid {BORDER};")

        if self.current_page == 4:
            self.diag_text.setText(
                f"Amostras recebidas : {self.n_recv}\n"
                f"fs estimada (buffer): {fs:.2f} Hz\n"
                f"fs usada no modelo : {FS_TREINO} Hz (fixa, igual ao treino)\n"
                f"Buffer atual       : {len(buf)} / {JANELA}\n"
                f"Janela classificação: {CLF_WINDOW} amostras "
                f"(atualiza a cada {CLF_EVERY * TIMER_MS / 1000:.1f}s, "
                f"média das últimas {CLF_SMOOTH})\n"
                f"Coletando          : {'Sim' if self.collecting else 'Não'}\n"
                f"Amostras coletadas : {len(self.col_buf)}\n"
                f"Suavização espectro: "
                f"{'LIGADA (janela=' + str(janela_mm) + ' bins)' if suavizar else 'DESLIGADA'}\n"
            )

        self.statusBar().showMessage(
            f"Amostras: {self.n_recv}  |  fs ≈ {fs:.1f} Hz  |  "
            f"Buffer: {len(buf)}/{JANELA}  |  "
            f"Suavização: {'ON janela=' + str(janela_mm) + ' bins' if suavizar else 'OFF'}"
        )

    # == Classificação do estado do motor ======================================
    def _atualizar_diagnostico(self, ultimas_linhas):
        try:
            dados = np.array(ultimas_linhas, dtype=float)  # (CLF_WINDOW, 9)

            # CRÍTICO: usa FS_TREINO fixo (igual ao treino), e NÃO o fs estimado
            # do buffer. O fs estimado oscila com o jitter da serial e desloca
            # as features de frequência, que são as mais importantes do modelo.
            feat = extrair_features_clf(dados, FS_TREINO)

            x = np.array(
                [[feat[f] for f in FEATURES_SELECIONADAS]], dtype=np.float32
            )
            proba = MODELO.predict_proba(x)[0]

            # Suavização: média das últimas CLF_SMOOTH predições.
            # Evita que uma única janela ruidosa mude o estado exibido.
            self.proba_hist.append(proba)
            proba_media = np.mean(self.proba_hist, axis=0)

            classes = LABEL_ENCODER.classes_

            # Agrupa as saídas originais em apenas 2 estados:
            # - motor_limpo  -> Balanceado
            # - demais       -> Desbalanceado
            p_balanceado = 0.0
            p_desbalanceado = 0.0

            for classe, p in zip(classes, proba_media):
                if classe == "motor_limpo":
                    p_balanceado += float(p)
                else:
                    p_desbalanceado += float(p)

            # Atualiza barras de probabilidade agrupadas
            if "balanceado" in self.diag_bars:
                self.diag_bars["balanceado"].setValue(int(round(p_balanceado * 100)))
            if "desbalanceado" in self.diag_bars:
                self.diag_bars["desbalanceado"].setValue(int(round(p_desbalanceado * 100)))

            if p_balanceado >= p_desbalanceado:
                classe_pred = "balanceado"
                confianca = float(p_balanceado)
            else:
                classe_pred = "desbalanceado"
                confianca = float(p_desbalanceado)

            cor = CORES_ESTADO.get(classe_pred, TEXT)
            self.diag_icon.setText(ICONES_ESTADO.get(classe_pred, "❓"))
            self.diag_estado.setText(
                NOMES_CLASSES_DISPLAY.get(classe_pred, classe_pred).upper()
            )
            self.diag_estado.setStyleSheet(
                f"color:{cor};font-size:26px;font-weight:bold;background:transparent;")

            # Indica baixa confiança explicitamente para o usuário
            if confianca < 0.55:
                self.diag_confianca.setText(
                    f"Confiança: {confianca * 100:.1f}%  (baixa — estado incerto)")
            else:
                self.diag_confianca.setText(f"Confiança: {confianca * 100:.1f}%")

            self.diag_card.setStyleSheet(
                f"background:{PANEL};border-radius:14px;border:3px solid {cor};")
        except Exception as e:
            self.diag_icon.setText("❌")
            self.diag_estado.setText("Erro na predição")
            self.diag_estado.setStyleSheet(
                f"color:{ERR_CLR};font-size:22px;font-weight:bold;background:transparent;")
            self.diag_confianca.setText(str(e))

    def closeEvent(self, event):
        if self.stop_event:
            self.stop_event.set()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 9))
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())