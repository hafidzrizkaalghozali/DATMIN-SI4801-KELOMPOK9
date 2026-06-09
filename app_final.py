# ============================================================
# TUGAS BESAR PENAMBANGAN DATA — KELOMPOK 9 SI4801
# Analisis Lahan Pertanian: K-Means, Logistic Regression,
# Naïve Bayes Classifier
# Semester Genap TA 2025/2026 — Universitas Telkom
# ============================================================
# FILE INI MENGGABUNGKAN SELURUH KODE NOTEBOOK + DASHBOARD
# Jalankan: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from io import StringIO

# ── Sklearn ──────────────────────────────────────────────────────────────────
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, classification_report, confusion_matrix,
    silhouette_score, roc_curve, auc, ConfusionMatrixDisplay
)
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Penambangan Data | Kelompok 9 SI4801",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

.stApp { background: linear-gradient(135deg,#0f1923 0%,#1a2a3a 50%,#0f1923 100%); color:#e8f0f8; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#0d1f2d 0%,#152535 100%); border-right:1px solid #2a4a6a; }

.kpi { background:linear-gradient(135deg,#1a3a5c,#0d2540); border:1px solid #2a5a8a;
       border-radius:14px; padding:18px 20px; text-align:center; }
.kpi .lbl { font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:1.2px; color:#7fb3d3; margin-bottom:6px; }
.kpi .val { font-size:28px; font-weight:800; color:#4fc3f7; line-height:1; }
.kpi .sub { font-size:11px; color:#5a8aaa; margin-top:4px; }

.sec { font-size:20px; font-weight:800; color:#e8f0f8; border-left:4px solid #4fc3f7;
       padding-left:12px; margin:24px 0 14px; }
.box { background:rgba(79,195,247,.08); border:1px solid rgba(79,195,247,.25);
       border-radius:12px; padding:14px 18px; font-size:13.5px; color:#c8dff0; line-height:1.75; margin:10px 0; }
.tag { display:inline-block; background:rgba(79,195,247,.12); border:1px solid rgba(79,195,247,.35);
       border-radius:20px; padding:3px 11px; font-size:11px; font-weight:700; color:#4fc3f7; margin:2px; }
.code-cell { background:#0d1f2d; border:1px solid #1e3a5c; border-radius:10px;
             padding:14px 18px; font-family:'JetBrains Mono',monospace; font-size:12px;
             color:#a8d8f0; margin:8px 0; white-space:pre-wrap; overflow-x:auto; }
.out-cell  { background:#071218; border-left:3px solid #4fc3f7;
             border-radius:0 8px 8px 0; padding:10px 16px; font-family:'JetBrains Mono',monospace;
             font-size:12px; color:#7fb3d3; margin:0 0 14px; white-space:pre-wrap; }
.pred0 { background:linear-gradient(135deg,#1a3a5c,#0d2540); border:2px solid #4fc3f7;
         border-radius:14px; padding:20px; text-align:center; }
.pred1 { background:linear-gradient(135deg,#3a1a0d,#401a00); border:2px solid #ff7043;
         border-radius:14px; padding:20px; text-align:center; }
.plbl { font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:1px; color:#aaa; margin-bottom:6px; }
.pval { font-size:24px; font-weight:800; }

.stTabs [data-baseweb="tab-list"] { background:rgba(13,31,45,.8); border-radius:10px; padding:4px; gap:4px; }
.stTabs [data-baseweb="tab"] { border-radius:8px; color:#7fb3d3; font-weight:600; font-size:13px; }
.stTabs [aria-selected="true"] { background:linear-gradient(135deg,#1a4a7a,#0d2f5a) !important; color:#4fc3f7 !important; }
.stButton>button { background:linear-gradient(135deg,#1565C0,#0d47a1); color:white; border:none;
                   border-radius:10px; font-weight:700; padding:10px 26px; transition:all .2s; }
.stButton>button:hover { transform:translateY(-1px); box-shadow:0 4px 16px rgba(21,101,192,.4); }
div[data-testid="stMetric"] { background:linear-gradient(135deg,#1a3a5c,#0d2540); border:1px solid #2a5a8a; border-radius:12px; padding:14px 18px; }
div[data-testid="stMetric"] label { color:#7fb3d3 !important; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color:#4fc3f7 !important; font-weight:800 !important; }
[data-testid="stDataFrame"] { border-radius:12px; overflow:hidden; }
hr { border-color:#2a4a6a; margin:20px 0; }
</style>
""", unsafe_allow_html=True)

# ─── MATPLOTLIB DARK ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':'#0f1923','axes.facecolor':'#111d2a','axes.edgecolor':'#2a4a6a',
    'axes.labelcolor':'#c8dff0','xtick.color':'#7fb3d3','ytick.color':'#7fb3d3',
    'text.color':'#e8f0f8','grid.color':'#1e3448','grid.linewidth':0.6,
    'axes.titlecolor':'#e8f0f8','axes.titlesize':11,'axes.titleweight':'bold',
    'legend.facecolor':'#111d2a','legend.edgecolor':'#2a4a6a','legend.labelcolor':'#c8dff0',
})
PAL = ['#4fc3f7','#ff7043','#66bb6a','#ffa726','#ab47bc','#26c6da','#ef5350']

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def kpi(label, value, sub=""):
    return f"<div class='kpi'><div class='lbl'>{label}</div><div class='val'>{value}</div><div class='sub'>{sub}</div></div>"

def sec(text):
    st.markdown(f"<div class='sec'>{text}</div>", unsafe_allow_html=True)

def box(html):
    st.markdown(f"<div class='box'>{html}</div>", unsafe_allow_html=True)

def show_code(code):
    st.markdown(f"<div class='code-cell'>{code}</div>", unsafe_allow_html=True)

def show_output(text):
    st.markdown(f"<div class='out-cell'>{text}</div>", unsafe_allow_html=True)

NC = ['N','P','K','temperature','humidity','ph','rainfall']

# ─── DATA & MODEL (cached) ────────────────────────────────────────────────────
@st.cache_data
def load_data(f):
    return pd.read_csv(f)

def get_default_df():
    import os
    # Cari file CSV otomatis di folder yang sama
    candidates = [
        "Crop_recommendation.csv",
        os.path.join(os.path.dirname(__file__), "Crop_recommendation.csv")
    ]
    for path in candidates:
        if os.path.exists(path):
            return pd.read_csv(path)
    return None

@st.cache_resource
def run_all(df):
    """Jalankan SELURUH pipeline notebook: Clustering → Regression → Classification"""

    # ── STEP 1: Data Identification ──────────────────────────────────
    buf = StringIO()
    buf.write(f"Nama variabel: {df.columns.tolist()}\n")
    buf.write(f"Jumlah variabel: {len(df.columns)}\n")
    buf.write(f"Jumlah baris, kolom (shape): {df.shape}\n")
    buf.write(f"Jumlah record: {df.shape[0]}\n")
    id_output = buf.getvalue()

    info_buf = StringIO()
    df.info(buf=info_buf)
    info_output = info_buf.getvalue()

    # ── STEP 2: Data Preparation ──────────────────────────────────────
    missing = df.isnull().sum()
    duplikat = df.duplicated().sum()

    # ── STEP 3: Data Selection ────────────────────────────────────────
    df_sel = df[NC].copy()

    # ── STEP 4: Scaling ───────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_sel[NC])

    # ── STEP 5: Elbow Method ──────────────────────────────────────────
    inertia_list, sil_list = [], []
    K_range = range(1, 11)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertia_list.append(km.inertia_)
        if k >= 2:
            sil_list.append(silhouette_score(X_scaled, km.labels_))

    # ── STEP 6: K-Means K=2 ──────────────────────────────────────────
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    df_sel['cluster'] = kmeans.fit_predict(X_scaled)
    cluster_counts = df_sel['cluster'].value_counts()
    cluster_mean   = df_sel.groupby('cluster')[NC].mean().round(3)
    centroids      = kmeans.cluster_centers_
    sil_final      = silhouette_score(X_scaled, df_sel['cluster'])

    # ── STEP 7: Regression (Linear) ──────────────────────────────────
    X_reg = df_sel[NC]
    y_reg = df_sel['cluster']
    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    sc_reg = StandardScaler()
    X_tr_rs = sc_reg.fit_transform(X_tr_r)
    X_te_rs  = sc_reg.transform(X_te_r)
    model_reg = LinearRegression()
    model_reg.fit(X_tr_rs, y_tr_r)
    y_pred_reg = model_reg.predict(X_te_rs)
    mae  = mean_absolute_error(y_te_r, y_pred_reg)
    mse  = mean_squared_error(y_te_r, y_pred_reg)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_te_r, y_pred_reg)
    coeff_df = pd.DataFrame(model_reg.coef_, NC, columns=['Coefficient']).round(6)

    # ── STEP 8: Classification — Logistic Regression ──────────────────
    X_clf = df_sel[NC]
    y_clf = df_sel['cluster']
    X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)
    sc_clf = StandardScaler()
    X_tr_cs = sc_clf.fit_transform(X_tr_c)
    X_te_cs  = sc_clf.transform(X_te_c)

    # VIF
    vif_df = None
    if HAS_STATSMODELS:
        vif_vals = [variance_inflation_factor(X_tr_c.values, i) for i in range(len(NC))]
        vif_df = pd.DataFrame({'Fitur': NC, 'VIF': [round(v,4) for v in vif_vals]})
        vif_df['Status'] = vif_df['VIF'].apply(lambda v: '✅ OK (VIF<10)' if v < 10 else '⚠️ Tinggi')

    model_lr = LogisticRegression(random_state=42, max_iter=1000)
    model_lr.fit(X_tr_cs, y_tr_c)
    cv_lr     = cross_val_score(model_lr, X_tr_cs, y_tr_c, cv=5, scoring='accuracy')
    y_pred_lr = model_lr.predict(X_te_cs)
    y_prob_lr = model_lr.predict_proba(X_te_cs)[:,1]
    acc_lr    = accuracy_score(y_te_c, y_pred_lr)
    rep_lr    = classification_report(y_te_c, y_pred_lr, output_dict=True)
    cm_lr     = confusion_matrix(y_te_c, y_pred_lr)
    coef_lr   = pd.DataFrame({'Fitur': NC, 'Koefisien': model_lr.coef_[0]}).sort_values('Koefisien', key=abs, ascending=True)

    # ── STEP 9: Classification — Naïve Bayes ──────────────────────────
    model_nb = GaussianNB()
    model_nb.fit(X_tr_cs, y_tr_c)
    cv_nb     = cross_val_score(model_nb, X_tr_cs, y_tr_c, cv=5, scoring='accuracy')
    y_pred_nb = model_nb.predict(X_te_cs)
    y_prob_nb = model_nb.predict_proba(X_te_cs)[:,1]
    acc_nb    = accuracy_score(y_te_c, y_pred_nb)
    rep_nb    = classification_report(y_te_c, y_pred_nb, output_dict=True)
    cm_nb     = confusion_matrix(y_te_c, y_pred_nb)

    return dict(
        df_sel=df_sel, X_scaled=X_scaled, scaler=scaler,
        id_output=id_output, info_output=info_output,
        missing=missing, duplikat=duplikat,
        inertia_list=list(inertia_list), sil_list=sil_list,
        kmeans=kmeans, cluster_counts=cluster_counts,
        cluster_mean=cluster_mean, centroids=centroids, sil_final=sil_final,
        # regression
        model_reg=model_reg, sc_reg=sc_reg,
        y_te_r=y_te_r, y_pred_reg=y_pred_reg,
        mae=mae, mse=mse, rmse=rmse, r2=r2, coeff_df=coeff_df,
        # classification
        model_lr=model_lr, model_nb=model_nb,
        sc_clf=sc_clf, vif_df=vif_df,
        X_te_cs=X_te_cs, y_te_c=y_te_c,
        y_pred_lr=y_pred_lr, y_prob_lr=y_prob_lr,
        y_pred_nb=y_pred_nb, y_prob_nb=y_prob_nb,
        acc_lr=acc_lr, acc_nb=acc_nb,
        cv_lr=cv_lr, cv_nb=cv_nb,
        rep_lr=rep_lr, rep_nb=rep_nb,
        cm_lr=cm_lr, cm_nb=cm_nb, coef_lr=coef_lr,
    )

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:18px 0 10px'>
        <div style='font-size:38px'>🌾</div>
        <div style='font-size:15px;font-weight:800;color:#4fc3f7;margin-top:6px'>Crop Land Analysis</div>
        <div style='font-size:10px;color:#5a8aaa;letter-spacing:1px;margin-top:3px'>KELOMPOK 9 — SI4801</div>
    </div><hr>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("📂 Upload CSV (opsional — sudah auto-load)", type=['csv'])
    if uploaded is None:
        st.markdown("<div style='font-size:11px;color:#4fc3f7;margin-top:4px'>✅ Menggunakan dataset bawaan</div>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    page = st.radio("", [
        "🏠  Overview",
        "📓  Notebook",
        "📊  EDA",
        "🔵  Clustering",
        "📉  Regresi",
        "📈  Klasifikasi",
        "🎯  Prediksi",
    ], label_visibility="collapsed")

    st.markdown("""<hr>
    <div style='font-size:11px;color:#3a6a8a;line-height:1.9'>
        <b style='color:#5a8aaa'>Algoritma</b><br>
        • K-Means Clustering<br>
        • Linear Regression<br>
        • Logistic Regression<br>
        • Naïve Bayes Classifier<br><br>
        <b style='color:#5a8aaa'>Dataset</b><br>
        Crop Recommendation<br>
        <span style='color:#4fc3f7'>2200 records × 8 fitur</span>
    </div>""", unsafe_allow_html=True)

# ─── LOAD DATA: pakai upload atau fallback ke file lokal ──────────────────────
if uploaded is not None:
    df = load_data(uploaded)
else:
    df = get_default_df()
    if df is None:
        st.markdown("""
        <div style='text-align:center;padding:80px 20px'>
            <div style='font-size:68px;margin-bottom:20px'>🌾</div>
            <h1 style='font-size:30px;font-weight:800;color:#e8f0f8;margin-bottom:10px'>
                Dashboard Penambangan Data</h1>
            <p style='font-size:15px;color:#7fb3d3;max-width:460px;margin:0 auto 28px'>
                Kelompok 9 — SI4801 | Semester Genap TA 2025/2026</p>
            <div style='margin-bottom:32px'>
                <span class='tag'>K-Means Clustering</span>
                <span class='tag'>Linear Regression</span>
                <span class='tag'>Logistic Regression</span>
                <span class='tag'>Naïve Bayes</span>
            </div>
            <div style='background:rgba(79,195,247,.08);border:1px solid rgba(79,195,247,.25);
                        border-radius:14px;padding:18px 28px;display:inline-block;font-size:14px;color:#7fb3d3'>
                👈 Pastikan <b style='color:#4fc3f7'>Crop_recommendation.csv</b> ada di folder yang sama dengan app_final.py,<br>
                atau upload manual lewat sidebar.
            </div>
        </div>""", unsafe_allow_html=True)
        st.stop()

m   = run_all(df)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Overview":
    st.markdown("<h1 style='font-size:26px;font-weight:800;color:#e8f0f8;margin-bottom:4px'>🌾 Dashboard Penambangan Data — Kelompok 9</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#5a8aaa;margin-bottom:24px'>Analisis Lahan Pertanian | SI4801 | Semester Genap TA 2025/2026 | Universitas Telkom</p>", unsafe_allow_html=True)

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    for col,lbl,val,sub in zip([c1,c2,c3,c4,c5,c6],[
        "Total Record","Fitur Numerik","Jenis Tanaman","Akurasi LR","Akurasi NB","Silhouette Score"],[
        f"{len(df):,}","7",str(df['label'].nunique()),
        f"{m['acc_lr']*100:.1f}%",f"{m['acc_nb']*100:.1f}%",f"{m['sil_final']:.4f}"],[
        "baris data","variabel input","kelas unik","Logistic Reg","Naïve Bayes","K-Means K=2"]):
        with col:
            st.markdown(kpi(lbl,val,sub), unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    ca, cb = st.columns([1.2,1])

    with ca:
        sec("📋 Latar Belakang Penelitian")
        box("""Pertanian merupakan sektor vital yang sangat bergantung pada kondisi tanah dan iklim.
Penelitian ini menganalisis hubungan antara kandungan unsur hara tanah (N, P, K) dan faktor lingkungan
(suhu, kelembapan, pH, curah hujan) dengan karakteristik lahan menggunakan metode <b>Penambangan Data</b>.<br><br>
<b>Rumusan Masalah:</b><br>
1. Bagaimana pengelompokan lahan berdasarkan unsur hara & iklim? (<i>K-Means</i>)<br>
2. Seberapa baik model <i>Logistic Regression</i> mengklasifikasikan kelompok lahan?<br>
3. Seberapa baik model <i>Naïve Bayes</i> mengklasifikasikan kelompok lahan?<br>
4. Bagaimana perbandingan performa kedua model klasifikasi?""")

        sec("📌 Variabel Dataset")
        var_df = pd.DataFrame({
            'Variabel': NC + ['label'],
            'Deskripsi': ['Kandungan Nitrogen','Kandungan Fosfor','Kandungan Kalium',
                          'Suhu lingkungan','Kelembapan udara','Tingkat keasaman tanah',
                          'Curah hujan','Jenis tanaman rekomendasi'],
            'Satuan': ['mg/kg','mg/kg','mg/kg','°C','%','-','mm','Kategorik'],
            'Tipe': ['Numerik']*7+['Kategorik']
        })
        st.dataframe(var_df, use_container_width=True, hide_index=True)

    with cb:
        sec("🔬 Alur CRISP-DM")
        for n,title,desc in [
            ("1","Business Understanding","Identifikasi masalah & tujuan analisis"),
            ("2","Data Understanding","Eksplorasi & statistik deskriptif dataset"),
            ("3","Data Preparation","Cleaning, outlier, normalisasi fitur"),
            ("4","Modeling","K-Means, Linear Reg, Logistic Reg, Naïve Bayes"),
            ("5","Evaluation","Accuracy, F1, ROC-AUC, R², MAE, RMSE"),
            ("6","Deployment","Dashboard interaktif Streamlit"),
        ]:
            st.markdown(f"""
            <div style='display:flex;align-items:flex-start;gap:12px;margin-bottom:12px'>
                <div style='min-width:28px;height:28px;border-radius:50%;
                            background:linear-gradient(135deg,#1565C0,#0d47a1);
                            display:flex;align-items:center;justify-content:center;
                            font-size:12px;font-weight:800;color:#4fc3f7'>{n}</div>
                <div>
                    <div style='font-size:13px;font-weight:700;color:#e8f0f8'>{title}</div>
                    <div style='font-size:11px;color:#5a8aaa;margin-top:1px'>{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)

        sec("👀 Preview Data (5 baris pertama)")
        st.dataframe(df.head(), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: NOTEBOOK (menampilkan kode + output persis seperti Jupyter)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📓  Notebook":
    st.markdown("<h2 style='font-size:24px;font-weight:800;color:#e8f0f8;margin-bottom:4px'>📓 Notebook — Kode Lengkap</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#5a8aaa;margin-bottom:20px'>Seluruh kode notebook ditampilkan di sini beserta output-nya, persis seperti di Jupyter/Colab.</p>", unsafe_allow_html=True)

    # ── Cell 0: Deskripsi ──────────────────────────────────────────────────────
    st.markdown("**`[MD]`** Deskripsi Penelitian")
    box("""Penelitian ini bertujuan untuk melihat hubungan antara kondisi unsur hara tanah dan faktor lingkungan/iklim
dengan karakteristik kelompok lahan pertanian. Fokusnya adalah untuk mengetahui apakah faktor seperti kandungan
Nitrogen (N), Fosfor (P), Kalium (K), suhu, kelembapan, tingkat pH tanah, dan curah hujan berpengaruh secara
signifikan dalam membentuk pengelompokan (clustering) lahan serta memprediksi pola klasifikasi dan regresi
dari karakteristik lahan tersebut.""")

    # ── Load Data ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**`[1]`** Load Data")
    show_code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("Crop_recommendation.csv")
df.head()""")
    st.dataframe(df.head(), use_container_width=True, hide_index=True)

    # ── Data Identification ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**`[MD]`** # Data Identification")
    st.markdown("**`[2]`** Nama variabel dan jumlah variabel")
    show_code("""print("Nama variabel dalam dataset:")
print(df.columns.tolist())
print("\\nJumlah variabel :", len(df.columns))""")
    show_output(m['id_output'])

    st.markdown("**`[3]`** Jumlah baris dan kolom")
    show_code("""print("Jumlah baris, kolom (shape) :", df.shape)
print(f"Jumlah record : {df.shape[0]}")""")
    show_output(f"Jumlah baris, kolom (shape) : {df.shape}\nJumlah record : {df.shape[0]}")

    st.markdown("**`[4]`** Informasi lengkap & tipe data")
    show_code("df.info()")
    show_output(m['info_output'])

    st.markdown("**`[5]`** Statistik Deskriptif")
    show_code("df.describe()")
    st.dataframe(df[NC].describe().round(3), use_container_width=True)

    # ── Data Preparation ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**`[MD]`** # Data Preparation")
    st.markdown("**`[6]`** Mengecek missing value")
    show_code("df.isnull().sum()")
    show_output(str(m['missing'].to_string()))

    st.markdown("**`[7]`** Mengecek duplikat")
    show_code("df.duplicated().sum()")
    show_output(f"{m['duplikat']}")

    st.markdown("**`[8]`** Deteksi Outlier (IQR Method)")
    show_code("""for col in ['N','P','K','temperature','humidity','ph','rainfall']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    n_out = df[(df[col] < lower) | (df[col] > upper)].shape[0]
    print(f"{col:15s}: {n_out} outlier | Batas: [{lower:.2f}, {upper:.2f}]")""")
    out_lines = []
    for col in NC:
        Q1,Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3-Q1
        lb,ub = Q1-1.5*IQR, Q3+1.5*IQR
        n_out = df[(df[col]<lb)|(df[col]>ub)].shape[0]
        out_lines.append(f"{col:15s}: {n_out:4d} outlier  |  Batas: [{lb:.2f}, {ub:.2f}]")
    show_output("\n".join(out_lines))

    # ── Data Selection ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**`[MD]`** # Data Selection")
    st.markdown("**`[9]`** Memilih variabel numerik")
    show_code("""df_selected = df[['N','P','K','temperature','humidity','ph','rainfall']].copy()
df_selected.head()""")
    st.dataframe(m['df_sel'][NC].head(), use_container_width=True, hide_index=True)

    # ── Clustering ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**`[MD]`** # Clustering")
    st.markdown("**`[10]`** Scaling data")
    show_code("""from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_selected[['N','P','K','temperature','humidity','ph','rainfall']])""")
    show_output("Data berhasil di-scaling menggunakan StandardScaler.")

    st.markdown("**`[11]`** Elbow Method")
    show_code("""from sklearn.cluster import KMeans
inertia = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.plot(range(1,11), inertia, marker='o', color='b')
plt.title('Elbow Method')
plt.xlabel('Jumlah Cluster')
plt.ylabel('Inertia')
plt.show()""")
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(range(1,11), m['inertia_list'], marker='o', color='#4fc3f7', lw=2, ms=7)
    ax.axvline(2, color='#ff7043', ls='--', lw=1.5, label='K=2 dipilih')
    ax.set_title('Elbow Method'); ax.set_xlabel('Jumlah Cluster'); ax.set_ylabel('Inertia')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("**`[12]`** Silhouette Score")
    show_code("""from sklearn.metrics import silhouette_score
sil_scores = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))

plt.plot(range(2,11), sil_scores, marker='s', color='darkorange')
plt.title('Silhouette Score per K')
plt.xlabel('K'); plt.ylabel('Silhouette Score')
plt.show()""")
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(range(2,11), m['sil_list'], marker='s', color='#ff7043', lw=2, ms=7)
    ax.axvline(2, color='#4fc3f7', ls='--', lw=1.5, label='K=2 dipilih')
    ax.set_title('Silhouette Score per K'); ax.set_xlabel('K'); ax.set_ylabel('Silhouette Score')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("**`[13]`** K-Means K=2 + distribusi cluster")
    show_code("""kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df_selected['cluster'] = kmeans.fit_predict(X_scaled)
print(df_selected['cluster'].value_counts())""")
    show_output(str(m['cluster_counts'].to_string()))

    st.markdown("**`[14]`** Rata-rata fitur per cluster")
    show_code("df_selected.groupby('cluster')[['N','P','K','temperature','humidity','ph','rainfall']].mean()")
    st.dataframe(m['cluster_mean'], use_container_width=True)

    st.markdown("**`[15]`** Visualisasi Cluster & Centroid")
    show_code("""centroids = kmeans.cluster_centers_
plt.figure(figsize=(8,6))
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=df_selected['cluster'], cmap='viridis', alpha=0.5)
plt.scatter(centroids[:,0], centroids[:,1], s=200, marker='X', color='red', label='Centroids')
plt.xlabel('Feature 1 (N - Scaled)'); plt.ylabel('Feature 2 (P - Scaled)')
plt.title('Cluster dan Centroid (Dataset Lahan)')
plt.legend(); plt.show()""")
    fig, ax = plt.subplots(figsize=(8,5))
    Xs = m['X_scaled']; ds = m['df_sel']
    for c,col,nm in [(0,'#4fc3f7','Cluster 0'),(1,'#ff7043','Cluster 1')]:
        msk = ds['cluster']==c
        ax.scatter(Xs[msk,0], Xs[msk,1], c=col, label=nm, alpha=0.45, s=18)
    ax.scatter(m['centroids'][:,0], m['centroids'][:,1], s=250, marker='X',
               c='white', zorder=10, edgecolors='black', lw=0.8, label='Centroids')
    ax.set_xlabel('Feature 1 (N - Scaled)'); ax.set_ylabel('Feature 2 (P - Scaled)')
    ax.set_title('Cluster dan Centroid (Dataset Lahan)')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    # ── Regression ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**`[MD]`** ## Regression — Memprediksi label target dari hasil clustering")
    st.markdown("**`[16]`** Train-Test Split + Scaling")
    show_code("""X = df_selected[['N','P','K','temperature','humidity','ph','rainfall']]
y = df_selected['cluster']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler_reg = StandardScaler()
X_train_scaled = scaler_reg.fit_transform(X_train)
X_test_scaled  = scaler_reg.transform(X_test)""")
    show_output(f"Training set: {int(len(df)*0.8)} data | Testing set: {int(len(df)*0.2)} data")

    st.markdown("**`[17]`** Linear Regression + Evaluasi")
    show_code("""from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

model_reg = LinearRegression()
model_reg.fit(X_train_scaled, y_train)
y_pred_reg = model_reg.predict(X_test_scaled)

print("MAE      :", mean_absolute_error(y_test, y_pred_reg))
print("MSE      :", mean_squared_error(y_test, y_pred_reg))
print("RMSE     :", np.sqrt(mean_squared_error(y_test, y_pred_reg)))
print("R2 Score :", r2_score(y_test, y_pred_reg))""")
    show_output(f"MAE      : {m['mae']:.6f}\nMSE      : {m['mse']:.6f}\nRMSE     : {m['rmse']:.6f}\nR2 Score : {m['r2']:.6f}")

    st.markdown("**`[18]`** Visualisasi Actual vs Predicted (Regression)")
    show_code("""plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_reg, color='orange', alpha=0.4)
plt.xlabel("Actual Cluster Label"); plt.ylabel("Predicted Cluster Value")
plt.title("Actual vs Predicted (Regression)")
plt.show()""")
    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(m['y_te_r'], m['y_pred_reg'], color='#ffa726', alpha=0.4, s=20)
    ax.set_xlabel("Actual Cluster Label"); ax.set_ylabel("Predicted Cluster Value")
    ax.set_title("Actual vs Predicted (Regression)"); ax.grid(True, alpha=0.3)
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("**`[19]`** Feature Importance (Koefisien Regresi)")
    show_code("coeff_df = pd.DataFrame(model_reg.coef_, X.columns, columns=['Coefficient'])\nprint(coeff_df)")
    show_output(m['coeff_df'].to_string())

    # ── Classification ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**`[MD]`** ## Classification — Logistic Regression")
    st.markdown("**`[20]`** Logistic Regression Training")
    show_code("""from sklearn.linear_model import LogisticRegression
model_clf = LogisticRegression(random_state=42)
model_clf.fit(X_train_clf_scaled, y_train_clf)
y_pred_clf = model_clf.predict(X_test_clf_scaled)""")
    show_output(f"Model berhasil dilatih.\nHasil prediksi (10 pertama): {list(m['y_pred_lr'][:10])}")

    st.markdown("**`[21]`** Evaluasi Logistic Regression")
    show_code("""from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print("Accuracy :", accuracy_score(y_test_clf, y_pred_clf))
print(classification_report(y_test_clf, y_pred_clf))
print(confusion_matrix(y_test_clf, y_pred_clf))""")
    show_output(f"Accuracy : {m['acc_lr']:.4f}\n\n"
                f"Classification Report:\n{classification_report(m['y_te_c'], m['y_pred_lr'])}\n\n"
                f"Confusion Matrix:\n{m['cm_lr']}")

    st.markdown("""<div class='box'>
    <b>✅ Analisis Hasil Klasifikasi (Logistic Regression)</b><br><br>
    1. <b>Akurasi Sempurna (100%):</b> Model berhasil memprediksi seluruh kelompok cluster data uji dengan tepat. Nilai Precision, Recall, dan F1-Score untuk kedua cluster mencapai angka maksimal 1.00.<br><br>
    2. <b>Interpretasi Confusion Matrix:</b><br>
    • Baris 1: Seluruh data Cluster 0 berhasil diprediksi dengan benar (0 kesalahan).<br>
    • Baris 2: Seluruh data Cluster 1 berhasil diprediksi dengan benar (0 kesalahan).<br><br>
    <b>Kesimpulan:</b> Akurasi sempurna terjadi karena label target (cluster) dibuat langsung secara matematis
    oleh K-Means dari fitur yang sama. Logistic Regression berhasil mempelajari decision boundary
    antar cluster dengan sangat baik.
    </div>""", unsafe_allow_html=True)

    # Naïve Bayes
    st.markdown("---")
    st.markdown("**`[MD]`** ## Classification — Naïve Bayes Classifier")
    st.markdown("**`[22]`** Naïve Bayes Training + Evaluasi")
    show_code("""from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

model_nb = GaussianNB()
model_nb.fit(X_train_clf_scaled, y_train_clf)
y_pred_nb = model_nb.predict(X_test_clf_scaled)

print("Accuracy :", accuracy_score(y_test_clf, y_pred_nb))
print(classification_report(y_test_clf, y_pred_nb))
print("CV 5-fold:", cross_val_score(model_nb, X_train_clf_scaled, y_train_clf, cv=5).mean())""")
    show_output(f"Accuracy : {m['acc_nb']:.4f}\n\n"
                f"Classification Report:\n{classification_report(m['y_te_c'], m['y_pred_nb'])}\n\n"
                f"CV 5-fold: {m['cv_nb'].mean():.4f}")

    st.markdown("**`[23]`** Confusion Matrix Naïve Bayes")
    show_code("print(confusion_matrix(y_test_clf, y_pred_nb))")
    show_output(str(m['cm_nb']))

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  EDA":
    st.markdown("<h2 style='font-size:24px;font-weight:800;color:#e8f0f8;margin-bottom:4px'>📊 Exploratory Data Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#5a8aaa;margin-bottom:20px'>Eksplorasi statistik dan visualisasi distribusi dataset lahan pertanian</p>", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col,lbl,val,sub in zip([c1,c2,c3,c4],
        ["Missing Value","Duplikat","Total Fitur","Kelas Tanaman"],
        ["0",str(m['duplikat']),str(len(df.columns)),str(df['label'].nunique())],
        ["tidak ada","baris","variabel","jenis unik"]):
        with col: st.markdown(kpi(lbl,val,sub), unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    t1,t2,t3,t4 = st.tabs(["📈 Distribusi","🔥 Korelasi","📦 Outlier","🌿 Tanaman"])

    with t1:
        sec("Distribusi Fitur Numerik")
        fig,axes = plt.subplots(2,4,figsize=(18,9)); axes=axes.flatten()
        for i,col in enumerate(NC):
            axes[i].hist(df[col],bins=35,color=PAL[i],edgecolor='none',alpha=0.85)
            axes[i].axvline(df[col].mean(),color='#ffd54f',ls='--',lw=1.5,label=f'μ={df[col].mean():.1f}')
            axes[i].axvline(df[col].median(),color='#ef9a9a',ls=':',lw=1.5,label=f'Med={df[col].median():.1f}')
            axes[i].set_title(col,fontweight='bold'); axes[i].legend(fontsize=8); axes[i].grid(True,alpha=0.3)
        axes[-1].axis('off')
        plt.suptitle('Distribusi Variabel Numerik',fontsize=13,fontweight='bold',y=1.01)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
        sec("Statistik Deskriptif"); st.dataframe(df[NC].describe().round(3), use_container_width=True)

    with t2:
        sec("Heatmap Korelasi Pearson")
        fig,ax = plt.subplots(figsize=(9,7))
        corr = df[NC].corr()
        mask = np.triu(np.ones_like(corr,dtype=bool))
        sns.heatmap(corr,ax=ax,annot=True,fmt='.2f',cmap='RdYlGn',mask=mask,
                    vmin=-1,vmax=1,linewidths=0.5,annot_kws={"size":11,"weight":"bold"})
        ax.set_title('Korelasi Antar Variabel Numerik',fontweight='bold',pad=14)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
        box("Korelasi antar fitur yang rendah mendukung asumsi independensi pada Naïve Bayes, dan tidak ada multikolinearitas serius untuk Logistic Regression.")

    with t3:
        sec("Deteksi Outlier — Boxplot")
        fig,axes = plt.subplots(2,4,figsize=(18,9)); axes=axes.flatten()
        for i,col in enumerate(NC):
            axes[i].boxplot(df[col],patch_artist=True,
                            boxprops=dict(facecolor=PAL[i],alpha=0.6),
                            medianprops=dict(color='white',linewidth=2.5),
                            whiskerprops=dict(color='#7fb3d3'),capprops=dict(color='#7fb3d3'),
                            flierprops=dict(marker='o',color=PAL[i],alpha=0.4,markersize=3))
            axes[i].set_title(col,fontweight='bold'); axes[i].grid(True,alpha=0.3)
        axes[-1].axis('off')
        plt.suptitle('Deteksi Outlier dengan Boxplot',fontsize=13,fontweight='bold',y=1.01)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
        out_data = []
        for col in NC:
            Q1,Q3=df[col].quantile(0.25),df[col].quantile(0.75); IQR=Q3-Q1
            lb,ub=Q1-1.5*IQR,Q3+1.5*IQR
            n_out=df[(df[col]<lb)|(df[col]>ub)].shape[0]
            out_data.append({'Fitur':col,'Q1':round(Q1,2),'Q3':round(Q3,2),'IQR':round(IQR,2),
                             'Batas Bawah':round(lb,2),'Batas Atas':round(ub,2),'Outlier':n_out})
        st.dataframe(pd.DataFrame(out_data), use_container_width=True, hide_index=True)
        box("ℹ️ Outlier <b>tidak dihapus</b> karena nilai ekstrem pada data pertanian dapat merepresentasikan kondisi lahan nyata yang valid.")

    with t4:
        sec("Distribusi Jenis Tanaman")
        lc = df['label'].value_counts()
        fig,ax = plt.subplots(figsize=(14,6))
        colors_bar = plt.cm.tab20(np.linspace(0,1,len(lc)))
        bars = ax.bar(lc.index,lc.values,color=colors_bar,edgecolor='none')
        for bar,val in zip(bars,lc.values):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.3,str(val),
                    ha='center',va='bottom',fontsize=8,color='#c8dff0')
        ax.set_xlabel('Jenis Tanaman'); ax.set_ylabel('Jumlah Data')
        ax.set_title(f'Distribusi {df["label"].nunique()} Jenis Tanaman',fontweight='bold')
        ax.set_xticklabels(lc.index,rotation=45,ha='right',fontsize=9)
        ax.grid(True,axis='y',alpha=0.3)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔵  Clustering":
    st.markdown("<h2 style='font-size:24px;font-weight:800;color:#e8f0f8;margin-bottom:4px'>🔵 K-Means Clustering</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#5a8aaa;margin-bottom:20px'>Pengelompokan lahan pertanian berdasarkan kandungan unsur hara dan kondisi iklim</p>", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    cc = m['cluster_counts']
    for col,lbl,val,sub in zip([c1,c2,c3,c4],
        ["Jumlah Cluster","Silhouette Score","Cluster 0","Cluster 1"],
        ["2",f"{m['sil_final']:.4f}",f"{cc.get(0,0):,}",f"{cc.get(1,0):,}"],
        ["K optimal","> 0.5 = baik",f"{cc.get(0,0)/len(m['df_sel'])*100:.1f}% data",f"{cc.get(1,0)/len(m['df_sel'])*100:.1f}% data"]):
        with col: st.markdown(kpi(lbl,val,sub), unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    t1,t2,t3 = st.tabs(["📐 K Optimal","🗺️ Visualisasi","📋 Profil Cluster"])

    with t1:
        sec("Elbow Method & Silhouette Score")
        fig,(ax1,ax2) = plt.subplots(1,2,figsize=(14,5))
        ax1.plot(range(1,11),m['inertia_list'],marker='o',color='#4fc3f7',lw=2.5,ms=8)
        ax1.fill_between(range(1,11),m['inertia_list'],alpha=0.1,color='#4fc3f7')
        ax1.axvline(2,color='#ff7043',ls='--',lw=1.5,label='K=2 dipilih')
        ax1.set_title('Elbow Method',fontweight='bold'); ax1.set_xlabel('K'); ax1.set_ylabel('Inertia')
        ax1.legend(); ax1.grid(True,alpha=0.3)
        ax2.plot(range(2,11),m['sil_list'],marker='s',color='#ff7043',lw=2.5,ms=8)
        ax2.fill_between(range(2,11),m['sil_list'],alpha=0.1,color='#ff7043')
        ax2.axvline(2,color='#4fc3f7',ls='--',lw=1.5,label='K=2 dipilih')
        ax2.set_title('Silhouette Score per K',fontweight='bold'); ax2.set_xlabel('K'); ax2.set_ylabel('Score')
        ax2.legend(); ax2.grid(True,alpha=0.3)
        plt.suptitle('Penentuan Jumlah Cluster Optimal',fontsize=13,fontweight='bold')
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

        sil_df = pd.DataFrame({'K':list(range(2,11)),'Silhouette Score':[round(s,4) for s in m['sil_list']],
                               'Inertia':[round(m['inertia_list'][k],2) for k in range(1,10)]})
        sil_df['Dipilih'] = sil_df['K'].apply(lambda k: '✅' if k==2 else '')
        st.dataframe(sil_df, use_container_width=True, hide_index=True)

    with t2:
        sec("Scatter Plot Cluster")
        cx = st.selectbox("Sumbu X", NC, index=0)
        cy = st.selectbox("Sumbu Y", NC, index=1)
        ci,cj = NC.index(cx), NC.index(cy)
        Xs = m['X_scaled']; ds = m['df_sel']; cents = m['centroids']
        fig,axes = plt.subplots(1,2,figsize=(14,5))
        for c,color,nm in [(0,'#4fc3f7','Cluster 0'),(1,'#ff7043','Cluster 1')]:
            msk = ds['cluster']==c
            axes[0].scatter(Xs[msk,ci],Xs[msk,cj],c=color,label=nm,alpha=0.45,s=18)
            axes[1].scatter(Xs[msk,2],Xs[msk,6],c=color,label=nm,alpha=0.45,s=18)
        for ax in axes:
            ax.scatter(cents[:,ci if ax==axes[0] else 2],cents[:,cj if ax==axes[0] else 6],
                       s=280,marker='X',c='white',zorder=10,edgecolors='black',lw=0.7,label='Centroid')
            ax.legend(); ax.grid(True,alpha=0.3)
        axes[0].set_xlabel(f'{cx} (Scaled)'); axes[0].set_ylabel(f'{cy} (Scaled)')
        axes[0].set_title(f'Cluster: {cx} vs {cy}',fontweight='bold')
        axes[1].set_xlabel('K (Scaled)'); axes[1].set_ylabel('Rainfall (Scaled)')
        axes[1].set_title('Cluster: K vs Rainfall',fontweight='bold')
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with t3:
        sec("Profil Rata-rata per Cluster")
        st.dataframe(m['cluster_mean'], use_container_width=True)
        profile = m['cluster_mean']
        fig,ax = plt.subplots(figsize=(12,5))
        x=np.arange(len(NC)); w=0.35
        b1=ax.bar(x-w/2,[profile.loc[0,c] for c in NC],w,label='Cluster 0',color='#4fc3f7',alpha=0.85)
        b2=ax.bar(x+w/2,[profile.loc[1,c] for c in NC],w,label='Cluster 1',color='#ff7043',alpha=0.85)
        for bar in list(b1)+list(b2):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.3,f'{bar.get_height():.1f}',
                    ha='center',va='bottom',fontsize=8,color='#c8dff0')
        ax.set_xticks(x); ax.set_xticklabels(NC)
        ax.set_title('Rata-rata Fitur per Cluster',fontweight='bold')
        ax.set_ylabel('Nilai Rata-rata'); ax.legend(); ax.grid(True,axis='y',alpha=0.3)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
        c1,c2=st.columns(2)
        with c1: box("<b>🔵 Cluster 0 — Lahan Standar:</b><br>Kadar N, P, K relatif lebih rendah. Umumnya cocok untuk tanaman yang tidak membutuhkan nutrisi tanah berlebih.")
        with c2: box("<b>🟠 Cluster 1 — Lahan Kaya Nutrisi:</b><br>Kadar K dan P lebih tinggi secara signifikan. Sesuai untuk tanaman buah-buahan dan tanaman yang membutuhkan Kalium tinggi.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: REGRESI
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📉  Regresi":
    st.markdown("<h2 style='font-size:24px;font-weight:800;color:#e8f0f8;margin-bottom:4px'>📉 Linear Regression</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#5a8aaa;margin-bottom:20px'>Prediksi nilai cluster menggunakan regresi linier</p>", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col,lbl,val,sub in zip([c1,c2,c3,c4],
        ["MAE","MSE","RMSE","R² Score"],
        [f"{m['mae']:.6f}",f"{m['mse']:.6f}",f"{m['rmse']:.6f}",f"{m['r2']:.6f}"],
        ["Mean Abs Error","Mean Sq Error","Root MSE","Koefisien Determinasi"]):
        with col: st.markdown(kpi(lbl,val,sub), unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1:
        sec("Actual vs Predicted")
        fig,ax=plt.subplots(figsize=(7,5))
        ax.scatter(m['y_te_r'],m['y_pred_reg'],color='#ffa726',alpha=0.4,s=20)
        ax.set_xlabel("Actual Cluster Label"); ax.set_ylabel("Predicted Cluster Value")
        ax.set_title("Actual vs Predicted (Regression)",fontweight='bold'); ax.grid(True,alpha=0.3)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
    with c2:
        sec("Koefisien Regresi (Feature Importance)")
        cf = m['coeff_df'].reset_index().rename(columns={'index':'Fitur'})
        fig,ax=plt.subplots(figsize=(7,5))
        colors_bar=['#ff7043' if v>0 else '#4fc3f7' for v in cf['Coefficient']]
        ax.barh(cf['Fitur'],cf['Coefficient'],color=colors_bar,alpha=0.85)
        ax.axvline(0,color='white',lw=0.8,alpha=0.5)
        ax.set_title('Koefisien Regresi',fontweight='bold'); ax.set_xlabel('Nilai Koefisien')
        ax.grid(True,axis='x',alpha=0.3)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    sec("Tabel Koefisien Regresi")
    st.dataframe(m['coeff_df'], use_container_width=True)
    box(f"""<b>Interpretasi:</b><br>
    • MAE = <b>{m['mae']:.6f}</b> — rata-rata selisih absolut prediksi sangat kecil mendekati 0<br>
    • R² = <b>{m['r2']:.6f}</b> — model mampu menjelaskan hampir seluruh variansi data<br>
    • Koefisien positif (merah) mendorong prediksi ke Cluster 1, negatif (biru) ke Cluster 0""")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: KLASIFIKASI
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈  Klasifikasi":
    st.markdown("<h2 style='font-size:24px;font-weight:800;color:#e8f0f8;margin-bottom:4px'>📈 Model Klasifikasi</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#5a8aaa;margin-bottom:20px'>Evaluasi dan perbandingan Logistic Regression vs Naïve Bayes Classifier</p>", unsafe_allow_html=True)

    c1,c2,c3,c4=st.columns(4)
    for col,lbl,val,sub in zip([c1,c2,c3,c4],
        ["Accuracy LR","Accuracy NB","CV Score LR","CV Score NB"],
        [f"{m['acc_lr']*100:.2f}%",f"{m['acc_nb']*100:.2f}%",
         f"{m['cv_lr'].mean()*100:.2f}%",f"{m['cv_nb'].mean()*100:.2f}%"],
        ["Logistic Regression","Naïve Bayes","5-Fold CV","5-Fold CV"]):
        with col: st.markdown(kpi(lbl,val,sub), unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    t1,t2,t3,t4=st.tabs(["🔷 Logistic Regression","🟠 Naïve Bayes","⚖️ Perbandingan","📉 ROC Curve"])

    with t1:
        sec("Uji Asumsi VIF")
        if m['vif_df'] is not None:
            st.dataframe(m['vif_df'], use_container_width=True, hide_index=True)
        else:
            box("statsmodels tidak tersedia. Install: pip install statsmodels")
        sec("Confusion Matrix & Feature Importance")
        c1,c2=st.columns(2)
        with c1:
            fig,ax=plt.subplots(figsize=(6,5))
            sns.heatmap(m['cm_lr'],annot=True,fmt='d',cmap='Blues',ax=ax,
                        xticklabels=['C0','C1'],yticklabels=['C0','C1'],
                        annot_kws={'size':18,'weight':'bold'},linewidths=2,linecolor='#0f1923')
            ax.set_title('Confusion Matrix\nLogistic Regression',fontweight='bold')
            ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
        with c2:
            fig,ax=plt.subplots(figsize=(6,5))
            cf=m['coef_lr']
            colors_bar=['#ff7043' if v>0 else '#4fc3f7' for v in cf['Koefisien']]
            ax.barh(cf['Fitur'],cf['Koefisien'],color=colors_bar,alpha=0.85)
            ax.axvline(0,color='white',lw=0.8,alpha=0.5)
            ax.set_title('Feature Importance\n(Koefisien LR)',fontweight='bold')
            ax.set_xlabel('Nilai Koefisien'); ax.grid(True,axis='x',alpha=0.3)
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
        sec("Classification Report")
        rep_df=pd.DataFrame(m['rep_lr']).T.round(4)
        rep_df=rep_df[rep_df.index.isin(['0','1','macro avg','weighted avg'])]
        rep_df.index=['Cluster 0','Cluster 1','Macro Avg','Weighted Avg']
        st.dataframe(rep_df[['precision','recall','f1-score','support']], use_container_width=True)

    with t2:
        sec("Confusion Matrix & Distribusi Probabilitas")
        c1,c2=st.columns(2)
        with c1:
            fig,ax=plt.subplots(figsize=(6,5))
            sns.heatmap(m['cm_nb'],annot=True,fmt='d',cmap='Oranges',ax=ax,
                        xticklabels=['C0','C1'],yticklabels=['C0','C1'],
                        annot_kws={'size':18,'weight':'bold'},linewidths=2,linecolor='#0f1923')
            ax.set_title('Confusion Matrix\nNaïve Bayes',fontweight='bold')
            ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
        with c2:
            fig,ax=plt.subplots(figsize=(6,5))
            ax.scatter(range(len(m['y_prob_nb'])),m['y_prob_nb'],
                       c=['#ff7043' if p==1 else '#4fc3f7' for p in m['y_pred_nb']],alpha=0.4,s=12)
            ax.axhline(0.5,color='white',ls='--',lw=1.2,label='Threshold 0.5')
            ax.set_title('Distribusi Probabilitas\nP(Cluster 1)',fontweight='bold')
            ax.set_xlabel('Data Index'); ax.set_ylabel('Probabilitas'); ax.legend()
            ax.grid(True,alpha=0.3)
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
        sec("Classification Report")
        rep_df2=pd.DataFrame(m['rep_nb']).T.round(4)
        rep_df2=rep_df2[rep_df2.index.isin(['0','1','macro avg','weighted avg'])]
        rep_df2.index=['Cluster 0','Cluster 1','Macro Avg','Weighted Avg']
        st.dataframe(rep_df2[['precision','recall','f1-score','support']], use_container_width=True)

    with t3:
        sec("Perbandingan Metrik Evaluasi")
        rl,rn=m['rep_lr'],m['rep_nb']
        cats=['Accuracy','Precision\n(Macro)','Recall\n(Macro)','F1-Score\n(Macro)','CV Score']
        lr_v=[m['acc_lr'],rl['macro avg']['precision'],rl['macro avg']['recall'],rl['macro avg']['f1-score'],m['cv_lr'].mean()]
        nb_v=[m['acc_nb'],rn['macro avg']['precision'],rn['macro avg']['recall'],rn['macro avg']['f1-score'],m['cv_nb'].mean()]
        fig,ax=plt.subplots(figsize=(12,5))
        x=np.arange(len(cats)); w=0.35
        b1=ax.bar(x-w/2,lr_v,w,label='Logistic Regression',color='#4fc3f7',alpha=0.85)
        b2=ax.bar(x+w/2,nb_v,w,label='Naïve Bayes',color='#ff7043',alpha=0.85)
        for bar in list(b1)+list(b2):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.003,
                    f'{bar.get_height():.3f}',ha='center',va='bottom',fontsize=9,color='#c8dff0')
        ax.set_xticks(x); ax.set_xticklabels(cats,fontsize=10); ax.set_ylim(0,1.18)
        ax.set_ylabel('Nilai Metrik'); ax.set_title('Perbandingan Metrik Model',fontweight='bold')
        ax.legend(); ax.grid(True,axis='y',alpha=0.3)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
        comp=pd.DataFrame({'Metrik':['Accuracy','Precision (Macro)','Recall (Macro)','F1-Score (Macro)','CV Score'],
                           'Logistic Regression':[round(v,4) for v in lr_v],
                           'Naïve Bayes':[round(v,4) for v in nb_v]})
        comp['Lebih Unggul']=comp.apply(lambda r:'🔷 LR' if r['Logistic Regression']>r['Naïve Bayes'] else('🟠 NB' if r['Naïve Bayes']>r['Logistic Regression'] else '🟰 Seri'),axis=1)
        st.dataframe(comp, use_container_width=True, hide_index=True)

    with t4:
        sec("ROC Curve & AUC")
        fpr_lr,tpr_lr,_=roc_curve(m['y_te_c'],m['y_prob_lr'])
        fpr_nb,tpr_nb,_=roc_curve(m['y_te_c'],m['y_prob_nb'])
        auc_lr=auc(fpr_lr,tpr_lr); auc_nb=auc(fpr_nb,tpr_nb)
        fig,ax=plt.subplots(figsize=(9,6))
        ax.plot(fpr_lr,tpr_lr,color='#4fc3f7',lw=2.5,label=f'Logistic Regression (AUC={auc_lr:.4f})')
        ax.plot(fpr_nb,tpr_nb,color='#ff7043',lw=2.5,label=f'Naïve Bayes (AUC={auc_nb:.4f})')
        ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.5,label='Random Classifier')
        ax.fill_between(fpr_lr,tpr_lr,alpha=0.07,color='#4fc3f7')
        ax.fill_between(fpr_nb,tpr_nb,alpha=0.07,color='#ff7043')
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve — Perbandingan Model',fontweight='bold')
        ax.legend(fontsize=11); ax.grid(True,alpha=0.3)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
        c1,c2=st.columns(2)
        with c1: st.markdown(kpi("AUC — Logistic Regression",f"{auc_lr:.4f}","Semakin mendekati 1.0 = semakin baik"), unsafe_allow_html=True)
        with c2: st.markdown(kpi("AUC — Naïve Bayes",f"{auc_nb:.4f}","Semakin mendekati 1.0 = semakin baik"), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDIKSI
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯  Prediksi":
    st.markdown("<h2 style='font-size:24px;font-weight:800;color:#e8f0f8;margin-bottom:4px'>🎯 Prediksi Cluster Lahan</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#5a8aaa;margin-bottom:20px'>Masukkan kondisi lahan untuk prediksi real-time dari 4 model sekaligus</p>", unsafe_allow_html=True)

    sec("⚙️ Input Parameter Lahan")
    stats = df[NC].describe()
    c1,c2=st.columns(2)
    with c1:
        N_v  = st.slider("🌱 Nitrogen (N) mg/kg",   float(stats.loc['min','N']),   float(stats.loc['max','N']),   float(stats.loc['mean','N']))
        P_v  = st.slider("🌿 Fosfor (P) mg/kg",     float(stats.loc['min','P']),   float(stats.loc['max','P']),   float(stats.loc['mean','P']))
        K_v  = st.slider("🌾 Kalium (K) mg/kg",     float(stats.loc['min','K']),   float(stats.loc['max','K']),   float(stats.loc['mean','K']))
        tp   = st.slider("🌡️ Suhu (°C)",             float(stats.loc['min','temperature']),float(stats.loc['max','temperature']),float(stats.loc['mean','temperature']))
    with c2:
        hm   = st.slider("💧 Kelembapan (%)",       float(stats.loc['min','humidity']),   float(stats.loc['max','humidity']),   float(stats.loc['mean','humidity']))
        ph   = st.slider("⚗️ pH Tanah",              float(stats.loc['min','ph']),         float(stats.loc['max','ph']),         float(stats.loc['mean','ph']),step=0.01)
        rain = st.slider("🌧️ Curah Hujan (mm)",      float(stats.loc['min','rainfall']),   float(stats.loc['max','rainfall']),   float(stats.loc['mean','rainfall']))

    inp = np.array([[N_v, P_v, K_v, tp, hm, ph, rain]])
    inp_scaled_km  = m['scaler'].transform(inp)
    inp_scaled_clf = m['sc_clf'].transform(inp)
    inp_scaled_reg = m['sc_reg'].transform(inp)

    km_pred  = m['kmeans'].predict(inp_scaled_km)[0]
    lr_pred  = m['model_lr'].predict(inp_scaled_clf)[0]
    lr_prob  = m['model_lr'].predict_proba(inp_scaled_clf)[0]
    nb_pred  = m['model_nb'].predict(inp_scaled_clf)[0]
    nb_prob  = m['model_nb'].predict_proba(inp_scaled_clf)[0]
    reg_pred = m['model_reg'].predict(inp_scaled_reg)[0]

    sec("🔮 Hasil Prediksi 4 Model")
    c1,c2,c3,c4=st.columns(4)
    models=[("K-Means",km_pred,None),("Logistic Reg",lr_pred,lr_prob),
            ("Naïve Bayes",nb_pred,nb_prob),("Linear Reg (val)",None,None)]
    for col,(mname,pred,prob) in zip([c1,c2,c3,c4],models):
        with col:
            if mname == "Linear Reg (val)":
                st.markdown(f"""<div class='kpi'>
                    <div class='plbl'>{mname}</div>
                    <div class='pval' style='color:#ffa726'>{reg_pred:.4f}</div>
                    <div style='font-size:11px;color:#aaa;margin-top:4px'>nilai kontinu prediksi</div>
                </div>""", unsafe_allow_html=True)
            else:
                col_cls = 'pred0' if pred==0 else 'pred1'
                color   = '#4fc3f7' if pred==0 else '#ff7043'
                label   = 'Lahan Standar' if pred==0 else 'Lahan Kaya Nutrisi'
                extra   = f"<div style='font-size:11px;color:#aaa;margin-top:5px'>P(C0)={prob[0]:.3f} | P(C1)={prob[1]:.3f}</div>" if prob is not None else ""
                st.markdown(f"""<div class='{col_cls}'>
                    <div class='plbl'>{mname}</div>
                    <div class='pval' style='color:{color}'>{'🔵 Cluster 0' if pred==0 else '🟠 Cluster 1'}</div>
                    <div style='font-size:12px;color:#aaa;margin-top:4px'>{label}</div>
                    {extra}
                </div>""", unsafe_allow_html=True)

    all_same = (km_pred == lr_pred == nb_pred)
    bg = "rgba(79,195,247,.1)" if all_same else "rgba(255,112,67,.1)"
    bc = "rgba(79,195,247,.3)" if all_same else "rgba(255,112,67,.3)"
    msg = "✅ <b>Konsisten:</b> Ketiga model klasifikasi memprediksi hasil yang sama." if all_same else "⚠️ <b>Tidak Konsisten:</b> Terdapat perbedaan prediksi. Pertimbangkan hasil mayoritas."
    st.markdown(f"<div style='background:{bg};border:1px solid {bc};border-radius:12px;padding:14px 20px;margin-top:12px;font-size:14px;color:#c8dff0'>{msg}</div>", unsafe_allow_html=True)

    sec("📊 Distribusi Probabilitas (LR & NB)")
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,(mname,prob) in zip(axes,[("Logistic Regression",lr_prob),("Naïve Bayes",nb_prob)]):
        bars=ax.bar(['Cluster 0','Cluster 1'],prob,color=['#4fc3f7','#ff7043'],alpha=0.85,width=0.4)
        for bar,val in zip(bars,prob):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.01,f'{val:.3f}',
                    ha='center',va='bottom',fontsize=13,fontweight='bold',color='white')
        ax.axhline(0.5,color='white',ls='--',lw=1,alpha=0.4,label='Threshold 0.5')
        ax.set_ylim(0,1.2); ax.set_title(f'Probabilitas — {mname}',fontweight='bold')
        ax.set_ylabel('Probabilitas'); ax.legend(); ax.grid(True,axis='y',alpha=0.3)
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    sec("📋 Ringkasan Input vs Rata-rata Dataset")
    inp_df=pd.DataFrame({
        'Parameter':['N (Nitrogen)','P (Fosfor)','K (Kalium)','Temperature','Humidity','pH','Rainfall'],
        'Nilai Input':[N_v,P_v,K_v,tp,hm,ph,rain],
        'Satuan':['mg/kg','mg/kg','mg/kg','°C','%','-','mm'],
        'Rata-rata Dataset':[round(df[c].mean(),2) for c in NC],
        'Status':['⬆️ Di atas rata-rata' if v>df[c].mean() else '⬇️ Di bawah rata-rata'
                  for v,c in zip([N_v,P_v,K_v,tp,hm,ph,rain],NC)]
    })
    st.dataframe(inp_df, use_container_width=True, hide_index=True)
