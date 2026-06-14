import streamlit as st
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Dashboard", layout="wide", initial_sidebar_state="collapsed")

COR_FUNDO   = "#0a0a0a"
COR_CARD    = "#000000"
COR_TEXTO   = "#e2e8f0"
COR_DIVISA  = "#2d2d2d"
CORES_DISTINTAS = ["#38bdf8","#fb923c","#4ade80","#d61f1f","#f5c453","#ec4899","#a855f7","#64748b"]
COR_VERDE   = "#004e1e"
COR_CIANO   = "#0a0180"
COR_LARANJA = "#fb923c"
COR_ROXO    = "#480069"

st.markdown(f"""
    <style>
    header[data-testid="stHeader"]          {{ visibility: hidden; height: 0% !important; display: none !important; }}
    div[data-testid="stToolbar"]            {{ visibility: hidden; display: none !important; }}
    section[data-testid="stSidebar"]        {{ display: none !important; }}
    div[data-testid="stVerticalBlock"] > div {{
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0px !important;
    }}
    .stApp, div[data-testid="stAppViewContainer"] {{
        background-color: {COR_FUNDO} !important;
        color: {COR_TEXTO};
        font-family: 'Inter', sans-serif;
    }}
    .block-container {{
        padding-top: 1.5rem !important;
        padding-bottom: 2rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
    }}
    div[data-testid="stHeaderActionElements"] {{ display: none; }}
    div[data-testid="stMetric"] {{
        background-color: {COR_CARD} !important;
        border: 1px solid {COR_DIVISA} !important;
        border-radius: 6px !important;
        padding: 12px 15px !important;
        box-shadow: none !important;
    }}
    div[data-testid="stMetricValue"] {{
        color: #ffffff !important;
        font-size: 24px !important;
        font-weight: 700 !important;
    }}
    div[data-testid="stMetricLabel"] {{
        color: #94a3b8 !important;
        font-size: 11px !important;
        text-transform: uppercase;
    }}
    .caixa-grafico {{
        background-color: {COR_CARD} !important;
        padding: 22px;
    }}
    .titulo-grafico {{
        font-size: 12px;
        color: #ffffff;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.05em;
        margin-bottom: 15px;
        display: block;
    }}
    div[data-testid="stRadio"] label,
    div[data-testid="stSelectbox"] label {{
        color: #94a3b8 !important;
        font-size: 11px !important;
        text-transform: uppercase;
        font-weight: 600;
    }}
    </style>
""", unsafe_allow_html=True)

_ROOT = os.path.dirname(os.path.abspath(__file__))

BASE_TREINO_MULTI = os.path.join(_ROOT, "relatorios", "treinamento", "multiclasse")
BASE_TREINO_BIN   = os.path.join(_ROOT, "relatorios", "treinamento", "binario")
BASE_TESTE_MULTI  = os.path.join(_ROOT, "relatorios", "teste", "multiclasse")
BASE_TESTE_BIN    = os.path.join(_ROOT, "relatorios", "teste", "binario")

BASE_OURO_MULTI   = os.path.join(_ROOT, "data", "ouro", "multiclasse", "relatorios")
BASE_OURO_BIN     = os.path.join(_ROOT, "data", "ouro", "binario", "relatorios")
BASE_OURO_BIN_ALT = os.path.join(_ROOT, "data", "ouro", "binário", "relatórios")

ALGORITMOS_LABEL = {
    "decision_tree":   "Decision Tree",
    "extra_trees":     "Extra Trees",
    "lightgbm":        "LightGBM",
    "random_forest":   "Random Forest",
    "xgboost":         "XGBoost",
}

if "auc_roc_extraido" not in st.session_state:
    st.session_state["auc_roc_extraido"] = None

@st.cache_data
def carregar_distribuicao(tipo: str) -> pl.DataFrame:
    base = BASE_TREINO_MULTI if tipo == "multiclasse" else BASE_TREINO_BIN
    return pl.read_csv(os.path.join(base, "distribuicao_classes.csv"))

@st.cache_data
def carregar_hardware_treino(tipo: str) -> pl.DataFrame:
    base = BASE_TREINO_MULTI if tipo == "multiclasse" else BASE_TREINO_BIN
    return pl.read_csv(os.path.join(base, "metricas_hardware.csv"))

@st.cache_data
def carregar_metricas_comparativas(tipo: str, abordagem: str) -> pl.DataFrame:
    base = BASE_TESTE_MULTI if tipo == "multiclasse" else BASE_TESTE_BIN
    nome = (
        "metricas_comparativas_teste_holdout.csv"
        if abordagem == "holdout"
        else "metricas_comparativas_teste_generalizacao.csv"
    )
    return pl.read_csv(os.path.join(base, nome))

def carregar_relatorio_classes(tipo: str, abordagem: str, alg_key: str) -> pl.DataFrame:
    base = BASE_TESTE_MULTI if tipo == "multiclasse" else BASE_TESTE_BIN
    
    nome_procurado = f"relatorio_{alg_key}_{abordagem}.csv".lower()
    arquivo_real = None
    
    if os.path.exists(base):
        for f in os.listdir(base):
            if f.lower() == nome_procurado:
                arquivo_real = f
                break
                
    if not arquivo_real:
        variacoes = [
            f"relatorio_{alg_key}_{abordagem}.csv",
            f"relatorio_{ALGORITMOS_LABEL[alg_key].replace(' ', '_')}_{abordagem}.csv"
        ]
        for v in variacoes:
            if os.path.exists(os.path.join(base, v)):
                arquivo_real = v
                break
        if not arquivo_real:
            arquivo_real = variacoes[0]
        
    path = os.path.join(base, arquivo_real)
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não localizado no disco: {arquivo_real}")
        
    df = pl.read_csv(path)
    
    # Padroniza e limpa colunas
    df = df.rename({c: c.strip() for c in df.columns})
    col_classe = df.columns[0]
    
    linha_roc = df.filter(pl.col(col_classe).str.contains("METRICAS_HARDWARE"))
    auc_valor = "N/A"
    if not linha_roc.is_empty():

        for col_name in df.columns:
            celula_str = str(linha_roc[col_name][0])
            if "AUC ROC:" in celula_str:
                auc_valor = celula_str.split("AUC ROC:")[-1].strip()
                break
    
    st.session_state["auc_roc_extraido"] = auc_valor
  
    excluir = {"accuracy", "macro avg", "weighted avg"}
    df = df.filter(~pl.col(col_classe).is_in(excluir))
    df = df.filter(~pl.col(col_classe).str.contains("METRICAS_HARDWARE"))
    

    colunas_validas = []
    for col in ["precision", "recall", "f1-score", "support"]:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.String).str.strip_chars().cast(pl.Float64, strict=False))
            colunas_validas.append(col)
            
    df = df.select([col_classe] + colunas_validas)
    
    novos_nomes = {col_classe: "Classe"}
    if "precision" in df.columns: novos_nomes["precision"] = "Precisao"
    if "recall" in df.columns:    novos_nomes["recall"] = "Recall"
    if "f1-score" in df.columns:  novos_nomes["f1-score"] = "F1-Score"
    if "support" in df.columns:   novos_nomes["support"] = "Suporte"
    
    return df.rename(novos_nomes)

@st.cache_data
def carregar_volumetria_ouro(tipo: str) -> dict:
    if tipo == "multiclasse":
        base = BASE_OURO_MULTI
    else:
        base = BASE_OURO_BIN if os.path.exists(BASE_OURO_BIN) else BASE_OURO_BIN_ALT
        
    return {
        "treino": pl.read_csv(os.path.join(base, "relatorio_treino_final.csv")),
        "teste":  pl.read_csv(os.path.join(base, "relatorio_teste_final.csv")),
    }

f_escopo1, f_escopo2, _ = st.columns([0.25, 0.25, 0.5])
with f_escopo1:
    tipo_classificacao = st.radio("Classificação:", ["multiclasse", "binário"], horizontal=True)
with f_escopo2:
    board_abordagem = st.radio("Abordagem:", ["holdout", "generalização"], horizontal=True)

tipo_pasta = "multiclasse" if tipo_classificacao == "multiclasse" else "binario"
abordagem_arquivo = "holdout" if board_abordagem == "holdout" else "generalizacao"

st.markdown(f"<hr style='margin-top:-5px;margin-bottom:20px;border-color:{COR_DIVISA};'>", unsafe_allow_html=True)

df_distribuicao  = carregar_distribuicao(tipo_pasta)
df_hw_treino     = carregar_hardware_treino(tipo_pasta)
df_metricas_comp = carregar_metricas_comparativas(tipo_pasta, abordagem_arquivo)
vol_ouro         = carregar_volumetria_ouro(tipo_pasta)

df_dist_p = df_distribuicao.to_pandas()

col_pizza_1, col_pizza_2, col_pizza_3 = st.columns(3)

with col_pizza_1:
    st.markdown('<div class="caixa-grafico"><span class="titulo-grafico">Volumetria Bruta</span>', unsafe_allow_html=True)
    fig_antes = px.pie(df_dist_p, values="Antes do Bal.", names="Classe",
                       color_discrete_sequence=CORES_DISTINTAS, hole=0.5)
    fig_antes.update_layout(template="plotly_dark", paper_bgcolor=COR_CARD, plot_bgcolor=COR_CARD,
                             height=240, margin=dict(l=5, r=5, t=5, b=5),
                             legend=dict(orientation="h", yanchor="bottom", y=-0.7, xanchor="center", x=0.5))
    fig_antes.update_traces(textposition='inside', textinfo='percent')
    st.plotly_chart(fig_antes, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

with col_pizza_2:
    st.markdown('<div class="caixa-grafico"><span class="titulo-grafico">Volumetria Balanceada</span>', unsafe_allow_html=True)
    fig_depois = px.pie(df_dist_p, values="Após Bal. (Final)", names="Classe",
                        color_discrete_sequence=CORES_DISTINTAS, hole=0.5)
    fig_depois.update_layout(template="plotly_dark", paper_bgcolor=COR_CARD, plot_bgcolor=COR_CARD,
                              height=240, margin=dict(l=5, r=5, t=5, b=5),
                              legend=dict(orientation="h", yanchor="bottom", y=-0.7, xanchor="center", x=0.5))
    fig_depois.update_traces(textposition='inside', textinfo='percent')
    st.plotly_chart(fig_depois, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

with col_pizza_3:
    st.markdown('<div class="caixa-grafico"><span class="titulo-grafico">Volumetria Base de Teste</span>', unsafe_allow_html=True)
    df_teste_ouro = vol_ouro["teste"].to_pandas()
    fig_teste = px.pie(df_teste_ouro, values="count", names="Label",
                       color_discrete_sequence=CORES_DISTINTAS, hole=0.5)
    fig_teste.update_layout(template="plotly_dark", paper_bgcolor=COR_CARD, plot_bgcolor=COR_CARD,
                             height=240, margin=dict(l=5, r=5, t=5, b=5),
                             legend=dict(orientation="h", yanchor="bottom", y=-0.7, xanchor="center", x=0.5))
    fig_teste.update_traces(textposition='inside', textinfo='percent')
    st.plotly_chart(fig_teste, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

col_sel, _ = st.columns([0.3, 0.7])
with col_sel:
    algoritmo_label = st.selectbox("Selecione o Algoritmo para Filtro de Processo:",
                                   list(ALGORITMOS_LABEL.values()))

algoritmo_key = next(k for k, v in ALGORITMOS_LABEL.items() if v == algoritmo_label)

df_rc_p = None
erro_rc  = None
try:
    df_rc   = carregar_relatorio_classes(tipo_pasta, abordagem_arquivo, algoritmo_key)
    df_rc_p = df_rc.to_pandas()
    for col in ["Precisao", "Recall", "F1-Score"]:
        if col in df_rc_p.columns:
            df_rc_p[col] = df_rc_p[col].round(2)
except Exception as e:
    erro_rc = str(e)

st.markdown(f"<p style='font-size:11px;color:#94a3b8;text-transform:uppercase;font-weight:600;margin-bottom:6px;'>Treino</p>", unsafe_allow_html=True)

d_hw = df_hw_treino.filter((pl.col("Algoritmo") == algoritmo_label) | (pl.col("Algoritmo") == algoritmo_key)).to_dicts()
if not d_hw:
    d_hw = df_hw_treino.filter(pl.col("Algoritmo").str.to_lowercase() == algoritmo_label.lower()).to_dicts()

if d_hw:
    d_hw = d_hw[0]
    t1, t2, t3, t4, t5 = st.columns(5)
    with t1: st.metric("Tempo",          f"{round(d_hw['Tempo de Treino (s)'], 2)} s")
    with t2: st.metric("RAM Média ",        f"{round(d_hw['Uso Medio de Memoria (MB)'], 2)} MB")
    with t3: st.metric("Pico de RAM ",      f"{round(d_hw['Pico de RAM (MB)'], 2)} MB")
    with t4: st.metric("CPU Média",        f"{round(d_hw['Uso Medio de CPU (%)'], 2)}%")
    with t5: st.metric("Pico de CPU",      f"{round(d_hw['Pico de CPU (%)'], 2)}%")
else:
    st.warning(f"Hardware de treino não encontrado para: {algoritmo_label}")

st.markdown(f"<p style='font-size:11px;color:#94a3b8;text-transform:uppercase;font-weight:600;margin-top:10px;margin-bottom:6px;'>Teste</p>", unsafe_allow_html=True)

d_pred = df_metricas_comp.filter(pl.col("Algoritmo") == algoritmo_key).to_dicts()
if not d_pred:
    d_pred = df_metricas_comp.filter(pl.col("Algoritmo") == algoritmo_label).to_dicts()
if not d_pred:
    d_pred = df_metricas_comp.filter(pl.col("Algoritmo").str.to_lowercase() == algoritmo_label.lower()).to_dicts()

if d_pred:
    d_pred = d_pred[0]

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1: st.metric("Acurácia Geral",             f"{round(d_pred['Acuracia'] * 100, 2)}%")
    with m2: st.metric("Precisão Ponderada",         f"{round(d_pred['Precisao'], 2)}")
    with m3: st.metric("Recall Ponderado",           f"{round(d_pred['Recall'], 2)}")
    with m4: st.metric("F1-Score Ponderado",         f"{round(d_pred['F1-Score'], 2)}")
    
    auc_display = st.session_state.get("auc_roc_extraido", "N/A")
    with m5: st.metric("Curva AUC ROC",              auc_display)
    
    with m6: st.metric("Tempo de Teste",             f"{round(d_pred['Tempo Predicao (s)'], 2)} s")
else:
    st.warning(f"Métricas comparativas não encontradas para: {algoritmo_label}")

st.markdown("<br>", unsafe_allow_html=True)

col_ram, col_cpu = st.columns(2)

with col_ram:
    st.markdown('<div class="caixa-grafico"><span class="titulo-grafico">RAM — Média vs Pico</span>', unsafe_allow_html=True)
    if d_hw and d_pred:
        ram_media_treino = d_hw.get("Uso Medio de Memoria (MB)", 0)
        ram_pico_treino  = d_hw.get("Pico de RAM (MB)", 0)
        ram_media_teste  = d_pred.get("Uso Medio de RAM (MB)", 0)
        ram_pico_teste   = d_pred.get("Pico RAM Predicao (MB)", 0)

        fig_ram = go.Figure()

        fig_ram.add_trace(go.Bar(
            y=['Teste', 'Treino'],
            x=[ram_media_teste, ram_media_treino],
            name='Média',
            orientation='h',
            marker=dict(color=[COR_ROXO, COR_CIANO]),
            text=[f"{round(ram_media_teste, 1)} MB", f"{round(ram_media_treino, 1)} MB"],
            textposition='inside',
            width=0.4
        ))

        fig_ram.add_trace(go.Bar(
            y=['Teste', 'Treino'],
            x=[ram_pico_teste, ram_pico_treino],
            name='Pico',
            orientation='h',
            marker=dict(color=[COR_LARANJA, COR_LARANJA], opacity=0.6),
            text=[f"Pico: {round(ram_pico_teste, 1)}", f"Pico: {round(ram_pico_treino, 1)}"],
            textposition='outside',
            width=0.2
        ))

        fig_ram.update_layout(
            template="plotly_dark", paper_bgcolor=COR_CARD, plot_bgcolor=COR_CARD,
            barmode='overlay', showlegend=False, height=140,      
            margin=dict(l=60, r=80, t=10, b=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(tickfont=dict(size=12, color="#94a3b8"))
        )
        st.plotly_chart(fig_ram, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

with col_cpu:
    st.markdown('<div class="caixa-grafico"><span class="titulo-grafico">CPU — Média vs Pico</span>', unsafe_allow_html=True)
    if d_hw and d_pred:
        cpu_media_treino = d_hw.get("Uso Medio de CPU (%)", 0)
        cpu_pico_treino  = d_hw.get("Pico de CPU (%)", 0)
        cpu_media_teste  = d_pred.get("Uso Medio de CPU (%)", 0)
        cpu_pico_teste   = d_pred.get("Pico de CPU (%)", 0)

        fig_cpu = go.Figure()
        
        fig_cpu.add_trace(go.Bar(
            y=['Teste', 'Treino'],
            x=[cpu_media_teste, cpu_media_treino],
            name='Média',
            orientation='h',
            marker=dict(color=[COR_VERDE, COR_CIANO]),
            text=[f"{round(cpu_media_teste, 1)}%", f"{round(cpu_media_treino, 1)}%"],
            textposition='inside',
            width=0.4
        ))
    
        fig_cpu.add_trace(go.Bar(
            y=['Teste', 'Treino'],
            x=[cpu_pico_teste, cpu_pico_treino],
            name='Pico',
            orientation='h',
            marker=dict(color=["#d61f1f", "#d61f1f"], opacity=0.6),
            text=[f"Pico: {round(cpu_pico_teste, 1)}%", f"Pico: {round(cpu_pico_treino, 1)}%"],
            textposition='outside',
            width=0.2
        ))

        fig_cpu.update_layout(
            template="plotly_dark", paper_bgcolor=COR_CARD, plot_bgcolor=COR_CARD,
            barmode='overlay', showlegend=False, height=140,         
            margin=dict(l=60, r=80, t=10, b=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 120]), 
            yaxis=dict(tickfont=dict(size=12, color="#94a3b8"))
        )
        st.plotly_chart(fig_cpu, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="caixa-grafico"><span class="titulo-grafico">Métricas por Classe </span>', unsafe_allow_html=True)
if erro_rc:
    st.warning(f"Aviso: Não foi possível carregar as classes para '{algoritmo_label}'. Erro: {erro_rc}")
elif df_rc_p is not None and not df_rc_p.empty:
    fig_linhas = go.Figure()
    if "Precisao" in df_rc_p.columns:
        fig_linhas.add_trace(go.Scatter(x=df_rc_p["Classe"], y=df_rc_p["Precisao"], name="Precisão",
                                        mode='lines+markers', line=dict(color=COR_VERDE, width=2),
                                        marker=dict(size=7)))
    if "Recall" in df_rc_p.columns:
        fig_linhas.add_trace(go.Scatter(x=df_rc_p["Classe"], y=df_rc_p["Recall"], name="Recall",
                                        mode='lines+markers', line=dict(color=COR_CIANO, width=2),
                                        marker=dict(size=7)))
    if "F1-Score" in df_rc_p.columns:
        fig_linhas.add_trace(go.Scatter(x=df_rc_p["Classe"], y=df_rc_p["F1-Score"], name="F1-Score",
                                        mode='lines+markers', line=dict(color=COR_LARANJA, width=2),
                                        marker=dict(size=7)))
    fig_linhas.update_layout(
        template="plotly_dark", paper_bgcolor=COR_CARD, plot_bgcolor=COR_CARD,
        height=320, margin=dict(l=120, r=120, t=20, b=50),
        yaxis=dict(gridcolor=COR_DIVISA, range=[0, 1.05]),
        xaxis=dict(gridcolor=COR_DIVISA),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_linhas, use_container_width=True, config={'displayModeBar': False})
else:
    st.info("Nenhum dado de classe disponível para exibir o gráfico.")
st.markdown('</div>', unsafe_allow_html=True)