import os
import sys
import time
import statistics
import joblib
import argparse
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
from memory_profiler import memory_usage
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)


def carregar_holdout(pasta_modelos: Path) -> tuple:
    caminho = pasta_modelos / 'holdout_teste.pkl'
    if not caminho.is_file():
        print(f"Erro Crítico: O arquivo '{caminho.name}' não foi localizado.")
        print("Execute o script de treinamento primeiro para gerar o holdout de validação interna.")
        sys.exit(1)

    print("\n[I/O] Carregando conjunto de validação interna (Holdout)...")
    x_test, y_test, colunas = joblib.load(caminho)
    print(f"-> Holdout carregado: {len(y_test):,} amostras, {len(colunas)} features.")
    return x_test, y_test


def carregar_arquivos_parquet_teste(arquivos_parquet: list, colunas_treino: list, label_map: dict, tamanho_amostra_alvo=None) -> tuple:
    print("\n[I/O] Carregando e unificando arquivos Parquet de teste com Polars...")
    
    lista_dfs = []
    for caminho in arquivos_parquet:
        print(f"  -> Escaneando fragmento Gold: {caminho.name}")
        
        schema_real = pl.scan_parquet(caminho).collect_schema().names()
        cols_existentes = [c for c in colunas_treino if c in schema_real]
        cols_para_carregar = cols_existentes + ["Label"]
        
        df_temp = pl.read_parquet(caminho, columns=cols_para_carregar)
        
        colunas_faltantes = [c for c in colunas_treino if c not in schema_real]
        if colunas_faltantes:
            df_temp = df_temp.with_columns([pl.lit(0.0).alias(c) for c in colunas_faltantes])
            
        df_temp = df_temp.select(list(colunas_treino) + ["Label"])
        lista_dfs.append(df_temp)
        
    df_pl = pl.concat(lista_dfs)
    df_pd = df_pl.to_pandas()
    
    df_pd["Label_Id"] = df_pd["Label"].astype(str).map(label_map)
    
    n_nulos = df_pd["Label_Id"].isna().sum()
    if n_nulos > 0:
        df_pd.dropna(subset=["Label_Id"], inplace=True)

    if tamanho_amostra_alvo and len(df_pd) > tamanho_amostra_alvo:
        print(f"\n[Amostragem] Reduzindo base externa de forma estratificada para {tamanho_amostra_alvo:,} linhas...")
        proporcoes = df_pd["Label_Id"].value_counts(normalize=True).to_dict()
        
        lista_subs = []
        for classe_id, prop in proporcoes.items():
            df_sub = df_pd[df_pd["Label_Id"] == classe_id]
            qtd_calculada = int(tamanho_amostra_alvo * prop)
            qtd_final = max(min(qtd_calculada, len(df_sub)), min(len(df_sub), 100))
            
            if len(df_sub) > qtd_final:
                df_sub = df_sub.sample(n=qtd_final, random_state=42)
            lista_subs.append(df_sub)
            
        df_pd = pd.concat(lista_subs).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"-> Base de teste reduzida com sucesso para: {len(df_pd):,} linhas.")

    print("\n" + "="*50)
    print("VOLUMETRIA EFETIVA DE REGISTROS NO TESTE:")
    print("="*50)
    contagens_finais = df_pd["Label"].value_counts()
    for classe_nome, total in contagens_finais.items():
        print(f"  - {classe_nome:<22}: {total:,} amostras")
    print("="*50 + "\n")
        
    x_test = df_pd[colunas_treino].values
    y_test = df_pd["Label_Id"].astype(int).values
    
    return x_test, y_test


class MonitorHardware:
    """Classe assíncrona para mapeamento real de consumo da CPU em tempo de execução."""
    def __init__(self, intervalo=0.1):
        self.intervalo = intervalo
        self.historico_cpu = []
        self._ativo = False
        self._thread = None

    def _monitorar(self):
        psutil.cpu_percent(interval=None)
        while self._ativo:
            self.historico_cpu.append(psutil.cpu_percent(interval=None))
            time.sleep(self.intervalo)

    def iniciar(self):
        self.historico_cpu = []
        self._ativo = True
        self._thread = threading.Thread(target=self._monitorar, daemon=True)
        self._thread.start()

    def parar(self):
        self._ativo = False
        if self._thread:
            self._thread.join()
        if not self.historico_cpu:
            return 0.0, 0.0
        return statistics.mean(self.historico_cpu), max(self.historico_cpu)


def testar_modelo(nome_arquivo: str, x_dados: np.ndarray, y_dados: np.ndarray, pasta_modelos: Path, pasta_relatorios: Path, pasta_cm_graficos: Path, sufixo_relatorio: str) -> dict:
    caminho_modelo = pasta_modelos / nome_arquivo
    nome_clean = nome_arquivo.replace('modelo_', '').replace('.pkl', '').replace('_', ' ')
    print(f"\n[Inferência] Avaliando modelo: {nome_clean.upper()}")

    modelo = joblib.load(caminho_modelo)
    classes = joblib.load(pasta_modelos / 'classes.pkl')
    monitor_cpu = MonitorHardware(intervalo=0.05)

    def predizer():
        monitor_cpu.iniciar()
        return modelo.predict(x_dados)

    uso_memoria = memory_usage(predizer, interval=0.05)
    media_cpu, pico_cpu = monitor_cpu.parar()
    
    pico_ram = max(uso_memoria) - min(uso_memoria)
    media_ram = statistics.mean(uso_memoria) - min(uso_memoria)

    inicio = time.time()
    y_pred = modelo.predict(x_dados)
    fim = time.time()
    tempo_pred = fim - inicio

    return calcular_e_salvar(nome_clean, y_dados, y_pred, classes, sufixo_relatorio, tempo_pred, pico_ram, media_ram, media_cpu, pico_cpu, pasta_relatorios, pasta_cm_graficos)


def calcular_e_salvar(nome_clean, y_real, y_pred, classes, sufixo, tempo_pred, pico_ram, media_ram, media_cpu, pico_cpu, pasta_relatorios, pasta_cm_graficos):
    classes_presentes_ids = np.unique(np.concatenate([y_real, y_pred]))
    classes_presentes_nomes = [classes[i] for i in classes_presentes_ids]

    # Matriz gráfica limpa
    gerar_matriz(y_pred, y_real, nome_clean + sufixo, classes, pasta_cm_graficos)

    print(f"\n[Relatório - {nome_clean.upper()}]")
    relatorio = classification_report(y_real, y_pred, target_names=classes_presentes_nomes, zero_division=0)
    print(relatorio)

    nome_base = f"relatorio_{nome_clean.replace(' ', '_')}{sufixo}"
    with open(pasta_relatorios / f"{nome_base}.txt", 'w', encoding='utf-8') as f:
        f.write(f"Algoritmo: {nome_clean}\nAbordagem: {sufixo[1:]}\n")
        f.write(f"Tempo de Predicao: {tempo_pred:.4f}s\nPico RAM: {pico_ram:.2f} MB\n")
        f.write(f"Uso Medio de CPU: {media_cpu:.1f}%\nPico de CPU: {pico_cpu:.1f}%\n\n")
        f.write(relatorio)

    return {
        "Algoritmo": nome_clean,
        "Acuracia": accuracy_score(y_real, y_pred),
        "Precisao": precision_score(y_real, y_pred, average='weighted', zero_division=0),
        "Recall": recall_score(y_real, y_pred, average='weighted', zero_division=0),
        "F1-Score": f1_score(y_real, y_pred, average='weighted', zero_division=0),
        "Tempo Predicao (s)": tempo_pred,
        "Uso Medio de RAM (MB)": media_ram,
        "Pico RAM Predicao (MB)": pico_ram,
        "Uso Medio de CPU (%)": media_cpu,
        "Pico de CPU (%)": pico_cpu
    }


def gerar_matriz(y_pred, y, nome_algoritmo, classes, pasta_cm_graficos: Path):
    cm = confusion_matrix(y, y_pred)
    classes_presentes_ids = np.unique(np.concatenate([y, y_pred]))
    classes_presentes_nomes = [classes[i] for i in classes_presentes_ids]
    
    qtd_classes = len(classes_presentes_nomes)
    largura = max(8, 2 + (np.sqrt(qtd_classes) * 4))
    altura = max(6, 1 + (np.sqrt(qtd_classes) * 3.5))

    fig, ax = plt.subplots(figsize=(largura, altura))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes_presentes_nomes)
    disp.plot(cmap='Blues', values_format='d', xticks_rotation=45, ax=ax)
    
    plt.title(f"Matriz de Confusao - {nome_algoritmo.upper()}\n", fontsize=11, weight='bold')
    plt.tight_layout()
    
    nome_img = f"matriz_confusao_{nome_algoritmo.replace(' ', '_')}.png"
    plt.savefig(pasta_cm_graficos / nome_img, dpi=300)
    plt.close()

def gerar_matriz(y_pred, y, nome_algoritmo, classes, pasta_cm_graficos: Path):
    cm = confusion_matrix(y, y_pred)
    classes_presentes_ids = np.unique(np.concatenate([y, y_pred]))
    classes_presentes_nomes = [classes[i] for i in classes_presentes_ids]
    
    qtd_classes = len(classes_presentes_nomes)
    largura = max(8, 2 + (np.sqrt(qtd_classes) * 4))
    altura = max(6, 1 + (np.sqrt(qtd_classes) * 3.5))

    fig, ax = plt.subplots(figsize=(largura, altura))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes_presentes_nomes)
    disp.plot(cmap='Blues', values_format='d', xticks_rotation=45, ax=ax)
    
    plt.title("Matriz de Confusão por Algoritmo", fontsize=11, weight='bold')
    plt.tight_layout()
    
    nome_img = f"matriz_confusao_{nome_algoritmo.replace(' ', '_')}.png"
    plt.savefig(pasta_cm_graficos / nome_img, dpi=300)
    plt.close()


def gerar_graficos_performance(df_resultados: pd.DataFrame, pasta_perf_graficos: Path, sufixo=""):
    plt.style.use('ggplot')
    sns.set_theme(style="whitegrid", palette="muted")

    plt.figure(figsize=(9, 5))
    df_f1 = df_resultados.sort_values(by="F1-Score", ascending=False)
    ax1 = sns.barplot(x='F1-Score', y='Algoritmo', data=df_f1, hue='Algoritmo', palette="viridis", legend=False)
    plt.title("Comparação de F1-Score Ponderado", fontsize=11, weight='bold')
    plt.xlabel("Score")
    plt.ylabel("")
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.4f', padding=5, weight='bold')
    plt.tight_layout()
    plt.savefig(pasta_perf_graficos / "f1_score_comparacao.png", dpi=300)
    plt.close()

    plt.figure(figsize=(9, 5))
    df_tempo = df_resultados.sort_values(by="Tempo Predicao (s)")
    ax2 = sns.barplot(x='Tempo Predicao (s)', y='Algoritmo', data=df_tempo, hue='Algoritmo', palette="magma", legend=False)
    plt.title("Tempo de Inferência por Algoritmo", fontsize=11, weight='bold')
    plt.xlabel("Segundos (s)")
    plt.ylabel("")
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%.4fs', padding=5, weight='bold')
    plt.tight_layout()
    plt.savefig(pasta_perf_graficos / "tempo_predicao_comparacao.png", dpi=300)
    plt.close()

    plt.figure(figsize=(9, 5))
    df_ram_pico = df_resultados.sort_values(by="Pico RAM Predicao (MB)", ascending=False)
    ax3 = sns.barplot(x='Pico RAM Predicao (MB)', y='Algoritmo', data=df_ram_pico, hue='Algoritmo', palette="rocket", legend=False)
    plt.title("Pico de Memória RAM na Inferência", fontsize=11, weight='bold')
    plt.xlabel("Megabytes (MB)")
    plt.ylabel("")
    for container in ax3.containers:
        ax3.bar_label(container, fmt='%.2f MB', padding=5, weight='bold')
    plt.tight_layout()
    plt.savefig(pasta_perf_graficos / "pico_ram_comparacao.png", dpi=300)
    plt.close()

    plt.figure(figsize=(9, 5))
    df_ram_med = df_resultados.sort_values(by="Uso Medio de RAM (MB)", ascending=False)
    ax4 = sns.barplot(x='Uso Medio de RAM (MB)', y='Algoritmo', data=df_ram_med, hue='Algoritmo', palette="crest", legend=False)
    plt.title("Consumo Médio de Memória RAM na Inferência", fontsize=11, weight='bold')
    plt.xlabel("Megabytes (MB)")
    plt.ylabel("")
    for container in ax4.containers:
        ax4.bar_label(container, fmt='%.2f MB', padding=5, weight='bold')
    plt.tight_layout()
    plt.savefig(pasta_perf_graficos / "consumo_medio_ram_comparacao.png", dpi=300)
    plt.close()

    plt.figure(figsize=(9, 5))
    df_cpu_pico = df_resultados.sort_values(by="Pico de CPU (%)", ascending=False)
    ax5 = sns.barplot(x='Pico de CPU (%)', y='Algoritmo', data=df_cpu_pico, hue='Algoritmo', palette="flare", legend=False)
    plt.title("Pico de Uso de CPU na Inferência", fontsize=11, weight='bold')
    plt.xlabel("Porcentagem (%)")
    plt.ylabel("")
    for container in ax5.containers:
        ax5.bar_label(container, fmt='%.1f%%', padding=5, weight='bold')
    plt.tight_layout()
    plt.savefig(pasta_perf_graficos / "pico_cpu_comparacao.png", dpi=300)
    plt.close()
    
    plt.figure(figsize=(9, 5))
    df_cpu_med = df_resultados.sort_values(by="Uso Medio de CPU (%)", ascending=False)
    ax6 = sns.barplot(x='Uso Medio de CPU (%)', y='Algoritmo', data=df_cpu_med, hue='Algoritmo', palette="vlag", legend=False)
    plt.title("Consumo Médio de CPU na Inferência", fontsize=11, weight='bold')
    plt.xlabel("Porcentagem (%)")
    plt.ylabel("")
    for container in ax6.containers:
        ax6.bar_label(container, fmt='%.1f%%', padding=5, weight='bold')
    plt.tight_layout()
    plt.savefig(pasta_perf_graficos / "consumo_medio_cpu_comparacao.png", dpi=300)
    plt.close()

modelos_arquivos = [
    "modelo_decision_tree.pkl",
    "modelo_knn.pkl",
    "modelo_logistic_regression.pkl",
    "modelo_naive_bayes.pkl",
    "modelo_random_forest.pkl"
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pipeline de Testes Unificado")
    parser.add_argument("--tipo", choices=["binario", "multiclasse"], required=True, help="Tipo de abordagem de dados")
    args = parser.parse_args()

    BASE_DIR = Path(__file__).resolve().parent
    PASTA_OURO = BASE_DIR / "data" / "ouro" / args.tipo
    PASTA_MODELOS = BASE_DIR / "modelos" / args.tipo
    PASTA_RELATORIOS = BASE_DIR / "relatorios" / "teste" / args.tipo
    PASTA_PERF_GRAFICOS = BASE_DIR / "graficos" / "performance" / args.tipo
    PASTA_CM_GRAFICOS = BASE_DIR / "graficos" / "matriz_de_confusao" / args.tipo

    for pasta in [PASTA_RELATORIOS, PASTA_PERF_GRAFICOS, PASTA_CM_GRAFICOS]:
        pasta.mkdir(parents=True, exist_ok=True)

    print("\n" + ("=" * 60))
    print(f" AMBIENTE DE AVALIAÇÃO DE MODELOS - [{args.tipo.upper()}]")
    print("=" * 60)
    print("[1] Experimento 1 - Holdout Interno (Validação interna de treino)")
    print("[2] Experimento 2 - Generalização (CIC-IDS-2017 da Gold)")
    print("=" * 60 + "\n")

    experimento = input("Escolha o Experimento (1 ou 2): ").strip()

    x, y = None, None
    sufixo_relatorio = ""

    if not (PASTA_MODELOS / 'classes.pkl').exists() or not (PASTA_MODELOS / 'scaler.pkl').exists():
        print(f"Erro Crítico: Arquivos 'classes.pkl' ou 'scaler.pkl' ausentes em {PASTA_MODELOS}")
        sys.exit(1)
        
    classes = joblib.load(PASTA_MODELOS / 'classes.pkl')
    scaler = joblib.load(PASTA_MODELOS / 'scaler.pkl')
    label_map = {nome: i for i, nome in enumerate(classes)}
    colunas_treino = scaler.feature_names_in_

    if experimento == "1":
        print("\n--- INICIANDO EXPERIMENTO 1: HOLDOUT ---")
        x, y = carregar_holdout(PASTA_MODELOS)
        sufixo_relatorio = "_holdout"

    elif experimento == "2":
        print("\n--- INICIANDO EXPERIMENTO 2: GENERALIZAÇÃO ---")
        
        arquivos_parquet = sorted(list(PASTA_OURO.glob('*.parquet')))
        arquivos_parquet = [f for f in arquivos_parquet if "treino" not in f.name]
            
        if not arquivos_parquet:
            print(f"Erro Crítico: Nenhum arquivo Parquet de teste localizado em: '{PASTA_OURO}'.")
            sys.exit(1)

        print("\nQual volumetria de dados deseja utilizar para este teste real?")
        print("[1] Usar a base de teste cheia")
        print("[2] Usar amostra de 500.000 linhas")
        print("[3] Usar amostra de 250.000 linhas")
        print("[4] Usar amostra de 100.000 linhas")
        print("[5] Usar amostra de 25.000 linhas (Super leve)")
        
        opcao_vol = input("Escolha a opção (Padrão=1): ").strip()
        
        tamanho_teste_alvo = None
        if opcao_vol == "2": tamanho_teste_alvo = 500000
        elif opcao_vol == "3": tamanho_teste_alvo = 250000
        elif opcao_vol == "4": tamanho_teste_alvo = 100000
        elif opcao_vol == "5": tamanho_teste_alvo = 25000

        x, y = carregar_arquivos_parquet_teste(arquivos_parquet, colunas_treino, label_map, tamanho_teste_alvo)
        sufixo_relatorio = "_generalizacao"
        x = scaler.transform(x)
    else:
        print("Opção inválida de experimento.")
        sys.exit(1)

    resultados = []
    
    for arquivo in modelos_arquivos:
        if not (PASTA_MODELOS / arquivo).exists():
            continue

        nome_clean = arquivo.replace('modelo_', '').replace('.pkl', '').replace('_', ' ')
        resp = input(f"\nDeseja avaliar o modelo '{nome_clean.upper()}'? (y/n) [Default=y]: ").strip().lower()

        if resp == 'y' or not resp:
            try:
                res_metrics = testar_modelo(arquivo, x, y, PASTA_MODELOS, PASTA_RELATORIOS, PASTA_CM_GRAFICOS, sufixo_relatorio)
                resultados.append(res_metrics)
            except Exception as e:
                print(f"  Erro crítico ao avaliar o modelo {arquivo}: {e}")

    if resultados:
        df_resultados = pd.DataFrame(resultados)
        print("\n" + "="*80)
        print(" TABELA COMPARATIVA DE PERFORMANCE DOS MODELOS (TESTE)")
        print("="*80)
        print(df_resultados.sort_values(by="F1-Score", ascending=False).to_string(index=False))
        print("="*80)

        df_resultados.to_csv(PASTA_RELATORIOS / "metricas_comparativas_teste.csv", index=False, encoding="utf-8")
        print(f"-> Resumo comparativo das métricas salvo em CSV: metricas_comparativas_teste.csv")

        gerar_graficos_performance(df_resultados, PASTA_PERF_GRAFICOS, sufixo_relatorio)
        print(f"\nAvaliação concluída. Todos os gráficos individuais salvos com sucesso em: {PASTA_PERF_GRAFICOS}")
    else:
        print("\nNenhum modelo foi selecionado para o teste.")