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
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize


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
    def __init__(self, intervalo=0.1):
        self.intervalo = intervalo
        self.historico_cpu = []
        self._ativo = False
        self._thread = None

    def abrir(self):
        pass

    def iniciar(self):
        self.historico_cpu = []
        self._ativo = True
        self._thread = threading.Thread(target=self._monitorar, daemon=True)
        self._thread.start()

    def _monitorar(self):
        psutil.cpu_percent(interval=None)
        while self._ativo:
            self.historico_cpu.append(psutil.cpu_percent(interval=None))
            time.sleep(self.intervalo)

    def parar(self):
        self._ativo = False
        if self._thread:
            self._thread.join()
        if not self.historico_cpu:
            return 0.0, 0.0
        return statistics.mean(self.historico_cpu), max(self.historico_cpu)


def testar_modelo(nome_arquivo: str, x_dados: np.ndarray, y_dados: np.ndarray, pasta_modelos: Path, pasta_relatorios: Path, pasta_cm_graficos: Path, sufixo_relatorio: str) -> tuple:
    caminho_modelo = pasta_modelos / nome_arquivo
    
   
    nome_clean = nome_arquivo.replace('modelo_', '').replace('.pkl', '').replace('_', ' ').title()
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
    y_proba = modelo.predict_proba(x_dados)
    fim = time.time()
    tempo_pred = fim - inicio

    metricas_res = calcular_e_salvar(nome_clean, y_dados, y_pred, y_proba, classes, sufixo_relatorio, tempo_pred, pico_ram, media_ram, media_cpu, pico_cpu, pasta_relatorios, pasta_cm_graficos)
    return metricas_res, y_proba


def calcular_e_salvar(nome_clean, y_real, y_pred, y_proba, classes, sufixo, tempo_pred, pico_ram, media_ram, media_cpu, pico_cpu, pasta_relatorios, pasta_cm_graficos):
    classes_presentes_ids = np.unique(np.concatenate([y_real, y_pred]))
    classes_presentes_nomes = [classes[i] for i in classes_presentes_ids]

    gerar_matriz(y_pred, y_real, nome_clean + sufixo, classes, pasta_cm_graficos)

    print(f"\n[Relatório - {nome_clean.upper()}]")
    relatorio_dict = classification_report(y_real, y_pred, target_names=classes_presentes_nomes, zero_division=0, output_dict=True)
    relatorio_texto = classification_report(y_real, y_pred, target_names=classes_presentes_nomes, zero_division=0)
    print(relatorio_texto)

    if len(classes) == 2:
        auc_score = roc_auc_score(y_real, y_proba[:, 1])
    else:
        auc_score = roc_auc_score(y_real, y_proba, multi_class='ovr', average='weighted')

    nome_base = f"relatorio_{nome_clean.replace(' ', '_')}{sufixo}"
    
    df_rep = pd.DataFrame(relatorio_dict).transpose().reset_index()
    df_rep.rename(columns={'index': 'Metrica_Classe'}, inplace=True)
    
    df_meta = pd.DataFrame([{
        'Metrica_Classe': 'METRICAS_HARDWARE_E_ROC',
        'precision': f"Tempo Predicao: {tempo_pred:.4f}s",
        'recall': f"Pico RAM: {pico_ram:.2f} MB",
        'f1-score': f"Uso Medio CPU: {media_cpu:.1f}%",
        'support': f"AUC ROC: {auc_score:.4f}"
    }])
    
    df_final_report = pd.concat([df_rep, df_meta], ignore_index=True)
    df_final_report.to_csv(pasta_relatorios / f"{nome_base}.csv", index=False, encoding='utf-8')

    return {
        "Algoritmo": nome_clean,
        "Acuracia": accuracy_score(y_real, y_pred),
        "Precisao": precision_score(y_real, y_pred, average='weighted', zero_division=0),
        "Recall": recall_score(y_real, y_pred, average='weighted', zero_division=0),
        "F1-Score": f1_score(y_real, y_pred, average='weighted', zero_division=0),
        "AUC-ROC": auc_score,
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
    
    plt.title("Matriz de Confusão por Algoritmo", fontsize=11, weight='bold')
    plt.tight_layout()
    
    nome_img = f"matriz_confusao_{nome_algoritmo.replace(' ', '_')}.png"
    plt.savefig(pasta_cm_graficos / nome_img, dpi=300)
    plt.close()


def gerar_curva_roc_composta(dict_probabilidades: dict, y_real, classes, pasta_graficos: Path, sufixo: str):
    plt.figure(figsize=(9, 6))
    sns.set_theme(style="whitegrid")
    
    for nome_modelo, y_proba in dict_probabilidades.items():
        if len(classes) == 2:
            fpr, tpr, _ = roc_curve(y_real, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"{nome_modelo} (AUC = {roc_auc:.4f})")
        else:
            y_real_bin = label_binarize(y_real, classes=range(len(classes)))
            fpr = dict()
            tpr = dict()
            
            for i in range(len(classes)):
           
                fpr[i], tpr[i], _ = roc_curve(y_real_bin[:, i], y_proba[:, i])
            
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(len(classes)):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= len(classes)
            
            macro_auc = auc(all_fpr, mean_tpr)
            plt.plot(all_fpr, mean_tpr, lw=2, label=f"{nome_modelo} (Macro AUC = {macro_auc:.4f})")

    plt.plot([0, 1], [0, 1], color='tab:gray', lw=1.5, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos (FPR)')
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
    plt.title(f'Comparativo de Curva ROC - Experimento {sufixo.replace("_", "").upper()}', fontsize=12, weight='bold')
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    nome_img = f"curva_roc_comparativa{sufixo}.png"
    plt.savefig(pasta_graficos / nome_img, dpi=300)
    plt.close()
    print(f"\n-> Gráfico estruturado de Curva ROC unificado salvo em: {nome_img}")


def gerar_graficos_performance(df_resultados: pd.DataFrame, pasta_perf_graficos: Path, sufixo=""):
    sns.set_theme(style="ticks")

    plt.figure(figsize=(10, 5))
    df_tempo = df_resultados.sort_values(by="Tempo Predicao (s)", ascending=False)
    ax2 = sns.pointplot(
        x='Tempo Predicao (s)', y='Algoritmo', data=df_tempo,
        linestyles="--", markers="D", color="#d95f02", scale=1.2
    )
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    sns.despine(left=False, bottom=False)
    plt.title("Tempo por Algoritmo", fontsize=12, weight='bold', pad=15)
    plt.xlabel("Segundos (s)")
    plt.ylabel("")
    for i, valor in enumerate(df_tempo['Tempo Predicao (s)']):
        ax2.text(valor, i, f' {valor:.4f}s', va='center', ha='left', weight='bold', fontsize=10)
    plt.tight_layout()
    plt.savefig(pasta_perf_graficos / f"tempo_predicao_comparacao{sufixo}.png", dpi=300)
    plt.close()

    # Perfil de Memória RAM
    plt.figure(figsize=(10, 6))
    df_melted_ram = df_resultados.melt(
        id_vars=['Algoritmo'], 
        value_vars=['Uso Medio de RAM (MB)', 'Pico RAM Predicao (MB)'],
        var_name='Métrica', value_name='Megabytes'
    )
    df_melted_ram['Métrica'] = df_melted_ram['Métrica'].replace({
        'Uso Medio de RAM (MB)': 'Consumo Médio',
        'Pico RAM Predicao (MB)': 'Pico de Consumo'
    })
    ax3 = sns.barplot(
        x='Algoritmo', y='Megabytes', hue='Métrica', 
        data=df_melted_ram, palette=['#7570b3', '#e7298a']
    )
    sns.despine()
    plt.title("Consumo de Memória RAM", fontsize=12, weight='bold', pad=15)
    plt.xlabel("")
    plt.ylabel("Megabytes (MB)")
    plt.legend(frameon=True, facecolor='white', edgecolor='none')
    for container in ax3.containers:
        ax3.bar_label(container, fmt='%.1f MB', padding=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(pasta_perf_graficos / f"ram_perfil_comparacao{sufixo}.png", dpi=300)
    plt.close()

    # Perfil de CPU
    plt.figure(figsize=(10, 6))
    df_melted_cpu = df_resultados.melt(
        id_vars=['Algoritmo'], 
        value_vars=['Uso Medio de CPU (%)', 'Pico de CPU (%)'],
        var_name='Métrica', value_name='Porcentagem'
    )
    df_melted_cpu['Métrica'] = df_melted_cpu['Métrica'].replace({
        'Uso Medio de CPU (%)': 'Uso Médio',
        'Pico de CPU (%)': 'Pico de Uso'
    })
    ax4 = sns.barplot(
        x='Porcentagem', y='Algoritmo', hue='Métrica', 
        data=df_melted_cpu, palette=['#66c2a5', '#fc8d62']
    )
    sns.despine()
    plt.title("Uso de CPU", fontsize=12, weight='bold', pad=15)
    plt.xlabel("Porcentagem (%)")
    plt.ylabel("")
    plt.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none')
    for container in ax4.containers:
        ax4.bar_label(container, fmt='%.1f%%', padding=4, fontsize=9, weight='bold')
    plt.tight_layout()
    plt.savefig(pasta_perf_graficos / f"cpu_perfil_comparacao{sufixo}.png", dpi=300)
    plt.close()


modelos_arquivos = [
    "modelo_decision_tree.pkl",
    "modelo_random_forest.pkl",
    "modelo_extra_trees.pkl",
    "modelo_xgboost.pkl",
    "modelo_lightgbm.pkl"
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
    print("[2] Experimento 2 - Generalização")
    print("=" * 60 + "\n")

    experimento = input("Escolha o Experimento (1 ou 2): ").strip()

    x, y = None, None
    sufixo_relatorio = ""

    if not (PASTA_MODELOS / 'classes.pkl').exists() or not (PASTA_MODELOS / 'scaler.pkl').exists():
        print(f"Erro Crítico: Arquivos 'classes.pkl' ou 'scaler.pkl' ausentes in {PASTA_MODELOS}")
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
    dict_probabilidades_composta = {}
    
    for arquivo in modelos_arquivos:
        if not (PASTA_MODELOS / arquivo).exists():
            continue

        nome_clean = arquivo.replace('modelo_', '').replace('.pkl', '').replace('_', ' ').title()
        resp = input(f"\nDeseja avaliar o modelo '{nome_clean}'? (y/n) [Default=y]: ").strip().lower()

        if resp == 'y' or not resp:
            try:
                res_metrics, probs = testar_modelo(arquivo, x, y, PASTA_MODELOS, PASTA_RELATORIOS, PASTA_CM_GRAFICOS, sufixo_relatorio)
                resultados.append(res_metrics)
                dict_probabilidades_composta[nome_clean] = probs
            except Exception as e:
                print(f"  Erro crítico ao avaliar o modelo {arquivo}: {e}")

    if resultados:
        df_resultados = pd.DataFrame(resultados)
        print("\n" + "="*80)
        print(" TABELA COMPARATIVA DE PERFORMANCE DOS MODELOS (TESTE)")
        print("="*80)
        print(df_resultados.sort_values(by="F1-Score", ascending=False).to_string(index=False))
        print("="*80)

        nome_arquivo_csv = f"metricas_comparativas_teste{sufixo_relatorio}.csv"
        df_resultados.to_csv(PASTA_RELATORIOS / nome_arquivo_csv, index=False, encoding="utf-8")
        print(f"-> Resumo comparativo das métricas salvo em CSV: {nome_arquivo_csv}")

        gerar_graficos_performance(df_resultados, PASTA_PERF_GRAFICOS, sufixo_relatorio)
        gerar_curva_roc_composta(dict_probabilidades_composta, y, classes, PASTA_PERF_GRAFICOS, sufixo_relatorio)
        print(f"\nAvaliação concluída. Todos os gráficos salvos com sucesso em: {PASTA_PERF_GRAFICOS}")
    else:
        print("\nNenhum modelo foi selecionado para o teste.")