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

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from imblearn.over_sampling import SMOTE


def carregar_dados_parquet(caminho_arquivo: Path) -> pd.DataFrame:
    print(f"\n[I/O] Carregando {caminho_arquivo.name} com Polars...")
    df_pl = pl.read_parquet(caminho_arquivo)
    return df_pl.to_pandas()


class MonitorHardware:
    """Classe responsável por monitorar o uso de CPU em paralelo ao treinamento."""
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

        pico_cpu = max(self.historico_cpu)
        media_cpu = statistics.mean(self.historico_cpu)
        return media_cpu, pico_cpu


def treinar_modelo(nome: str, modelo, x_train: np.ndarray, y_train: np.ndarray) -> dict:
    print(f"\n--- Treinando modelo: {nome} ---")
    resultados_internos = {}
    monitor_cpu = MonitorHardware(intervalo=0.05)

    def treinar():
        monitor_cpu.iniciar()
        inicio = time.time()
        modelo.fit(x_train, y_train)
        fim = time.time()
        resultados_internos['tempo'] = fim - inicio

    uso_memoria = memory_usage(treinar, interval=0.1)
    media_cpu, pico_cpu = monitor_cpu.parar()

    pico_ram = max(uso_memoria) - min(uso_memoria)
    media_ram = statistics.mean(uso_memoria) - min(uso_memoria)
    tempo_treino = resultados_internos.get('tempo', 0)

    print(f"  > Tempo: {tempo_treino:.2f}s | Pico RAM: {pico_ram:.2f} MB | Pico CPU: {pico_cpu:.1f}%")

    metricas = {
        "Algoritmo": nome,
        "Tempo de Treino (s)": tempo_treino,
        "Uso Medio de Memoria (MB)": media_ram,
        "Pico de RAM (MB)": pico_ram,
        "Uso Medio de CPU (%)": media_cpu,
        "Pico de CPU (%)": pico_cpu
    }
    return metricas


def salvar_relatorios_finais(resultados_df: pd.DataFrame, distribuicao_df: pd.DataFrame, pasta_relatorios: Path, tipo_alvo: str):
    caminho_csv_hardware = pasta_relatorios / "metricas_hardware.csv"
    caminho_csv_classes = pasta_relatorios / "distribuicao_classes.csv"
    caminho_txt = pasta_relatorios / "relatorio_treinamento.txt"

    resultados_df.to_csv(caminho_csv_hardware, index=False, encoding="utf-8")
    distribuicao_df.to_csv(caminho_csv_classes, index=False, encoding="utf-8")

    print(f"-> Métricas de hardware salvas em CSV: {caminho_csv_hardware.name}")
    print(f"-> Distribuição de classes salva em CSV: {caminho_csv_classes.name}")

    with open(caminho_txt, "w", encoding="utf-8") as f:
        f.write("=" * 65 + "\n")
        f.write(f" RELATÓRIO DO EXPERIMENTO DE TREINAMENTO - [{tipo_alvo.upper()}]\n")
        f.write(f" Executado em: {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}\n")
        f.write("=" * 65 + "\n\n")

        f.write("1. DISTRIBUIÇÃO DAS CLASSES NO CONJUNTO DE TREINO\n")
        f.write("-" * 65 + "\n")
        f.write(distribuicao_df.to_string(index=False))
        f.write("\n\n")

        f.write("2. DESEMPENHO E CONSUMO DE HARDWARE POR ALGORITMO\n")
        f.write("-" * 65 + "\n")
        f.write(resultados_df.sort_values(by="Tempo de Treino (s)").to_string(index=False))
        f.write("\n\n" + "=" * 65 + "\n")

    print(f"-> Relatório textual consolidado salvo em: {caminho_txt.name}")


def main():
    parser = argparse.ArgumentParser(description="Pipeline de Treinamento Unificado")
    parser.add_argument("--tipo", choices=["binario", "multiclasse"], required=True, help="Tipo de abordagem de dados")
    args = parser.parse_args()

    BASE_DIR = Path(__file__).resolve().parent
    PASTA_OURO = BASE_DIR / "data" / "ouro" / args.tipo
    PASTA_MODELOS = BASE_DIR / "modelos" / args.tipo
    PASTA_GRAFICOS = BASE_DIR / "graficos" / "treinamento" / args.tipo
    PASTA_RELATORIOS = BASE_DIR / "relatorios" / "treinamento" / args.tipo

    for pasta in [PASTA_MODELOS, PASTA_GRAFICOS, PASTA_RELATORIOS]:
        pasta.mkdir(parents=True, exist_ok=True)

    caminho_arquivo = PASTA_OURO / "treino_final_gold.parquet"

    print(f"\n{'='*70}\n INICIANDO WORKFLOW DE TREINAMENTO: [{args.tipo.upper()}]\n{'='*70}")
    print(f"[INFO] Buscando base de dados ouro em: {caminho_arquivo}")

    if not caminho_arquivo.exists():
        print(f"Erro Crítico: O arquivo '{caminho_arquivo}' não foi localizado. Garanta a execução das camadas anteriores.")
        sys.exit(1)

    df = carregar_dados_parquet(caminho_arquivo)

    print(f"\n[INFO] O dataset carregado possui {len(df):,} linhas.")
    print("Deseja aplicar uma amostragem (downsampling) para o treino rodar mais rápido?")
    print("[1] Não, quero usar a base cheia")
    print("[2] Usar amostra de 1.000.000 de linhas")
    print("[3] Usar amostra de 500.000 de linhas")
    print("[4] Usar amostra de 100.000 de linhas")

    opcaos_sample = input("Escolha uma opção (Padrão=1): ").strip()

    if opcaos_sample == "2" and len(df) > 1000000:
        df = df.sample(n=1000000, random_state=42)
    elif opcaos_sample == "3" and len(df) > 500000:
        df = df.sample(n=500000, random_state=42)
    elif opcaos_sample == "4" and len(df) > 100000:
        df = df.sample(n=100000, random_state=42)

    le = LabelEncoder()
    df['Label'] = le.fit_transform(df['Label'].astype(str))
    mapeamento_classes = list(le.classes_)
    joblib.dump(mapeamento_classes, PASTA_MODELOS / 'classes.pkl')

    print("\nClasses de ataques indexadas:")
    for i, classe in enumerate(mapeamento_classes):
        print(f" [{i}] {classe}")

    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        print(f"\n- Removendo {len(constant_cols)} colunas irrelevantes/constantes para otimização.")
        df.drop(columns=constant_cols, inplace=True)

    x = df.drop('Label', axis=1)
    y = df['Label']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)
    distribuicao_antes = y_train.value_counts().to_dict()

    print("\n" + "-" * 45)
    print(" MENU DE REBALANCEAMENTO DA BASE DE TREINO")
    print("-" * 45)

    if args.tipo == "multiclasse":
        print("-[1] Manter desbalanceado")
        print("-[2] Limitar BENIGN ate 1.500.000 (Undersampling) + SMOTE nas minoritárias")
        escolha_bal = input("Escolha uma opção (Padrão=1): ").strip()

        if escolha_bal == "2":
            idx_benign = mapeamento_classes.index("BENIGN")
            teto_benign = 1200000
            
            # --- PASSO 1: Undersampling focado exclusivamente na classe BENIGN ---
            print(f"\n> Verificando teto limite da classe BENIGN ({teto_benign:,} amostras)...")
            
            indices_benign = np.where(y_train == idx_benign)[0]
            indices_ataques = np.where(y_train != idx_benign)[0]
            
            if len(indices_benign) > teto_benign:
                print(f"  -> Classe BENIGN reduzida de {len(indices_benign):,} para {teto_benign:,}")
                
                np.random.seed(42)
                indices_benign_reduzidos = np.random.choice(indices_benign, size=teto_benign, replace=False)
                indices_finais = np.concatenate([indices_benign_reduzidos, indices_ataques])
                
                if isinstance(x_train, pd.DataFrame):
                    x_train = x_train.iloc[indices_finais].reset_index(drop=True)
                    y_train = y_train.iloc[indices_finais].reset_index(drop=True)
                else:
                    x_train = x_train[indices_finais]
                    y_train = y_train[indices_finais]
            else:
                print(f"  -> Classe BENIGN possui {len(indices_benign):,} amostras (Abaixo do teto). Nenhuma redução necessária.")

            # --- PASSO 2: SMOTE regular nas classes que sobraram abaixo do piso ---
            print("\n> Aplicando SMOTE nas classes minoritárias...")
            contagem_classes = y_train.value_counts().to_dict()
            piso_amostras = 20000

            esg_smote = {
                classe: max(qtd, piso_amostras)
                for classe, qtd in contagem_classes.items()
                if qtd < piso_amostras
            }

            if esg_smote:
                menor_classe = min(contagem_classes.values())
                k_neighbors = min(5, max(1, menor_classe - 1))

                print(f"  -> Classes que receberão SMOTE: {list(esg_smote.keys())}")
                print(f"  -> k_neighbors utilizado: {k_neighbors}")

                try:
                    smote = SMOTE(sampling_strategy=esg_smote, k_neighbors=k_neighbors, random_state=42)
                    x_train, y_train = smote.fit_resample(x_train, y_train)
                    print("  -> SMOTE concluído! Classes grandes e BENIGN tratadas preservadas.")
                except Exception as e:
                    print(f"  -> SMOTE falhou: {e}. Mantendo dados antigos.")
            else:
                print("  -> Nenhuma classe abaixo do piso definido para SMOTE.")

    else:
        print("-[1] Manter desbalanceado")
        print("-[2] Aplicar técnica Híbrida Customizada (Aproximação controlada das classes)")
        escolha_bal = input("Escolha uma opção (Padrão=1): ").strip()

        if escolha_bal == "2":
            from imblearn.under_sampling import RandomUnderSampler
            print("\n> Aplicando rebalanceamento híbrido binário...")
            contagem_classes = y_train.value_counts().to_dict()
            idx_benign = mapeamento_classes.index("BENIGN")
            idx_attack = mapeamento_classes.index("ATTACK")

            qtd_benign_original = contagem_classes[idx_benign]
            qtd_attack_original = contagem_classes[idx_attack]

            # Alvos para trazer as classes para mais perto de forma suave:
            # Reduzimos BENIGN para um teto fixo e aproximamos o ATTACK com SMOTE para manter um balanceamento saudável (ex: ~60% / 40%)
            teto_benign = min(1600000, int(qtd_benign_original * 0.5)) # Desce o benigno de forma segura
            alvo_attack = int(teto_benign * 0.65)                     # Eleva o ataque proporcionalmente mais perto da classe limpa

            # Passo 1: Undersampling na classe majoritária se necessário
            if qtd_benign_original > teto_benign:
                print(f"  -> Aplicando Undersampling: BENIGN reduzida de {qtd_benign_original:,} para {teto_benign:,}")
                rus = RandomUnderSampler(sampling_strategy={idx_benign: teto_benign, idx_attack: qtd_attack_original}, random_state=42)
                x_train, y_train = rus.fit_resample(x_train, y_train)
            
            # Passo 2: SMOTE na classe minoritária para aproximá-las de verdade
            if qtd_attack_original < alvo_attack:
                print(f"  -> Aplicando SMOTE: ATTACK expandido de {qtd_attack_original:,} para {alvo_attack:,}")
                smote = SMOTE(sampling_strategy={idx_attack: alvo_attack}, random_state=42)
                x_train, y_train = smote.fit_resample(x_train, y_train)

            print("  -> Balanceamento de aproximação binária concluído!")

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    joblib.dump((x_test_scaled, y_test, list(x.columns)), PASTA_MODELOS / 'holdout_teste.pkl')
    joblib.dump(scaler, PASTA_MODELOS / 'scaler.pkl')

    valores, contagens = np.unique(y_train, return_counts=True)
    distribuicao_depois = dict(zip(valores, contagens))

    dados_distribuicao = []
    print("\n" + "="*70)
    print(" DISTRIBUIÇÃO DAS CLASSES NO CONJUNTO DE TREINO FINAL")
    print("="*70)
    print(f"{'Classe / Label':<22} | {'Antes do Bal.':<16} | {'Após Bal. (Final)':<20}")
    print("-"*70)
    for idx, classe_nome in enumerate(mapeamento_classes):
        qtd_antes = distribuicao_antes.get(idx, 0)
        qtd_depois = distribuicao_depois.get(idx, 0)
        print(f"{classe_nome:<22} | {qtd_antes:<16,} | {qtd_depois:<20,}")
        dados_distribuicao.append({"Classe": classe_nome, "Antes do Bal.": qtd_antes, "Após Bal. (Final)": qtd_depois})
    print("-" * 70)
    df_dist_final = pd.DataFrame(dados_distribuicao)

    print(f"\n[Sucesso] Validação interna armazenada com {x_test_scaled.shape[0]:,} amostras.")
    print(f"[Matriz] Dimensões finais de treino: Amostras={len(y_train):,} | Features={x_train_scaled.shape[1]}")

    modelos = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=50, n_jobs=-1, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=50, max_depth=6, n_jobs=-1, random_state=42, eval_metric='mlogloss'),
        "LightGBM": LGBMClassifier(n_estimators=50, max_depth=6, n_jobs=-1, random_state=42, verbosity=-1)
    }

    resultados = []

    for nome, modelo in modelos.items():
        res = input(f"\nDeseja treinar o modelo {nome}? (y/n) [Default=y]: ").strip().lower()
        if res == "y" or not res:
            metrics_resultado = treinar_modelo(nome, modelo, x_train_scaled, y_train)
            nome_arquivo = f"modelo_{nome.replace(' ', '_').lower()}.pkl"
            joblib.dump(modelo, PASTA_MODELOS / nome_arquivo)
            print(f"  -> Exportado: {nome_arquivo}")
            resultados.append(metrics_resultado)

    if not resultados:
        print("\n[Aviso] Nenhum algoritmo foi selecionado. Encerrando pipeline.")
        sys.exit(0)

    df_resultados = pd.DataFrame(resultados)
    print("\n" + "="*58)
    print(" PERFORMANCE DE HARDWARE DURANTE O TREINAMENTO")
    print("="*58)
    print(df_resultados.sort_values(by="Tempo de Treino (s)").to_string(index=False))
    print("="*58)

    salvar_relatorios_finais(df_resultados, df_dist_final, PASTA_RELATORIOS, args.tipo)

    sns.set_theme(style="whitegrid", palette="muted")

    plt.figure(figsize=(9, 5))
    df_tempo = df_resultados.sort_values(by="Tempo de Treino (s)")
    ax1 = sns.barplot(x="Tempo de Treino (s)", y="Algoritmo", data=df_tempo, hue="Algoritmo", palette="magma", legend=False)
    plt.title("Tempo de Treinamento por Algoritmo", fontsize=11, weight='bold')
    plt.xlabel("Segundos (s)")
    plt.ylabel("")
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.2fs', padding=5, weight='bold')
    plt.tight_layout()
    plt.savefig(PASTA_GRAFICOS / "tempo_treinamento.png", dpi=300)
    plt.close()

    plt.figure(figsize=(9, 5))
    df_pico = df_resultados.sort_values(by="Pico de RAM (MB)", ascending=False)
    ax2 = sns.barplot(x="Pico de RAM (MB)", y="Algoritmo", data=df_pico, hue="Algoritmo", palette="viridis", legend=False)
    plt.title("Pico de Memória RAM por Algoritmo", fontsize=11, weight='bold')
    plt.xlabel("Megabytes (MB)")
    plt.ylabel("")
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%.2f MB', padding=5, weight='bold')
    plt.tight_layout()
    plt.savefig(PASTA_GRAFICOS / "pico_ram.png", dpi=300)
    plt.close()

    plt.figure(figsize=(9, 5))
    df_media = df_resultados.sort_values(by="Uso Medio de Memoria (MB)", ascending=False)
    ax3 = sns.barplot(x="Uso Medio de Memoria (MB)", y="Algoritmo", data=df_media, hue="Algoritmo", palette="crest", legend=False)
    plt.title("Consumo Médio de Memória RAM por Algoritmo", fontsize=11, weight='bold')
    plt.xlabel("Megabytes (MB)")
    plt.ylabel("")
    for container in ax3.containers:
        ax3.bar_label(container, fmt='%.2f MB', padding=5, weight='bold')
    plt.tight_layout()
    plt.savefig(PASTA_GRAFICOS / "consumo_medio_ram.png", dpi=300)
    plt.close()

    plt.figure(figsize=(9, 5))
    df_cpu_pico = df_resultados.sort_values(by="Pico de CPU (%)", ascending=False)
    ax4 = sns.barplot(x="Pico de CPU (%)", y="Algoritmo", data=df_cpu_pico, hue="Algoritmo", palette="flare", legend=False)
    plt.title("Pico de Uso de CPU por Algoritmo", fontsize=11, weight='bold')
    plt.xlabel("Porcentagem de Uso (%)")
    plt.ylabel("")
    for container in ax4.containers:
        ax4.bar_label(container, fmt='%.1f%%', padding=5, weight='bold')
    plt.tight_layout()
    plt.savefig(PASTA_GRAFICOS / "pico_cpu.png", dpi=300)
    plt.close()

    plt.figure(figsize=(9, 5))
    df_cpu_med = df_resultados.sort_values(by="Uso Medio de CPU (%)", ascending=False)
    ax5 = sns.barplot(x="Uso Medio de CPU (%)", y="Algoritmo", data=df_cpu_med, hue="Algoritmo", palette="vlag", legend=False)
    plt.title("Consumo Médio de CPU por Algoritmo", fontsize=11, weight='bold')
    plt.xlabel("Porcentagem de Uso (%)")
    plt.ylabel("")
    for container in ax5.containers:
        ax5.bar_label(container, fmt='%.1f%%', padding=5, weight='bold')
    plt.tight_layout()
    plt.savefig(PASTA_GRAFICOS / "consumo_medio_cpu.png", dpi=300)
    plt.close()

    print(f"\n[Sucesso] 5 Imagens estatísticas individuais salvas em: {PASTA_GRAFICOS}")


if __name__ == '__main__':
    main()