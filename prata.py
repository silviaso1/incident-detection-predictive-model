import os
import sys
import argparse
from pathlib import Path
import polars as pl

def gerar_relatorio_e_print(df_antes_count, df_depois, df_teste, tipo_alvo):
    contagem_treino_antes = dict(zip(df_antes_count["Label"].to_list(), df_antes_count["count"].to_list()))
    
    contagem_treino = df_depois["Label"].value_counts().sort("count", descending=True)
    contagem_teste = df_teste["Label"].value_counts().sort("count", descending=True)
    
    dict_treino = dict(zip(contagem_treino["Label"].to_list(), contagem_treino["count"].to_list()))
    dict_teste = dict(zip(contagem_teste["Label"].to_list(), contagem_teste["count"].to_list()))
    
    all_classes = sorted(list(set(list(contagem_treino_antes.keys()) + list(dict_treino.keys()) + list(dict_teste.keys()))))
    
    print("\n" + "="*88)
    print(f" RELATÓRIO QUANTITATIVO VERIFICADO PARA O SEU TCC [{tipo_alvo.upper()}]")
    print("="*88)
    print(f"{'Classe / Tipologia':<22} | {'Treino (Antes)':<16} | {'Treino (Depois)':<16} | {'Volumetria Teste':<16}")
    print("-"*88)
    
    total_antes = df_antes_count["count"].sum()
    for cl in all_classes:
        q_antes = contagem_treino_antes.get(cl, 0)
        q_treino = dict_treino.get(cl, 0)
        q_teste = dict_teste.get(cl, 0)
        print(f"{cl:<22} | {q_antes:<16,} | {q_treino:<16,} | {q_teste:<16,}")
    print("-"*88)
    print(f"{'TOTAL DE LINHAS':<22} | {total_antes:<16,} | {len(df_depois):<16,} | {len(df_teste):<16,}")
    print("="*88)

    pasta_relatorios = Path("data") / "ouro" / tipo_alvo / "relatorios"
    pasta_relatorios.mkdir(parents=True, exist_ok=True)
    contagem_treino.write_csv(pasta_relatorios / "relatorio_treino_final.csv")
    contagem_teste.write_csv(pasta_relatorios / "relatorio_teste_final.csv")

def main():
    parser = argparse.ArgumentParser(description="Pipeline Prata para Gold - Anti-Join de Alta Fidelidade")
    parser.add_argument("--tipo", choices=["binario", "multiclasse"], required=True, help="Tipo de base a consolidar")
    args = parser.parse_args()

    PRATA_DIR = Path("data") / "prata" / args.tipo
    GOLD_DIR = Path("data") / "ouro" / args.tipo
    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    path_teste = PRATA_DIR / "teste_2017_silver.parquet"
    path_treino = PRATA_DIR / "treino_collection_intermediario.parquet"

    if not path_teste.exists() or not path_treino.exists():
        print(f"Erro: Arquivos necessários não encontrados em {PRATA_DIR}. Rode o bronze.py primeiro.")
        sys.exit(1)

    print(f"\n{'='*70}\n INICIANDO PROCESSO: PRATA ➔ OURO [{args.tipo.upper()}]\n{'='*70}")

    lf_teste = pl.scan_parquet(path_teste)
    lf_treino = pl.scan_parquet(path_treino)

    
    colunas_assinatura = [c for c in lf_teste.collect_schema().names() if c != "Label"]

    lf_assinaturas_teste = lf_teste.select(colunas_assinatura).unique()

    print("  -> Contabilizando volumetria original antes da purificação...")
    df_antes_count = lf_treino.group_by("Label").agg(pl.len().alias("count")).collect(streaming=True)

    print("  -> Executando Anti-Join Nativo para expurgar sobreposições do Treino...")
    lf_treino_purificado = lf_treino.join(lf_assinaturas_teste, on=colunas_assinatura, how="anti").unique()
    lf_teste = lf_teste.unique()

    df_teste_final = lf_teste.collect(streaming=True)
    df_treino_final = lf_treino_purificado.collect(streaming=True)


    output_gold_teste = GOLD_DIR / "teste_final_gold.parquet"
    output_gold_treino = GOLD_DIR / "treino_final_gold.parquet"

    df_teste_final.write_parquet(output_gold_teste, compression="snappy")
    df_treino_final.write_parquet(output_gold_treino, compression="snappy")

    print(f"-> Dataset Ouro de Teste gravado em: {output_gold_teste}")
    print(f"-> Dataset Ouro de Treino gravado em: {output_gold_treino}")

    gerar_relatorio_e_print(df_antes_count, df_treino_final, df_teste_final, args.tipo)

if __name__ == "__main__":
    main()