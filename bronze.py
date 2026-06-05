import os
import sys
import glob
import re
import argparse
from pathlib import Path
import polars as pl

CANONICAL_COLUMNS = [
    "Dst Port", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts",
    "Fwd Pkt Len Max", "Fwd Pkt Len Min", "Fwd Pkt Len Mean", "Fwd Pkt Len Std",
    "Bwd Pkt Len Max", "Bwd Pkt Len Min", "Bwd Pkt Len Mean", "Bwd Pkt Len Std",
    "Flow Byts/s", "Flow Pkts/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Tot", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Tot", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Len", "Bwd Header Len",
    "Fwd Pkts/s", "Bwd Pkts/s", "Pkt Len Min", "Pkt Len Max", "Pkt Len Mean", "Pkt Len Std", "Pkt Len Var",
    "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt", "ACK Flag Cnt", "URG Flag Cnt",
    "CWE Flag Count", "ECE Flag Cnt", "Down/Up Ratio", "Pkt Size Avg", "Fwd Seg Size Avg", "Bwd Seg Size Avg",
    "Fwd Byts/b Avg", "Fwd Pkts/b Avg", "Fwd Blk Rate Avg", "Bwd Byts/b Avg", "Bwd Pkts/b Avg", "Bwd Blk Rate Avg",
    "Subflow Fwd Pkts", "Subflow Fwd Bytes", "Subflow Bwd Pkts", "Subflow Bwd Byts",
    "Init Fwd Win Byts", "Init Bwd Win Byts", "Fwd Act Data Pkts", "Fwd Seg Size Min",
    "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
    "Label",
]

RENAME_2017 = {
    "Destination Port":           "Dst Port",
    "Total Fwd Packets":          "Tot Fwd Pkts",
    "Total Backward Packets":    "Tot Bwd Pkts",
    "Total Length of Fwd Packets": "TotLen Fwd Pkts",
    "Total Length of Bwd Packets": "TotLen Bwd Pkts", 
    "Fwd Packet Length Max":     "Fwd Pkt Len Max",
    "Fwd Packet Length Min":     "Fwd Pkt Len Min",
    "Fwd Packet Length Mean":    "Fwd Pkt Len Mean",
    "Fwd Packet Length Std":     "Fwd Pkt Len Std",
    "Bwd Packet Length Max":     "Bwd Pkt Len Max",
    "Bwd Packet Length Min":     "Bwd Pkt Len Min",
    "Bwd Packet Length Mean":    "Bwd Pkt Len Mean",
    "Bwd Packet Length Std":     "Bwd Pkt Len Std",
    "Flow Bytes/s":              "Flow Byts/s",
    "Flow Packets/s":            "Flow Pkts/s",
    "Fwd IAT Total":             "Fwd IAT Tot",
    "Bwd IAT Total":             "Bwd IAT Tot",
    "Fwd Header Length":         "Fwd Header Len",
    "Bwd Header Length":         "Bwd Header Len",
    "Fwd Packets/s":             "Fwd Pkts/s",
    "Bwd Packets/s":             "Bwd Pkts/s",
    "Min Packet Length":         "Pkt Len Min",
    "Max Packet Length":         "Pkt Len Max",
    "Packet Length Mean":        "Pkt Len Mean",
    "Packet Length Std":         "Pkt Len Std",
    "Packet Length Variance":    "Pkt Len Var",
    "FIN Flag Count":            "FIN Flag Cnt",
    "SYN Flag Count":            "SYN Flag Cnt",
    "RST Flag Count":            "RST Flag Cnt",
    "PSH Flag Count":            "PSH Flag Cnt",
    "ACK Flag Count":            "ACK Flag Cnt",
    "URG Flag Count":            "URG Flag Cnt",
    "ECE Flag Count":            "ECE Flag Cnt",
    "Average Packet Size":       "Pkt Size Avg",
    "Avg Fwd Segment Size":      "Fwd Seg Size Avg",
    "Avg Bwd Segment Size":      "Bwd Seg Size Avg",
    "Fwd Avg Bytes/Bulk":        "Fwd Byts/b Avg",
    "Fwd Avg Packets/Bulk":      "Fwd Pkts/b Avg",
    "Fwd Avg Bulk Rate":         "Fwd Blk Rate Avg",
    "Bwd Avg Bytes/Bulk":        "Bwd Byts/b Avg",
    "Bwd Avg Packets/Bulk":      "Bwd Pkts/b Avg",
    "Bwd Avg Bulk Rate":         "Bwd Blk Rate Avg",
    "Subflow Fwd Packets":       "Subflow Fwd Pkts",
    "Subflow Fwd Bytes":         "Subflow Fwd Byts",
    "Subflow Bwd Packets":       "Subflow Bwd Pkts",
    "Subflow Bwd Bytes":         "Subflow Bwd Byts",
    "Init_Win_bytes_forward":    "Init Fwd Win Byts",
    "Init_Win_bytes_backward":   "Init Bwd Win Byts",
    "act_data_pkt_fwd":          "Fwd Act Data Pkts",
    "min_seg_size_forward":      "Fwd Seg Size Min",
    "Label":                     "Label",
}

RAW_LABEL_MAP = {
    "BENIGN": "BENIGN", "FTP-PATATOR": "BRUTE-FORCE", "SSH-PATATOR": "BRUTE-FORCE",
    "FTP-BRUTEFORCE": "BRUTE-FORCE", "SSH-BRUTEFORCE": "BRUTE-FORCE", "BRUTEFORCE": "BRUTE-FORCE",
    "BRUTE FORCE": "BRUTE-FORCE", "WEB ATTACK - BRUTE FORCE": "BRUTE-FORCE", "BRUTE FORCE -WEB": "BRUTE-FORCE",
    "WEBATTACKBRUTEFORCE": "BRUTE-FORCE", "BRUTEFORCE-FTP": "BRUTE-FORCE", "BRUTEFORCE-SSH": "BRUTE-FORCE",
    "WEB ATTACK - XSS": "WEB-ATTACK", "WEB ATTACK - SQL INJECTION": "WEB-ATTACK", "BRUTE FORCE - XSS": "WEB-ATTACK",
    "XSS": "WEB-ATTACK", "SQL INJECTION": "WEB-ATTACK", "SQLINJECTION": "WEB-ATTACK", "WEBATTACKXSS": "WEB-ATTACK",
    "WEBATTACKSQLINJECTION": "WEB-ATTACK", "DOS HULK": "DOS", "DOS-HULK": "DOS", "DOSHULK": "DOS",
    "DOS GOLDENEYE": "DOS", "DOS-GOLDENEYE": "DOS", "DOSGOLDENEYE": "DOS", "DOS SLOWLORIS": "DOS",
    "DOS-SLOWLORIS": "DOS", "DOSSLOWLORIS": "DOS", "DOS SLOWHTTPTEST": "DOS", "DOS-SLOWHTTPTEST": "DOS",
    "DOSSLOWHTTPTEST": "DOS", "DOS ATTACKS-HULK": "DOS", "DOS ATTACKS-GOLDENEYE": "DOS",
    "DOS ATTACKS-SLOWLORIS": "DOS", "DOS ATTACKS-SLOWHTTPTEST": "DOS", "DDOS": "DDOS",
    "DDOS ATTACK-HOIC": "DDOS", "DDOS ATTACKS-LOIC-HTTP": "DDOS", "DDOS ATTACK-LOIC-UDP": "DDOS",
    "DDOS-HOIC": "DDOS", "DDOS-LOIC-HTTP": "DDOS", "DDOS-LOIC-UDP": "DDOS", "DDOS-DNS": "DDOS",
    "DDOS-NTP": "DDOS", "DDOS-UDP": "DDOS", "DDOS-SYN": "DDOS", "DDOS-MSSQL": "DDOS",
    "BOT": "BOTNET", "BOTNET": "BOTNET", "PORTSCAN": "PORTSCAN", "INFILTRATION": "INFILTRATION",
    "INFILTERATION": "INFILTRATION", "HEARTBLEED": "OTHER"
}

def normalizar_string_chave(s: str) -> str:
    s_clean = str(s).upper()
    s_clean = re.sub(r'[^A-Z0-9]', '', s_clean)
    if "WEBATTACK" in s_clean or "XSS" in s_clean or "SQL" in s_clean:
        if "BRUTE" in s_clean: return "WEBATTACKBRUTEFORCE"
        if "XSS" in s_clean: return "WEBATTACKXSS"
        if "SQL" in s_clean: return "WEBATTACKSQLINJECTION"
    return s_clean

LABEL_MAP_NORMALIZADO = {normalizar_string_chave(k): v for k, v in RAW_LABEL_MAP.items()}

def limpar_e_retornar_lazy(input_path: str) -> pl.LazyFrame:
    if input_path.lower().endswith('.parquet'):
        lf = pl.scan_parquet(input_path)
    else:
        lf = pl.scan_csv(input_path, encoding="utf8-lossy", infer_schema_length=0)
    
    schema_cols = lf.collect_schema().names()
    lf = lf.rename({c: c.strip().replace('"', '').replace("'", "") for c in schema_cols})
    cols_limpas = lf.collect_schema().names()

    is_2017 = "Destination Port" in cols_limpas or "Fwd Header Length.1" in cols_limpas
    
    if is_2017:
        if "Fwd Header Length.1" in cols_limpas:
            lf = lf.drop("Fwd Header Length.1")
        rename_dict = {k: v for k, v in RENAME_2017.items() if k in cols_limpas}
        lf = lf.rename(rename_dict)

    if "Label" in lf.collect_schema().names():
        lf = lf.filter((pl.col("Label").cast(pl.String).str.strip_chars() != "Label") & (pl.col("Label").is_not_null()))
        
    return lf

def aplicar_transformacoes_finais(lf: pl.LazyFrame, colunas_comuns: list, tipo_alvo: str) -> pl.LazyFrame:
    lf = lf.with_columns(
        pl.col("Label")
        .cast(pl.String)
        .str.to_uppercase()
        .str.replace_all(r"[^A-Za-z0-9]", "")
        .alias("Label_Lookup")
    )
    
    df_map = pl.DataFrame({
        "Chave_Lookup": list(LABEL_MAP_NORMALIZADO.keys()),
        "Label_Mapeado": list(LABEL_MAP_NORMALIZADO.values())
    }).lazy()

    lf = lf.join(df_map, left_on="Label_Lookup", right_on="Chave_Lookup", how="left")

    lf = lf.with_columns(pl.col("Label_Mapeado").fill_null("UNKNOWN").alias("Label"))

    lf = lf.filter((pl.col("Label") != "OTHER") & (pl.col("Label") != "UNKNOWN"))

    if tipo_alvo == "binario":
        lf = lf.with_columns(
            pl.when(pl.col("Label") == "BENIGN")
            .then(pl.lit("BENIGN"))
            .otherwise(pl.lit("ATTACK"))
            .alias("Label")
        )

    cols_finais = [c for c in colunas_comuns if c in lf.collect_schema().names()]
    lf = lf.select(cols_finais)
    
    feature_cols = [c for c in cols_finais if c != "Label"]
    
    expressions = []
    for c in feature_cols:
        col_cast = pl.col(c).cast(pl.Float64, strict=False)
        expr = (
            pl.when(col_cast.is_infinite() | col_cast.is_nan())
            .then(0.0)
            .otherwise(col_cast)
            .fill_null(0.0)
            .alias(c)
        )
        expressions.append(expr)
        
    lf = lf.with_columns(expressions)
    lf = lf.unique(subset=feature_cols)
    
    return lf
def pre_scan_colunas_comuns(bronze_dir: str) -> list:
    arquivos_treino = glob.glob(os.path.join(bronze_dir, "*.parquet"))
    arquivos_teste = glob.glob(os.path.join(bronze_dir, "*.csv"))
    
    if not arquivos_treino:
        print("Erro Crítico: Nenhum arquivo .parquet de Treino localizado em bronze.")
        sys.exit(1)
        
    cols_treino = set()
    for f in arquivos_treino:
        cols_treino.update(pl.scan_parquet(f).collect_schema().names())
    
    cols_teste_unificadas = set()
    if arquivos_teste:
        schema_csv = pl.scan_csv(arquivos_teste[0], encoding="utf8-lossy", infer_schema_length=0).collect_schema().names()
        for c in schema_csv:
            c_clean = c.strip().replace('"', '').replace("'", "")
            cols_teste_unificadas.add(RENAME_2017.get(c_clean, c_clean))
    else:
        return CANONICAL_COLUMNS
            
    colunas_comuns = [c for c in CANONICAL_COLUMNS if c in cols_treino and (c in cols_teste_unificadas or c == "Label")]
    return colunas_comuns

def main():
    parser = argparse.ArgumentParser(description="Pipeline Bronze - Higienização")
    parser.add_argument("--tipo", choices=["binario", "multiclasse"], required=True, help="Tipo de rotulagem alvo")
    args = parser.parse_args()

    BRONZE_DIR = os.path.join("data", "bronze")
    PRATA_TARGET_DIR = os.path.join("data", "prata", args.tipo)
    os.makedirs(PRATA_TARGET_DIR, exist_ok=True)

    print(f"\n{'='*70}\n INICIANDO PROCESSO: BRONZE ➔ SILVER [{args.tipo.upper()}]\n{'='*70}")
    
    colunas_comuns = pre_scan_colunas_comuns(BRONZE_DIR)
    arquivos_csv = sorted(glob.glob(os.path.join(BRONZE_DIR, "*.csv")))
    arquivos_parquet = sorted(glob.glob(os.path.join(BRONZE_DIR, "*.parquet")))

    if arquivos_csv:
        lista_testes_processados = []
        for caminho_csv in arquivos_csv:
            print(f"  -> Processando fragmento de teste: {os.path.basename(caminho_csv)}")
            lf_csv = limpar_e_retornar_lazy(caminho_csv)
            lf_csv = aplicar_transformacoes_finais(lf_csv, colunas_comuns, args.tipo)
            df_csv = lf_csv.collect(streaming=True)
            if df_csv.height > 0:
                lista_testes_processados.append(df_csv)
                
        df_teste_total = pl.concat(lista_testes_processados)
        output_teste = os.path.join(PRATA_TARGET_DIR, "teste_2017_silver.parquet")
        df_teste_total.write_parquet(output_teste, compression="snappy")
        print(f"-> Base Silver Teste salva em: {output_teste}")
    else:
        print("Erro crítico: Nenhum CSV encontrado na camada Bronze.")
        sys.exit(1)
        
    if arquivos_parquet:
        lista_treino_processados = []
        for caminho_pq in arquivos_parquet:
            print(f"  -> Processando fragmento de treino: {os.path.basename(caminho_pq)}")
            lf_pq = limpar_e_retornar_lazy(caminho_pq)
            lf_pq = aplicar_transformacoes_finais(lf_pq, colunas_comuns, args.tipo)
            df_pq = lf_pq.collect(streaming=True)
            if df_pq.height > 0:
                lista_treino_processados.append(df_pq)
                
        df_treino_total = pl.concat(lista_treino_processados)
        output_treino = os.path.join(PRATA_TARGET_DIR, "treino_collection_intermediario.parquet")
        df_treino_total.write_parquet(output_treino, compression="snappy")
        print(f"-> Base Silver Treino Intermediária salva em: {output_treino}")

if __name__ == "__main__":
    main()