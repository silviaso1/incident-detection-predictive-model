
# MACHINE LEARNING PARA DETECÇÃO DE CIBERATAQUES: UMA ANÁLISE DE CUSTO-BENEFÍCIO DE ALGORITMOS BASEADOS EM ÁRVORES DE DECISÃO PARA AMBIENTES COM RECURSOS LIMITADOS

Este repositório contém o pipeline completo de engenharia de dados, balanceamento estatístico, treinamento e inferência para detecção de ciberataques, divididos entre abordagens **Binária** e **Multiclasse**.

---

## Datasets Utilizados

* **Treinamento:** [CIC-IDS-Collection](https://www.kaggle.com/datasets/dhoogla/cicidscollection) (Base consolidada e unificada em formato `.parquet`).
* **Testes / Validação de Generalização:** [CIC-IDS-2017](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset) (Arquivos `.csv`).

## Arquitetura do Pipeline de Dados

O projeto adota o conceito de arquitetura medalhão utilizando a biblioteca **Polars**, garantindo baixo consumo de memória mesmo lidando com milhões de linhas.

1. **Bronze (`bronze.py`):** Higienização de strings, padronização de cabeçalhos e decodificação de caracteres malformados. Coloque todas as bases juntas aqui.
2. **Silver (`prata.py`):** Processo de **Anti-Join Nativo**. Como a base de treinamento (`CIC-IDS-Collection`) é uma compilação histórica que abrange múltiplos anos e inclui dados do próprio framework de 2017, existe o risco inerente de sobreposição amostral. Para eliminar o vazamento de dados, o script executa um cruzamento e remove do conjunto de treino todas as assinaturas idênticas presentes no conjunto de teste.
3. **Gold:** Dados purificados prontos para uso no treinamento e teste
---

## Rodando o projeto

### Clonar o Repositório
Abra o seu terminal na pasta onde deseja guardar o projeto e execute o comando abaixo:
```bash
# Baixar o repositório para a sua máquina local
git clone https://github.com/silviaso1/incident-detection-predictive-model.git

# Entrar na pasta do projeto que foi criada pelo Git
cd incident-detection-predictive-model
```
### Executar o código

Novamente no terminal, execute o seguinte comando:

```bash
python main.py
```
Um menu será aberto para fazer cada etapa do pipeline. No entanto, caso prefira rodar manualmente, é aceito dois tipos de argumentos. Para isso, digite nomedoarquivo.py --tipo binario ou --tipo multiclasse

### Visualizar os resultados

Após todas as fases do pipeline, digite o seguinte comando no terminal para ver o dashboard

```bash
streamlit run dashboard.py
```
