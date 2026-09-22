# Tratamento e analise de dados com Pandas

Projeto de tratamento e analise exploratoria usando Pandas.

## Dataset

- Nome: Wine Quality - Red Wine
- Fonte: UCI Machine Learning Repository
- Arquivo bruto: `data/winequality-red.csv`
- Arquivo tratado gerado pelo notebook: `data/winequality-red-clean.csv`
- URL de origem: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv

O dataset contem amostras de vinho tinto com variaveis fisico-quimicas, como acidez, densidade, sulfatos, alcool e a nota de qualidade.

## Arquivos principais

- `projeto.txt`: enunciado da atividade.
- `notebooks/analise_wine_quality.ipynb`: notebook com descricao, tratamento e respostas das 10 perguntas.
- `data/winequality-red.csv`: dataset bruto.
- `data/winequality-red-clean.csv`: dataset tratado pelo notebook.
- `scripts/build_notebook.py`: script usado para gerar e executar o notebook.

## Como executar

Instale as dependencias:

```bash
pip install -r requirements.txt
```

Execute o notebook no Jupyter ou regenere tudo com:

```bash
python scripts/build_notebook.py
```
