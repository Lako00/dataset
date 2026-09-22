from pathlib import Path
from textwrap import dedent

import nbformat as nbf
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "analise_wine_quality.ipynb"


def md(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip())


cells = [
    md(
        """
        # Tratamento e analise de dados com Pandas

        **Dataset:** Wine Quality - Red Wine, do UCI Machine Learning Repository.

        O objetivo deste notebook e descrever os dados, identificar problemas de qualidade, aplicar tratamento com Pandas e responder 10 perguntas usando filtragens, agrupamentos e transformacoes.
        """
    ),
    md(
        """
        ## 1. Carregamento das bibliotecas e do dataset

        O arquivo bruto foi salvo na pasta `data/` do repositorio. Ele usa ponto e virgula como separador.
        """
    ),
    code(
        """
        import pandas as pd
        import matplotlib.pyplot as plt
        from IPython.display import display

        pd.set_option("display.max_columns", None)
        plt.style.use("seaborn-v0_8-whitegrid")
        """
    ),
    code(
        """
        DATA_PATH = "../data/winequality-red.csv"
        df_original = pd.read_csv(DATA_PATH, sep=";")

        print(f"Linhas: {df_original.shape[0]}")
        print(f"Colunas: {df_original.shape[1]}")
        display(df_original.head())
        """
    ),
    md(
        """
        ## 2. Descricao inicial dos dados

        A tabela abaixo resume tipos, quantidade de nulos e valores distintos por coluna.
        """
    ),
    code(
        """
        descricao_inicial = pd.DataFrame({
            "tipo": df_original.dtypes.astype(str),
            "nulos": df_original.isna().sum(),
            "valores_distintos": df_original.nunique(),
        })

        print(f"Linhas duplicadas no dataset bruto: {df_original.duplicated().sum()}")
        display(descricao_inicial)
        display(df_original.describe().T.round(3))
        """
    ),
    md(
        """
        ## 3. Tratamento de qualidade dos dados

        Etapas aplicadas:

        - Padronizacao dos nomes das colunas para `snake_case`.
        - Remocao de linhas duplicadas.
        - Remocao de colunas constantes, se existirem.
        - Preenchimento de nulos numericos pela mediana, se existirem.
        - Criacao de variaveis derivadas para apoiar a analise.
        """
    ),
    code(
        """
        df = df_original.copy()
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_", regex=False)
        )

        linhas_antes = len(df)
        duplicadas_antes = int(df.duplicated().sum())
        nulos_antes = int(df.isna().sum().sum())

        df = df.drop_duplicates().reset_index(drop=True)

        colunas_constantes = [
            coluna for coluna in df.columns
            if df[coluna].nunique(dropna=False) <= 1
        ]
        if colunas_constantes:
            df = df.drop(columns=colunas_constantes)

        for coluna in df.select_dtypes(include="number").columns:
            if df[coluna].isna().any():
                df[coluna] = df[coluna].fillna(df[coluna].median())

        df["quality_class"] = pd.cut(
            df["quality"],
            bins=[0, 5, 6, 10],
            labels=["baixa", "media", "alta"],
            include_lowest=True,
        )
        df["alcohol_level"] = pd.qcut(
            df["alcohol"],
            q=3,
            labels=["baixo", "medio", "alto"],
        )
        df["volatile_acidity_level"] = pd.qcut(
            df["volatile_acidity"],
            q=3,
            labels=["baixa", "media", "alta"],
        )
        df["sulphates_group"] = [
            "alto" if valor >= df["sulphates"].median() else "baixo"
            for valor in df["sulphates"]
        ]

        df.to_csv("../data/winequality-red-clean.csv", index=False)

        relatorio_tratamento = pd.DataFrame({
            "item": [
                "linhas_antes",
                "linhas_depois",
                "duplicadas_removidas",
                "nulos_antes",
                "nulos_depois",
                "colunas_constantes_removidas",
            ],
            "valor": [
                linhas_antes,
                len(df),
                duplicadas_antes,
                nulos_antes,
                int(df.isna().sum().sum()),
                len(colunas_constantes),
            ],
        })

        display(relatorio_tratamento)
        display(df.head())
        """
    ),
    md(
        """
        ## 4. Perguntas e respostas com Pandas
        """
    ),
    md(
        """
        ### Pergunta 1: Qual e o tamanho do dataset depois da limpeza?
        """
    ),
    code(
        """
        print(f"Dataset bruto: {df_original.shape[0]} linhas e {df_original.shape[1]} colunas.")
        print(f"Dataset tratado: {df.shape[0]} linhas e {df.shape[1]} colunas.")
        print(f"Foram removidas {duplicadas_antes} linhas duplicadas e {len(colunas_constantes)} colunas constantes.")
        """
    ),
    md(
        """
        ### Pergunta 2: Como as notas de qualidade estao distribuidas?
        """
    ),
    code(
        """
        distribuicao_qualidade = (
            df["quality"]
            .value_counts()
            .sort_index()
            .rename_axis("quality")
            .to_frame("quantidade")
        )
        distribuicao_qualidade["percentual"] = (
            distribuicao_qualidade["quantidade"] / len(df) * 100
        ).round(2)

        display(distribuicao_qualidade)

        ax = distribuicao_qualidade["quantidade"].plot(
            kind="bar",
            figsize=(7, 4),
            color="#4c78a8",
            title="Distribuicao das notas de qualidade",
        )
        ax.set_xlabel("Nota de qualidade")
        ax.set_ylabel("Quantidade de vinhos")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ### Pergunta 3: Qual e o teor alcoolico medio por nota de qualidade?
        """
    ),
    code(
        """
        alcool_por_qualidade = (
            df.groupby("quality")["alcohol"]
            .agg(quantidade="count", media="mean", mediana="median")
            .round(2)
        )
        display(alcool_por_qualidade)

        ax = alcool_por_qualidade["media"].plot(
            kind="line",
            marker="o",
            figsize=(7, 4),
            color="#f58518",
            title="Alcool medio por qualidade",
        )
        ax.set_xlabel("Nota de qualidade")
        ax.set_ylabel("Alcool medio")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ### Pergunta 4: Vinhos de qualidade alta tem alcool medio maior que os demais?
        """
    ),
    code(
        """
        alcool_por_classe = (
            df.groupby("quality_class", observed=True)["alcohol"]
            .agg(quantidade="count", media="mean", mediana="median")
            .round(2)
        )
        display(alcool_por_classe)

        media_alta = df.loc[df["quality_class"] == "alta", "alcohol"].mean()
        media_demais = df.loc[df["quality_class"] != "alta", "alcohol"].mean()
        diferenca = media_alta - media_demais
        print(f"Alcool medio dos vinhos de qualidade alta: {media_alta:.2f}")
        print(f"Alcool medio dos demais vinhos: {media_demais:.2f}")
        print(f"Diferenca: {diferenca:.2f} ponto percentual de alcool.")
        """
    ),
    md(
        """
        ### Pergunta 5: Como a acidez volatil varia entre as classes de qualidade?
        """
    ),
    code(
        """
        acidez_volatil = (
            df.groupby("quality_class", observed=True)["volatile_acidity"]
            .agg(quantidade="count", media="mean", mediana="median")
            .round(3)
        )
        correlacao_acidez = df[["volatile_acidity", "quality"]].corr().iloc[0, 1]

        display(acidez_volatil)
        print(f"Correlacao entre acidez volatil e qualidade: {correlacao_acidez:.3f}")
        """
    ),
    md(
        """
        ### Pergunta 6: Quais variaveis numericas mais se relacionam com a qualidade?
        """
    ),
    code(
        """
        correlacoes = (
            df.select_dtypes(include="number")
            .corr()["quality"]
            .drop("quality")
            .sort_values(key=lambda serie: serie.abs(), ascending=False)
        )

        top_correlacoes = correlacoes.head(6).to_frame("correlacao_com_quality").round(3)
        display(top_correlacoes)

        ax = top_correlacoes["correlacao_com_quality"].sort_values().plot(
            kind="barh",
            figsize=(7, 4),
            color="#54a24b",
            title="Principais correlacoes com qualidade",
        )
        ax.set_xlabel("Correlacao")
        ax.set_ylabel("")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ### Pergunta 7: Qual e o perfil medio de vinhos baixos, medios e altos?
        """
    ),
    code(
        """
        perfil_classes = (
            df.groupby("quality_class", observed=True)[
                ["alcohol", "volatile_acidity", "sulphates", "citric_acid", "density", "chlorides"]
            ]
            .mean()
            .round(3)
        )
        display(perfil_classes)
        """
    ),
    md(
        """
        ### Pergunta 8: Vinhos com sulfatos acima da mediana tem qualidade media maior?
        """
    ),
    code(
        """
        sulfatos = (
            df.groupby("sulphates_group")["quality"]
            .agg(quantidade="count", qualidade_media="mean")
            .round(3)
        )
        taxa_alta_por_sulfatos = (
            df.assign(alta_qualidade=df["quality"] >= 7)
            .groupby("sulphates_group")["alta_qualidade"]
            .mean()
            .mul(100)
            .round(2)
            .to_frame("pct_qualidade_alta")
        )

        display(sulfatos)
        display(taxa_alta_por_sulfatos)
        """
    ),
    md(
        """
        ### Pergunta 9: A combinacao de alcool alto e acidez volatil baixa esta associada a notas melhores?
        """
    ),
    code(
        """
        limite_alcool_alto = df["alcohol"].quantile(0.75)
        limite_acidez_baixa = df["volatile_acidity"].quantile(0.25)

        df["segmento_alcool_acidez"] = "outros"
        df.loc[
            (df["alcohol"] >= limite_alcool_alto)
            & (df["volatile_acidity"] <= limite_acidez_baixa),
            "segmento_alcool_acidez",
        ] = "alcool alto e acidez volatil baixa"

        segmento = (
            df.groupby("segmento_alcool_acidez")["quality"]
            .agg(
                quantidade="count",
                qualidade_media="mean",
                pct_qualidade_alta=lambda serie: (serie >= 7).mean() * 100,
            )
            .round(2)
        )
        display(segmento)
        print(f"Limite de alcool alto (quartil 75%): {limite_alcool_alto:.2f}")
        print(f"Limite de acidez volatil baixa (quartil 25%): {limite_acidez_baixa:.2f}")
        """
    ),
    md(
        """
        ### Pergunta 10: Qual faixa de alcool apresenta melhor qualidade media?
        """
    ),
    code(
        """
        df["faixa_alcool"] = pd.cut(
            df["alcohol"],
            bins=[0, 9.5, 10.5, 11.5, 20],
            labels=["ate_9_5", "9_5_a_10_5", "10_5_a_11_5", "acima_11_5"],
        )

        faixas_alcool = (
            df.groupby("faixa_alcool", observed=True)
            .agg(
                quantidade=("quality", "size"),
                qualidade_media=("quality", "mean"),
                pct_qualidade_alta=("quality", lambda serie: (serie >= 7).mean() * 100),
            )
            .round(2)
        )
        display(faixas_alcool)
        """
    ),
    md(
        """
        ## 5. Conclusoes

        - O dataset bruto nao possui valores nulos, mas possui linhas duplicadas que foram removidas.
        - As notas se concentram principalmente nas classes intermediarias.
        - Alcool, sulfatos e acido citrico aparecem com associacao positiva com a qualidade.
        - Acidez volatil e densidade aparecem com associacao negativa com a qualidade.
        - O segmento com alcool alto e acidez volatil baixa tende a ter qualidade media maior e maior proporcao de vinhos com nota alta.
        """
    ),
]


nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "pygments_lexer": "ipython3",
    },
}

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, NOTEBOOK_PATH)

with NOTEBOOK_PATH.open("r", encoding="utf-8") as notebook_file:
    nb_to_execute = nbf.read(notebook_file, as_version=4)

client = NotebookClient(
    nb_to_execute,
    timeout=600,
    kernel_name="python3",
    resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}},
)
client.execute()

with NOTEBOOK_PATH.open("w", encoding="utf-8") as notebook_file:
    nbf.write(nb_to_execute, notebook_file)

print(f"Notebook gerado e executado: {NOTEBOOK_PATH}")
