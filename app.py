import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

st.set_page_config(page_title="Análise de Filmes - TMDb", layout="centered")
st.title("🎬 Análise de Dados do TMDb")

@st.cache_data
def load_data():
    df = pd.read_csv("tmdb_new.csv")
    return df

df = load_data()

st.header("📄 Pré-visualização dos Dados")

st.write(f"**Dimensões do conjunto de dados:** {df.shape[0]} linhas × {df.shape[1]} colunas")


num_rows = st.slider("Quantas linhas deseja visualizar?", min_value=5, max_value=100, value=10, step=5)
st.dataframe(df.head(num_rows))

with st.expander("🔍 Ver todos os dados (use com moderação)"):
    st.dataframe(df)


# --- Gráfico 1: Receita vs Orçamento ---
st.subheader("💰 Receita vs. Orçamento")
fig1, ax1 = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=df[df['budget'] > 0], x='budget', y='revenue', alpha=0.5, ax=ax1)
ax1.set_title('Receita vs. Orçamento')
ax1.set_xlabel('Orçamento (USD)')
ax1.set_ylabel('Receita (USD)')
st.pyplot(fig1)

# --- Gráfico 2: Nota média por idioma ---
st.subheader("🌍 Nota Média por Idioma (Top 10)")

valid_langs = df['original_language'].dropna()
valid_langs = valid_langs[valid_langs.str.isalpha()]

lang_counts = valid_langs.value_counts()
frequent_langs = lang_counts[lang_counts > 20].index

filtered_df = df[df['original_language'].isin(frequent_langs)]
language_ratings = (
    filtered_df.groupby('original_language')['vote_average']
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

fig2, ax2 = plt.subplots(figsize=(8, 5))
sns.barplot(x=language_ratings.values, y=language_ratings.index, palette='viridis', ax=ax2)
ax2.set_title("Nota Média por Idioma")
ax2.set_xlabel("Nota Média")
ax2.set_ylabel("Idioma")
st.pyplot(fig2)

# --- Gráfico 3: Top 10 Gêneros ---
st.subheader("🎭 Top 10 Gêneros por Número de Filmes")
genre_counts = Counter()
for genre_str in df['genres'].dropna():
    genres_list = [g.strip() for g in genre_str.split(',')]
    genre_counts.update(genres_list)

top_genres = genre_counts.most_common(10)
genres_names, genres_vals = zip(*top_genres)

fig3, ax3 = plt.subplots(figsize=(8, 5))
sns.barplot(x=list(genres_vals), y=list(genres_names), palette='mako', ax=ax3)
ax3.set_title("Top 10 Gêneros")
ax3.set_xlabel("Número de Filmes")
ax3.set_ylabel("Gênero")
st.pyplot(fig3)

# --- Gráfico 4A: Distribuição de Lucros Positivos com Filtro ---
st.subheader("📈 Distribuição de Lucros Positivos")

lucros = df[df['profit_percentage'] > 0]

max_lucro = int(min(5000, lucros['profit_percentage'].max()))
limite = st.slider("Limitar exibição de lucro (%)", 10, max_lucro, value=500, step=50)

lucros_filtrados = lucros[lucros['profit_percentage'] < limite]

fig_lucros, ax_lucros = plt.subplots(figsize=(8, 5))
sns.histplot(lucros_filtrados['profit_percentage'], bins=50, kde=True, ax=ax_lucros, color='green')
ax_lucros.set_title(f"Distribuição do Lucro Positivo (até {limite}%)")
ax_lucros.set_xlabel("Lucro (%)")
ax_lucros.set_ylabel("Frequência")
st.pyplot(fig_lucros)

st.markdown("**🔍 Estatísticas dos Lucros Positivos (filtrados):**")
st.write(lucros_filtrados['profit_percentage'].describe())

# --- Gráfico 4B: Distribuição de Prejuízos ---
st.subheader("📉 Distribuição de Prejuízos (Lucro Negativo)")
prejuizos = df[df['profit_percentage'] < 0]
fig_prejuizo, ax_prejuizo = plt.subplots(figsize=(8, 5))
sns.histplot(prejuizos['profit_percentage'], bins=50, kde=True, ax=ax_prejuizo, color='red')
ax_prejuizo.set_title("Distribuição dos Prejuízos (%)")
ax_prejuizo.set_xlabel("Lucro (%)")
ax_prejuizo.set_ylabel("Frequência")
st.pyplot(fig_prejuizo)

# --- Gráfico 6 ---
st.subheader("💎 Top 10 Filmes Bem Avaliados e Pouco Populares")

mediana_pop = df['popularity'].median()
undervalued = df[
    (df['popularity'] < mediana_pop) &
    (df['vote_average'] >= 7.5) &
    (df['vote_count'] >= 50)
]

top_pearl = undervalued.sort_values(
    ['vote_average', 'vote_count'], ascending=[False, False]
).head(10)

fig6, ax6 = plt.subplots(figsize=(8, 5))
sns.barplot(
    data=top_pearl, 
    x='vote_average', 
    y='title', 
    palette='magma', 
    ax=ax6
)
ax6.set_title("Top 10 Filmes com Alta Avaliação e Baixa Popularidade")
ax6.set_xlabel("Média de Votos")
ax6.set_ylabel("Título do Filme")
st.pyplot(fig6)


# Novos gráficos de prejuízo e lucro divididos por gênero
genero_traducao = {
    "Action": "Ação",
    "Adventure": "Aventura",
    "Animation": "Animação",
    "Comedy": "Comédia",
    "Crime": "Crime",
    "Documentary": "Documentário",
    "Drama": "Drama",
    "Family": "Família",
    "Fantasy": "Fantasia",
    "History": "História",
    "Horror": "Terror",
    "Music": "Música",
    "Mystery": "Mistério",
    "Romance": "Romance",
    "Science Fiction": "Ficção Científica",
    "TV Movie": "Filme para TV",
    "Thriller": "Suspense",
    "War": "Guerra",
    "Western": "Faroeste"
}

def traduzir_generos(lista_generos):
    return [genero_traducao.get(g, g) for g in lista_generos]

def agrupar_outros(df_agrupado, top_n=8):
    # Pega os top_n gêneros por número de filmes
    top_generos = df_agrupado['count'].nlargest(top_n).index.tolist()
    outros = df_agrupado.index.difference(top_generos)

    # Soma os valores do grupo Outros
    outros_df = df_agrupado.loc[outros]
    if not outros_df.empty:
        # Para média, usa média ponderada pelo count
        mean_ponderada = np.average(outros_df['mean_profit'], weights=outros_df['count'])
        count_soma = outros_df['count'].sum()
        # Monta novo df com top + "Outros"
        df_top = df_agrupado.loc[top_generos].copy()
        df_top.loc['Outros'] = {'mean_profit': mean_ponderada, 'count': count_soma}
        return df_top
    else:
        return df_agrupado.loc[top_generos]

def traduzir_generos(lista_generos):
    traducoes = {
        "Action": "Ação", "Adventure": "Aventura", "Animation": "Animação",
        "Comedy": "Comédia", "Crime": "Crime", "Documentary": "Documentário",
        "Drama": "Drama", "Family": "Família", "Fantasy": "Fantasia",
        "History": "História", "Horror": "Terror", "Music": "Música",
        "Mystery": "Mistério", "Romance": "Romance", "Science Fiction": "Ficção Científica",
        "TV Movie": "Filme de TV", "Thriller": "Suspense", "War": "Guerra", "Western": "Faroeste"
    }
    return [traducoes.get(genero, genero) for genero in lista_generos]

def agrupar_outros(df_agrupado, top_n=8):
    top = df_agrupado.sort_values('count', ascending=False).head(top_n)
    outros = df_agrupado.drop(top.index)
    outros_mean = outros['med_profit'].median() if 'med_profit' in outros else outros['mean_profit'].median()
    outros_total = outros['count'].sum()
    df_top = top.copy()
    df_top.loc["Outros"] = [outros_mean, outros_total]
    return df_top

def processar_por_genero(df_filtrado, lucro=True):
    df_filtrado = df_filtrado.copy()
    df_filtrado = df_filtrado[df_filtrado['genres'].notna()]
    df_filtrado['genres_list'] = df_filtrado['genres'].apply(lambda x: [g.strip() for g in x.split(',')])
    df_exploded = df_filtrado.explode('genres_list')

    if lucro:
        # Remover lucros extremos acima de 5000% para não distorcer e evitar outliers
        df_exploded = df_exploded[df_exploded['profit_percentage'] <= 50000]
        df_agrupado = df_exploded.groupby('genres_list')['profit_percentage'].agg(['median', 'count']).rename(columns={'median':'med_profit'})
        df_agrupado['med_profit'] = df_agrupado['med_profit'].clip(upper=500)
    else:
        df_agrupado = df_exploded.groupby('genres_list')['profit_percentage'].agg(['mean', 'count']).rename(columns={'mean':'mean_profit'})
        df_agrupado['mean_profit'] = df_agrupado['mean_profit'].clip(lower=-100)

    df_agrupado = agrupar_outros(df_agrupado, top_n=8)

    # Traduz os gêneros
    df_agrupado.index = traduzir_generos(df_agrupado.index.to_list())
    return df_agrupado

# --- Gráfico de Lucro por Gênero ---
st.subheader("💹 Mediana de Lucro (%) por Gênero (máx 500%)")
df_lucro = df[df['profit_percentage'] > 0]
df_lucro_agrupado = processar_por_genero(df_lucro, lucro=True)

fig_lucro_gen, ax_lucro_gen = plt.subplots(figsize=(10, 6))  # ← aumento no tamanho
sns.barplot(x=df_lucro_agrupado.index, y=df_lucro_agrupado['med_profit'], color='green', ax=ax_lucro_gen)
ax_lucro_gen.set_title("Lucro Mediano (%) por Gênero (máximo 500%)")
ax_lucro_gen.set_xlabel("Gênero")
ax_lucro_gen.set_ylabel("Lucro Mediano (%)")
ax_lucro_gen.set_xticklabels(ax_lucro_gen.get_xticklabels(), rotation=45, ha='right')
ax_lucro_gen.set_ylim(0, 550)  # ← margem visual maior
st.pyplot(fig_lucro_gen)


# --- Gráfico de Prejuízo por Gênero ---
st.subheader("📉 Média de Prejuízo (%) por Gênero (mínimo -100%)")
df_prejuizo = df[df['profit_percentage'] < 0]
df_prejuizo_agrupado = processar_por_genero(df_prejuizo, lucro=False)

fig_prejuizo_gen, ax_prejuizo_gen = plt.subplots(figsize=(10, 6))
sns.barplot(x=df_prejuizo_agrupado.index, y=-df_prejuizo_agrupado['mean_profit'], color='red', ax=ax_prejuizo_gen)
ax_prejuizo_gen.set_title("Prejuízo Médio (%) por Gênero (mínimo -100%)")
ax_prejuizo_gen.set_xlabel("Gênero")
ax_prejuizo_gen.set_ylabel("Prejuízo Médio (%)")
ax_prejuizo_gen.set_xticklabels(ax_prejuizo_gen.get_xticklabels(), rotation=45, ha='right')
ax_prejuizo_gen.set_ylim(0, 110)
st.pyplot(fig_prejuizo_gen)



