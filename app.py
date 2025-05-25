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