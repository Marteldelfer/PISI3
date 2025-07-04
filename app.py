import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings

# Ignorar avisos de depreciação do matplotlib
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# --- Configuração da Página ---
st.set_page_config(page_title="Análise e ML de Filmes - TMDb", layout="wide")

# --- Título Principal ---
st.title("🎬 Análise de Dados e Machine Learning do TMDb")
st.markdown("Uma aplicação para explorar dados de filmes e interagir com modelos preditivos e de recomendação.")

# --- Funções de Carregamento ---
@st.cache_data
def load_base_data():
    """Carrega o dataframe base para a análise exploratória."""
    try:
        df = pd.read_csv("tmdb_new.csv")
        if 'budget' in df.columns and 'revenue' in df.columns:
            df['profit_percentage'] = df.apply(
                lambda row: ((row['revenue'] - row['budget']) / row['budget']) * 100 if row['budget'] > 0 else 0,
                axis=1
            )
        if 'release_date' in df.columns:
            df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
        return df
    except FileNotFoundError:
        st.error("ERRO: Arquivo 'tmdb_new.csv' não encontrado. O app não pode continuar sem ele.")
        return None

@st.cache_resource
def load_ml_artifacts():
    """Carrega o pipeline de regressão e os arquivos do sistema de recomendação."""
    artifacts = {}
    try:
        artifacts['regression_pipeline'] = joblib.load("full_model_pipeline.pkl")
        artifacts['df_rec'] = pd.read_csv('df_rec.csv')
        artifacts['cosine_sim'] = joblib.load('cosine_sim.pkl')
        print("Artefatos de ML carregados com sucesso.")
        return artifacts
    except FileNotFoundError:
        return None

# --- Dicionários de Tradução (centralizados para estabilidade) ---
LANGUAGE_CODES_TO_PORTUGUESE = {
    'en': 'Inglês', 'fr': 'Francês', 'ko': 'Coreano', 'ja': 'Japonês', 'zh': 'Chinês',
    'es': 'Espanhol', 'de': 'Alemão', 'hi': 'Hindi', 'ru': 'Russo', 'it': 'Italiano',
    'pt': 'Português', 'ar': 'Árabe', 'cn': 'Cantonês', 'sv': 'Sueco', 'da': 'Dinamarquês',
    'no': 'Norueguês', 'fi': 'Finlandês', 'nl': 'Holandês', 'pl': 'Polonês', 'th': 'Tailandês',
    'id': 'Indonésio', 'cs': 'Checo', 'hu': 'Húngaro', 'tr': 'Turco', 'el': 'Grego',
    'fa': 'Persa', 'he': 'Hebraico', 'te': 'Telugo', 'ml': 'Malaiala', 'sr': 'Sérvio',
    'bg': 'Búlgaro', 'uk': 'Ucraniano', 'ta': 'Tâmil', 'ab': 'Abcázio', 'az': 'Azerbaijano',
    'bm': 'Bâmbara', 'bn': 'Bengali', 'bs': 'Bósnio', 'ca': 'Catalão', 'dv': 'Diveí',
    'dz': 'Dzongkha', 'et': 'Estoniano', 'eu': 'Basco', 'ff': 'Fula', 'ga': 'Irlandês',
    'gl': 'Galego', 'gu': 'Gujarati', 'hr': 'Croata', 'hy': 'Armênio', 'ig': 'Ibo',
    'is': 'Islandês', 'iu': 'Inuktitut', 'km': 'Khmer', 'kn': 'Canarês', 'ku': 'Curdo',
    'la': 'Latim', 'lt': 'Lituano', 'lv': 'Letão', 'mn': 'Mongol', 'mr': 'Marata',
    'ms': 'Malaio', 'ne': 'Nepali', 'pa': 'Panjabi', 'ps': 'Pachto', 'ro': 'Romeno',
    'si': 'Cingalês', 'sk': 'Eslovaco', 'sl': 'Esloveno', 'sw': 'Suaíli', 'tl': 'Tagalo',
    'tn': 'Tswana', 'ur': 'Urdu', 'vi': 'Vietnamita', 'xx': 'Desconhecido',
}

TRADUCOES_GENEROS = {
    "Action": "Ação", "Adventure": "Aventura", "Animation": "Animação", "Comedy": "Comédia",
    "Crime": "Crime", "Documentary": "Documentário", "Drama": "Drama", "Family": "Família",
    "Fantasy": "Fantasia", "History": "História", "Horror": "Terror", "Music": "Música",
    "Mystery": "Mistério", "Romance": "Romance", "Science Fiction": "Ficção Científica",
    "TV Movie": "Filme de TV", "Thriller": "Suspense", "War": "Guerra", "Western": "Faroeste"
}
REVERSE_TRADUCOES_GENEROS = {v: k for k, v in TRADUCOES_GENEROS.items()}


# --- Funções Auxiliares de Gráficos ---
def traduzir_generos(lista_generos):
    return [TRADUCOES_GENEROS.get(genero, genero) for genero in lista_generos]

def agrupar_outros(df_agrupado, top_n=8):
    if len(df_agrupado) <= top_n:
        return df_agrupado
    top = df_agrupado.sort_values('count', ascending=False).head(top_n)
    outros_df = df_agrupado.drop(top.index)
    if not outros_df.empty:
        if 'mean_profit' in outros_df.columns:
            mean_ponderada = np.average(outros_df['mean_profit'], weights=outros_df['count'])
            count_soma = outros_df['count'].sum()
            outros_row = pd.DataFrame([{'mean_profit': mean_ponderada, 'count': count_soma}], index=['Outros'])
            top = pd.concat([top, outros_row])
        elif 'med_profit' in outros_df.columns:
            mediana_outros = outros_df['med_profit'].median()
            count_soma = outros_df['count'].sum()
            outros_row = pd.DataFrame([{'med_profit': mediana_outros, 'count': count_soma}], index=['Outros'])
            top = pd.concat([top, outros_row])
    return top

def processar_por_genero(df_filtrado, lucro=True):
    df_filtrado = df_filtrado.copy()
    df_filtrado = df_filtrado[df_filtrado['genres'].notna()]
    df_filtrado['genres_list'] = df_filtrado['genres'].apply(lambda x: [g.strip() for g in x.split(',')])
    df_exploded = df_filtrado.explode('genres_list')
    if lucro:
        df_exploded = df_exploded[df_exploded['profit_percentage'] <= 50000]
        df_agrupado = df_exploded.groupby('genres_list')['profit_percentage'].agg(['median', 'count']).rename(columns={'median': 'med_profit'})
        df_agrupado['med_profit'] = df_agrupado['med_profit'].clip(upper=500)
    else:
        df_agrupado = df_exploded.groupby('genres_list')['profit_percentage'].agg(['mean', 'count']).rename(columns={'mean': 'mean_profit'})
        df_agrupado['mean_profit'] = df_agrupado['mean_profit'].clip(lower=-100)
    df_agrupado_final = agrupar_outros(df_agrupado, top_n=8)
    df_agrupado_final.index = traduzir_generos(df_agrupado_final.index.to_list())
    return df_agrupado_final

# Carregamento dos dados
df = load_base_data()
ml_artifacts = load_ml_artifacts()

# --- Abas para Organização ---
tab1, tab2 = st.tabs(["📊 Análise Exploratória", "🤖 Modelos de Machine Learning"])

# ==============================================================================
# === LÓGICA DA ABA 1: Múltiplos Fragmentos para UI e Performance =============
# ==============================================================================

@st.fragment
def render_main_plots(df_final_filtered):
    """
    Este fragmento renderiza TODOS os gráficos, EXCETO o de distribuição de lucros.
    Ele é atualizado apenas quando os filtros da SIDEBAR mudam.
    """
    st.header("📄 Análise Exploratória dos Dados")
    st.write(f"**Resultados para a seleção:** `{df_final_filtered.shape[0]}` filmes encontrados.")

    if df_final_filtered.empty:
        st.warning("Nenhum dado encontrado para os filtros da barra lateral.")
        return

    with st.expander("🔍 Visualizar Amostra dos Dados Filtrados"):
        st.dataframe(df_final_filtered.head(10))

    FIG_SIZE = (6, 4)
    FIG_DPI = 75

    st.subheader("💰 Receita vs. Orçamento")
    fig1, ax1 = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI)
    sns.scatterplot(data=df_final_filtered[df_final_filtered['budget'] > 1000], x='budget', y='revenue', alpha=0.4, ax=ax1, color='royalblue')
    ax1.set_xlabel('Orçamento (USD)')
    ax1.set_ylabel('Receita (USD)')
    st.pyplot(fig1, use_container_width=False)

    st.subheader("🌍 Nota Média por Idioma (Top 10)")
    lang_counts = df_final_filtered['original_language'].value_counts()
    frequent_langs = lang_counts[lang_counts > 20].index
    filtered_df_lang = df_final_filtered[df_final_filtered['original_language'].isin(frequent_langs)]
    fig2, ax2 = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI)
    if not filtered_df_lang.empty:
        language_ratings = filtered_df_lang.groupby('original_language')['vote_average'].mean().sort_values(ascending=False).head(10)
        sns.barplot(x=language_ratings.values, y=[LANGUAGE_CODES_TO_PORTUGUESE.get(lang, lang) for lang in language_ratings.index], palette='viridis', ax=ax2)
        ax2.set_xlabel("Nota Média")
        ax2.set_ylabel("Idioma")
    else:
        ax2.text(0.5, 0.5, "Dados insuficientes.", ha='center', va='center', transform=ax2.transAxes)
        ax2.set_xticks([]); ax2.set_yticks([])
    st.pyplot(fig2, use_container_width=False)
    
    st.subheader("🎭 Top 10 Gêneros por Número de Filmes")
    genre_counts = Counter([g.strip() for genre_str in df_final_filtered['genres'].dropna() for g in genre_str.split(',')])
    top_genres = genre_counts.most_common(10)
    if top_genres:
        genres_names, genres_vals = zip(*top_genres)
        fig3, ax3 = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI)
        sns.barplot(x=list(genres_vals), y=traduzir_generos(list(genres_names)), palette='mako', ax=ax3)
        ax3.set_xlabel("Número de Filmes")
        ax3.set_ylabel("Gênero")
        st.pyplot(fig3, use_container_width=False)

    st.subheader("💎 Top 10 'Joias Escondidas'")
    mediana_pop = df_final_filtered['popularity'].median()
    undervalued = df_final_filtered[(df_final_filtered['popularity'] < mediana_pop) & (df_final_filtered['vote_average'] >= 7.5) & (df_final_filtered['vote_count'] >= 100)]
    top_pearl = undervalued.sort_values(['vote_average', 'vote_count'], ascending=[False, False]).head(10)
    fig6, ax6 = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI)
    if not top_pearl.empty:
        sns.barplot(data=top_pearl, x='vote_average', y='title', palette='magma', ax=ax6)
        ax6.set_xlabel("Média de Votos")
        ax6.set_ylabel("Título do Filme")
    else:
        ax6.text(0.5, 0.5, "Nenhuma 'joia escondida' encontrada.", ha='center', va='center', transform=ax6.transAxes)
        ax6.set_xticks([]); ax6.set_yticks([])
    st.pyplot(fig6, use_container_width=False)

    st.subheader("⏱️ Distribuição da Duração dos Filmes (Runtime)")
    fig_runtime, ax_runtime = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI)
    sns.histplot(df_final_filtered['runtime'].dropna(), bins=50, kde=True, ax=ax_runtime, color='purple')
    ax_runtime.set_xlabel("Duração (minutos)")
    ax_runtime.set_ylabel("Frequência")
    st.pyplot(fig_runtime, use_container_width=False)

    st.subheader("⭐ Popularidade vs. Nota Média")
    fig_pop, ax_pop = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI)
    sns.scatterplot(data=df_final_filtered, x='popularity', y='vote_average', alpha=0.3, ax=ax_pop, color='gold')
    ax_pop.set_xlabel("Popularidade")
    ax_pop.set_ylabel("Nota Média")
    ax_pop.set_xlim(0, df_final_filtered['popularity'].quantile(0.95))
    st.pyplot(fig_pop, use_container_width=False)

    st.divider()
    st.header("Análise de Lucro e Prejuízo")


@st.fragment
def render_profit_distribution_plot(df_final_filtered):
    """
    Este fragmento renderiza APENAS o gráfico de distribuição de lucros.
    Ele contém seu PRÓPRIO slider, e apenas este fragmento será reexecutado
    quando esse slider for alterado.
    """
    st.subheader("📈 Distribuição de Lucros Positivos")
    lucros = df_final_filtered[df_final_filtered['profit_percentage'] > 0]
    
    # O slider agora vive DENTRO do seu fragmento dedicado
    profit_limit = st.slider("Limitar exibição de lucro (%)", 10, 5000, 500, 50)
    
    lucros_filtrados = lucros[lucros['profit_percentage'] < profit_limit]
    
    fig_lucros, ax_lucros = plt.subplots(figsize=(6, 4), dpi=75)
    sns.histplot(lucros_filtrados['profit_percentage'], bins=50, kde=True, ax=ax_lucros, color='green')
    ax_lucros.set_xlabel("Porcentagem de Lucro")
    ax_lucros.set_ylabel("Quantidade")
    st.pyplot(fig_lucros, use_container_width=False)


@st.fragment
def render_remaining_plots(df_final_filtered):
    """
    Este fragmento renderiza os gráficos restantes que dependem apenas dos filtros da sidebar.
    """
    FIG_SIZE = (6, 4)
    FIG_DPI = 75

    st.subheader("💹 Mediana de Lucro (%) por Gênero")
    df_lucro = df_final_filtered[df_final_filtered['profit_percentage'] > 0]
    if not df_lucro.empty:
        df_lucro_agrupado = processar_por_genero(df_lucro, lucro=True)
        fig_lucro_gen, ax_lucro_gen = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI)
        sns.barplot(x=df_lucro_agrupado.index, y=df_lucro_agrupado['med_profit'], color='green', ax=ax_lucro_gen)
        ax_lucro_gen.set_xlabel("Gênero")
        ax_lucro_gen.set_ylabel("Mediana do Lucro (%)")
        ax_lucro_gen.tick_params(axis='x', rotation=45)
        st.pyplot(fig_lucro_gen, use_container_width=False)

    st.subheader("📉 Distribuição de Prejuízos")
    prejuizos = df_final_filtered[df_final_filtered['profit_percentage'] < 0]
    fig_prejuizo, ax_prejuizo = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI)
    sns.histplot(prejuizos['profit_percentage'], bins=50, kde=True, ax=ax_prejuizo, color='red')
    ax_prejuizo.set_xlabel("Porcentagem de Lucro")
    ax_prejuizo.set_ylabel("Quantidade")
    st.pyplot(fig_prejuizo, use_container_width=False)

    st.subheader("📉 Média de Prejuízo (%) por Gênero")
    df_prejuizo = df_final_filtered[df_final_filtered['profit_percentage'] < 0]
    if not df_prejuizo.empty:
        df_prejuizo_agrupado = processar_por_genero(df_prejuizo, lucro=False)
        fig_prejuizo_gen, ax_prejuizo_gen = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI)
        sns.barplot(x=df_prejuizo_agrupado.index, y=-df_prejuizo_agrupado['mean_profit'], color='red', ax=ax_prejuizo_gen)
        ax_prejuizo_gen.set_xlabel("Gênero")
        ax_prejuizo_gen.set_ylabel("Média de Prejuízo (%)")
        ax_prejuizo_gen.tick_params(axis='x', rotation=45)
        st.pyplot(fig_prejuizo_gen, use_container_width=False)

    st.divider()
    st.header("Análise de Correlações")
    
    numeric_cols = df_final_filtered.select_dtypes(include=np.number).columns.tolist()
    traducao_colunas = {
        'popularity': 'Popularidade', 'budget': 'Orçamento',
        'revenue': 'Receita', 'runtime': 'Duração', 'vote_average': 'Nota Média',
        'vote_count': 'Qtd. de Votos', 'profit_percentage': '% de Lucro', 'release_year': 'Ano de Lançamento'
    }
    cols_to_corr = [col for col in numeric_cols if col in traducao_colunas]
    correlation_matrix = df_final_filtered[cols_to_corr].corr()
    correlation_matrix.rename(columns=traducao_colunas, index=traducao_colunas, inplace=True)
    
    fig_corr, ax_corr = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI)
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax_corr, vmin=-1, vmax=1, annot_kws={"size": 6})
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    st.pyplot(fig_corr, use_container_width=False)


with tab1:
    if df is not None:
        # --- Controles Ficam na Sidebar ---
        st.sidebar.header("⚙️ Filtros de Análise")

        # Filtro de Ano
        df_for_filters = df.dropna(subset=['release_year']).copy()
        df_for_filters['release_year'] = df_for_filters['release_year'].astype(int)
        min_year, max_year = int(df_for_filters['release_year'].min()), int(df_for_filters['release_year'].max())
        year_range = st.sidebar.slider("📅 Intervalo de Anos", min_year, max_year, (min_year, max_year))

        # Filtro de Gênero
        all_genres = sorted(list(set([g.strip() for s in df['genres'].dropna() for g in s.split(',')])))
        all_genres_translated = traduzir_generos(all_genres)
        selected_genres_translated = st.sidebar.multiselect("🎭 Gêneros", all_genres_translated, default=all_genres_translated)
        selected_genres_english = [REVERSE_TRADUCOES_GENEROS.get(g, g) for g in selected_genres_translated]

        # Filtro de Orçamento
        min_budget, max_budget = float(df['budget'].min()), float(df['budget'].max())
        budget_range = st.sidebar.slider("💸 Orçamento (USD)", min_budget, max_budget, (min_budget, max_budget), format="$%.0f")

        # Filtro de Receita
        min_revenue, max_revenue = float(df['revenue'].min()), float(df['revenue'].max())
        revenue_range = st.sidebar.slider("💰 Receita (USD)", min_revenue, max_revenue, (min_revenue, max_revenue), format="$%.0f")

        # Filtro de Idioma
        all_langs = sorted(df['original_language'].dropna().unique().tolist())
        lang_options = [f"{LANGUAGE_CODES_TO_PORTUGUESE.get(c, 'Desconhecido')} ({c})" for c in all_langs]
        selected_langs_display = st.sidebar.multiselect("🗣️ Idioma Original", lang_options, default=lang_options)
        selected_languages_codes = [opt[opt.rfind('(') + 1:opt.rfind(')')] for opt in selected_langs_display]
        
        # --- Lógica de Filtragem Principal ---
        df_filtered = df[
            df['release_year'].between(year_range[0], year_range[1]) &
            df['budget'].between(budget_range[0], budget_range[1]) &
            df['revenue'].between(revenue_range[0], revenue_range[1])
        ]
        if selected_genres_english:
            df_filtered = df_filtered[df_filtered['genres'].apply(lambda x: any(genre in str(x) for genre in selected_genres_english))]
        if selected_languages_codes:
            df_filtered = df_filtered[df_filtered['original_language'].isin(selected_languages_codes)]

        df_final_filtered = df_filtered.copy()
        
        # --- Chamada dos Fragmentos ---
        render_main_plots(df_final_filtered)
        render_profit_distribution_plot(df_final_filtered)
        render_remaining_plots(df_final_filtered)

    else:
        st.error("Não foi possível carregar os dados para a análise.")


# ==============================================================================
# === ABA 2: MACHINE LEARNING (não foi alterada) ===============================
# ==============================================================================
with tab2:
    st.header("🤖 Modelos de Machine Learning")

    if ml_artifacts is None:
        st.error("**Arquivos dos modelos não encontrados!** Por favor, execute os scripts de treinamento para gerá-los.")
    else:
        st.subheader("🍿 Sistema de Recomendação de Filmes")
        st.markdown("Selecione um filme e veja 5 recomendações baseadas no conteúdo.")
        
        df_rec = ml_artifacts['df_rec']
        cosine_sim = ml_artifacts['cosine_sim']
        indices = pd.Series(df_rec.index, index=df_rec['title']).drop_duplicates()

        def get_recommendations(title, cosine_sim=cosine_sim):
            idx = indices[title]
            sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:6]
            movie_indices = [i[0] for i in sim_scores]
            return df_rec['title'].iloc[movie_indices]

        movie_list = df_rec['title'].unique()
        selected_movie = st.selectbox("Escolha um filme:", movie_list)

        if st.button("Recomendar Filmes Similares"):
            recommendations = get_recommendations(selected_movie)
            st.success("Aqui estão suas recomendações:")
            for i, movie in enumerate(recommendations):
                st.write(f"**{i+1}.** {movie}")

        st.divider()

        st.subheader("💸 Previsão de Receita de Bilheteria")
        st.markdown("Insira os dados de um filme hipotético para prever sua receita potencial.")

        with st.form("prediction_form"):
            col_form1, col_form2 = st.columns(2)
            with col_form1:
                budget = st.number_input("Orçamento (USD)", min_value=10000, value=50000000, step=1000000)
                popularity = st.number_input("Popularidade (TMDb)", min_value=0.0, value=50.0, step=0.5)
                runtime = st.number_input("Duração (minutos)", min_value=60, value=120, step=5)
                genres = st.text_input("Gêneros (separados por vírgula)", "Action, Adventure, Science Fiction")
            
            with col_form2:
                production_companies = st.text_input("Produtora(s)", "Warner Bros. Pictures, Legendary Pictures")
                cast = st.text_input("Elenco Principal", "Leonardo DiCaprio, Joseph Gordon-Levitt, Elliot Page")
                director = st.text_input("Diretor(es)", "Christopher Nolan")

            submitted = st.form_submit_button("Prever Receita")
            
            if submitted:
                input_data = pd.DataFrame({
                    'budget': [budget], 'popularity': [popularity], 'runtime': [runtime],
                    'genres': [genres], 'production_companies': [production_companies],
                    'cast': [cast], 'director': [director]
                })

                with st.spinner("Processando..."):
                    pipeline = ml_artifacts['regression_pipeline']
                    prediction = pipeline.predict(input_data)
                    predicted_revenue = prediction[0]

                st.success("Previsão Concluída!")
                st.metric(label="Receita Estimada (USD)", value=f"$ {predicted_revenue:,.2f}")
