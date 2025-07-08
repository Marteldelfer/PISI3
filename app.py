import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import warnings
import random

# Ignorar avisos de depreciação
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# --- Configuração da Página ---
st.set_page_config(page_title="Análise e ML de Filmes - TMDb", layout="wide")

# --- Título Principal ---
st.title("🎬 Análise de Dados e Machine Learning do TMDb")
st.markdown("Uma aplicação para explorar dados de filmes e interagir com modelos preditivos e de recomendação.")

# --- Funções de Carregamento ---
@st.cache_data
def load_base_data():
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
    artifacts = {}
    try:
        artifacts['regression_pipeline'] = joblib.load("full_model_pipeline.pkl")
        artifacts['df_rec'] = pd.read_csv('df_rec.csv')
        artifacts['cosine_sim'] = joblib.load('cosine_sim.pkl')
        return artifacts
    except FileNotFoundError:
        st.error("**Arquivos dos modelos não encontrados!** Por favor, execute os scripts de treinamento para gerá-los.", icon="🚨")
        return None

# --- Dicionários e Funções Auxiliares ---
LANGUAGE_CODES_TO_PORTUGUESE = {'en': 'Inglês', 'fr': 'Francês', 'ko': 'Coreano', 'ja': 'Japonês', 'zh': 'Chinês', 'es': 'Espanhol', 'de': 'Alemão', 'hi': 'Hindi', 'ru': 'Russo', 'it': 'Italiano', 'pt': 'Português', 'ar': 'Árabe', 'cn': 'Cantonês', 'sv': 'Sueco', 'da': 'Dinamarquês', 'no': 'Norueguês', 'fi': 'Finlandês', 'nl': 'Holandês', 'pl': 'Polonês', 'th': 'Tailandês', 'id': 'Indonésio', 'cs': 'Checo', 'hu': 'Húngaro', 'tr': 'Turco', 'el': 'Grego', 'fa': 'Persa', 'he': 'Hebraico', 'te': 'Telugo', 'ml': 'Malaiala', 'sr': 'Sérvio', 'bg': 'Búlgaro', 'uk': 'Ucraniano', 'ta': 'Tâmil', 'ab': 'Abcázio', 'az': 'Azerbaijano', 'bm': 'Bâmbara', 'bn': 'Bengali', 'bs': 'Bósnio', 'ca': 'Catalão', 'dv': 'Diveí', 'dz': 'Dzongkha', 'et': 'Estoniano', 'eu': 'Basco', 'ff': 'Fula', 'ga': 'Irlandês', 'gl': 'Galego', 'gu': 'Gujarati', 'hr': 'Croata', 'hy': 'Armênio', 'ig': 'Ibo', 'is': 'Islandês', 'iu': 'Inuktitut', 'km': 'Khmer', 'kn': 'Canarês', 'ku': 'Curdo', 'la': 'Latim', 'lt': 'Lituano', 'lv': 'Letão', 'mn': 'Mongol', 'mr': 'Marata', 'ms': 'Malaio', 'ne': 'Nepali', 'pa': 'Panjabi', 'ps': 'Pachto', 'ro': 'Romeno', 'si': 'Cingalês', 'sk': 'Eslovaco', 'sl': 'Esloveno', 'sw': 'Suaíli', 'tl': 'Tagalo', 'tn': 'Tswana', 'ur': 'Urdu', 'vi': 'Vietnamita', 'xx': 'Desconhecido'}
TRADUCOES_GENEROS = {"Action": "Ação", "Adventure": "Aventura", "Animation": "Animação", "Comedy": "Comédia", "Crime": "Crime", "Documentary": "Documentário", "Drama": "Drama", "Family": "Família", "Fantasy": "Fantasia", "History": "História", "Horror": "Terror", "Music": "Música", "Mystery": "Mistério", "Romance": "Romance", "Science Fiction": "Ficção Científica", "TV Movie": "Filme de TV", "Thriller": "Suspense", "War": "Guerra", "Western": "Faroeste"}
REVERSE_TRADUCOES_GENEROS = {v: k for k, v in TRADUCOES_GENEROS.items()}
def traduzir_generos(lista_generos):
    return [TRADUCOES_GENEROS.get(genero, genero) for genero in lista_generos]
def prepare_data_for_boxplot(df, top_n=10):
    df_exploded = df.dropna(subset=['genres']).copy()
    df_exploded['genres'] = df_exploded['genres'].str.split(', ')
    df_exploded = df_exploded.explode('genres')
    top_genres = df_exploded['genres'].value_counts().nlargest(top_n).index
    df_filtered = df_exploded[df_exploded['genres'].isin(top_genres)]
    df_filtered['genres_translated'] = df_filtered['genres'].map(TRADUCOES_GENEROS)
    return df_filtered

# --- Funções de Renderização dos Gráficos ---
@st.fragment
def render_main_plots(df_final_filtered):
    st.header("📄 Análise Exploratória dos Dados")
    st.write(f"**Resultados para a seleção:** `{df_final_filtered.shape[0]}` filmes encontrados.")
    if df_final_filtered.empty:
        st.warning("Nenhum dado encontrado para os filtros da barra lateral.")
        return
    with st.expander("🔍 Visualizar Amostra dos Dados Filtrados"):
        st.dataframe(df_final_filtered.head(10))
    st.subheader("💰 Receita vs. Orçamento")
    df_budget_revenue = df_final_filtered[df_final_filtered['budget'] > 1000]
    fig = go.Figure(go.Histogram2dContour(x=df_final_filtered['budget'], y=df_final_filtered['revenue'], contours=dict(coloring='fill'), colorscale='Blues', reversescale=True, ncontours=20, hoverinfo='x+y+z'))
    fig.update_layout(title='', xaxis=dict(title=dict(text="Orçamento (USD)", font=dict(size=14)), tickfont=dict(size=12), range=[df_budget_revenue['budget'].min(), df_budget_revenue['budget'].quantile(0.75)]), yaxis=dict(title=dict(text="Receita (USD)", font=dict(size=14)), tickfont=dict(size=12), range=[df_budget_revenue['revenue'].min(), df_budget_revenue['revenue'].quantile(0.85)]))
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("🌍 Nota Média por Idioma (Top 10)")
    lang_counts = df_final_filtered['original_language'].value_counts()
    frequent_langs = lang_counts[lang_counts > 20].index
    filtered_df_lang = df_final_filtered[df_final_filtered['original_language'].isin(frequent_langs)]
    if not filtered_df_lang.empty:
        language_ratings = filtered_df_lang.groupby('original_language')['vote_average'].mean().sort_values(ascending=False).head(10)
        languages_pt = [LANGUAGE_CODES_TO_PORTUGUESE.get(lang, lang) for lang in language_ratings.index]
        fig2 = px.bar(x=language_ratings.values, y=languages_pt, orientation='h', color=language_ratings.values, color_continuous_scale='Viridis_r', labels={'x': 'Nota Média', 'y': 'Idioma'})
        st.plotly_chart(fig2, use_container_width=True)
    st.subheader("🎭 Top 10 Gêneros por Número de Filmes")
    genre_counts = Counter([g.strip() for genre_str in df_final_filtered['genres'].dropna() for g in genre_str.split(',')])
    top_genres = genre_counts.most_common(10)
    if top_genres:
        genres_names, genres_vals = zip(*top_genres)
        genres_names_traduzidos = traduzir_generos(list(genres_names))
        fig3 = px.bar(x=genres_vals, y=genres_names_traduzidos, orientation='h', color=genres_vals, color_continuous_scale='Blues_r', labels={'x': 'Número de Filmes', 'y': 'Gênero'})
        fig3.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig3, use_container_width=True)
    st.subheader("💎 Top 10 'Joias Escondidas'")
    mediana_pop = df_final_filtered['popularity'].median()
    undervalued = df_final_filtered[(df_final_filtered['popularity'] < mediana_pop) & (df_final_filtered['vote_average'] >= 7.5) & (df_final_filtered['vote_count'] >= 100)]
    top_pearl = undervalued.sort_values(['vote_average', 'vote_count'], ascending=[False, False]).head(10)
    if not top_pearl.empty:
        fig6 = px.bar(top_pearl, x='vote_average', y='title', orientation='h', color='vote_average', color_continuous_scale='Teal', labels={'vote_average': 'Média de Votos', 'title': 'Título do Filme'})
        fig6.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig6, use_container_width=True)
    st.subheader("⏱️ Distribuição da Duração dos Filmes (Runtime)")
    fig_runtime = px.histogram(df_final_filtered, x='runtime', nbins=50, color_discrete_sequence=['purple'], labels={'runtime': 'Duração (minutos)'})
    st.plotly_chart(fig_runtime, use_container_width=True)
    st.subheader("⭐ Popularidade vs. Nota Média")
    fig_pop = go.Figure(data=go.Histogram2dContour(x=df_final_filtered['popularity'], y=df_final_filtered['vote_average'], colorscale='OrRd', reversescale=False, contours=dict(coloring='fill', showlines=True), ncontours=20))
    fig_pop.update_layout(xaxis=dict(title='Popularidade', range=[0, df_final_filtered['popularity'].quantile(0.95)]), yaxis=dict(title='Nota Média'))
    st.plotly_chart(fig_pop, use_container_width=True)

def plot_profit_histogram(df, limit):
    st.subheader("📈 Lucros")
    lucros = df[df['profit_percentage'] > 0]
    lucros_filtrados = lucros[lucros['profit_percentage'] < limit]
    fig = px.histogram(lucros_filtrados, x='profit_percentage', nbins=50, color_discrete_sequence=['green'], labels={'profit_percentage': 'Porcentagem de Lucro'})
    st.plotly_chart(fig, use_container_width=True)

def plot_loss_histogram(df):
    st.subheader("📉 Prejuízos")
    prejuizos = df[df['profit_percentage'] < 0]
    fig = px.histogram(prejuizos, x='profit_percentage', nbins=50, color_discrete_sequence=['red'], labels={'profit_percentage': 'Porcentagem de Prejuízo'})
    st.plotly_chart(fig, use_container_width=True)

def plot_profit_boxplot(df):
    st.subheader("📊 Distribuição de Lucro (%) por Gênero")

    # Filtrar dados com lucro positivo realista
    df_lucro = df[df['profit_percentage'].between(0.01, 5000)].copy()

    if df_lucro.empty:
        st.info("⚠️ Não há dados de lucro para exibir.")
        return

    # Preparar dados por gênero traduzido
    df_box = prepare_data_for_boxplot(df_lucro)

    # Remover outliers com base no IQR (por gênero)
    def remover_outliers_por_genero(df):
        resultado = []
        for genero in df['genres_translated'].unique():
            subset = df[df['genres_translated'] == genero]
            q1 = subset['profit_percentage'].quantile(0.25)
            q3 = subset['profit_percentage'].quantile(0.75)
            iqr = q3 - q1
            filtro = subset[(subset['profit_percentage'] >= q1 - 1.5 * iqr) &
                            (subset['profit_percentage'] <= q3 + 1.5 * iqr)]
            resultado.append(filtro)
        return pd.concat(resultado)

    df_sem_outliers = remover_outliers_por_genero(df_box)

    # Ordenar os gêneros pela mediana do lucro
    order = (df_sem_outliers.groupby('genres_translated')['profit_percentage']
             .median()
             .sort_values(ascending=False)
             .index)

    # Plot com Plotly
    fig = px.box(
        df_sem_outliers,
        x='profit_percentage',
        y='genres_translated',
        category_orders={'genres_translated': list(order)},
        color_discrete_sequence=['mediumseagreen'],
        labels={
            'profit_percentage': 'Lucro (%)',
            'genres_translated': 'Gênero'
        },
    )


    st.plotly_chart(fig, use_container_width=True)

def plot_loss_boxplot(df):
    st.subheader("📊 Prejuízo por Gênero")
    df_prejuizo = df[df['profit_percentage'] < 0]
    if not df_prejuizo.empty:
        df_prejuizo_box = prepare_data_for_boxplot(df_prejuizo)
        order = (df_prejuizo_box.groupby('genres_translated')['profit_percentage'].median().sort_values(ascending=True).index)
        fig = px.box(df_prejuizo_box, x='profit_percentage', y='genres_translated', category_orders={'genres_translated': list(order)}, color_discrete_sequence=['lightcoral'], labels={'profit_percentage': 'Prejuízo (%)', 'genres_translated': 'Gênero'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Não há dados de prejuízo para exibir.")

# ==============================================================================
# === FUNÇÃO DE CORRELAÇÃO ===
# ==============================================================================
@st.fragment
def render_corr(df_final_filtered):
    st.divider()
    st.header("Análise de Correlações")
    numeric_cols = df_final_filtered.select_dtypes(include=np.number).columns.tolist()
    traducao_colunas = {'popularity': 'Popularidade', 'budget': 'Orçamento', 'revenue': 'Receita', 'runtime': 'Duração', 'vote_average': 'Nota Média', 'vote_count': 'Qtd. de Votos', 'profit_percentage': '% de Lucro', 'release_year': 'Ano de Lançamento'}
    cols_to_corr = [col for col in numeric_cols if col in traducao_colunas]
    correlation_matrix = df_final_filtered[cols_to_corr].corr()
    correlation_matrix.rename(columns=traducao_colunas, index=traducao_colunas, inplace=True)
    
    # Lógica original para criar a matriz triangular inferior
    mask = np.tril(np.ones(correlation_matrix.shape), k=-1).astype(bool)
    correlation_matrix = correlation_matrix.mask(~mask)
    filtered_corr = correlation_matrix.dropna(axis=0, how='all').dropna(axis=1, how='all')
    
    # Gráfico original com px.imshow
    fig_corr = px.imshow(filtered_corr, text_auto=".2f", color_continuous_scale="RdYlGn", zmin=-1, zmax=1, labels=dict(color="Correlação"))
    fig_corr.update_layout(title='', width=1000, height=800) # Layout mais próximo do original
    st.plotly_chart(fig_corr, use_container_width=True)

def prepare_opcoes_para_campos_de_ml(df):
    try:
        generos_unicos = sorted(set(
            g.strip()
            for gen_str in df['genres'].dropna()
            for g in gen_str.split(',')
        ))

        produtoras_unicas = sorted(set(
            g.strip()
            for gen_str in df['production_companies'].dropna()
            for g in gen_str.split(',')
        ))

        diretores_unicos = sorted(set(
            g.strip()
            for gen_str in df['director'].dropna()
            for g in gen_str.split(',')
        ))

        atores_unicos = sorted(set(
            ator.strip()
            for cast_str in df['cast'].dropna()
            for ator in cast_str.split(',')
        ))
        return generos_unicos, produtoras_unicas, diretores_unicos, atores_unicos
    except:
        return None



# Carregamento dos dados
df = load_base_data()
ml_artifacts = load_ml_artifacts()

# Carregando as opções dos campos para ML
generos_unicos, produtoras_unicas, diretores_unicos, atores_unicos = prepare_opcoes_para_campos_de_ml(df)

# --- NAVEGAÇÃO PRINCIPAL ---
st.sidebar.title("Navegação")
page = st.sidebar.radio(
    "Escolha a seção que deseja visualizar:",
    ["📊 Análise Exploratória", "🤖 Modelos de Machine Learning"],
    label_visibility="collapsed"
)
st.sidebar.divider()

# ==============================================================================
# === PÁGINA 1: ANÁLISE EXPLORATÓRIA ===========================================
# ==============================================================================
if page == "📊 Análise Exploratória":

    st.sidebar.header("⚙️ Filtros de Análise")
    if df is not None:
        df_for_filters = df.dropna(subset=['release_year']).copy()
        df_for_filters['release_year'] = df_for_filters['release_year'].astype(int)
        min_year, max_year = int(df_for_filters['release_year'].min()), int(df_for_filters['release_year'].max())
        year_range = st.sidebar.slider("📅 Intervalo de Anos", min_year, max_year, (min_year, max_year))
        
        all_genres = sorted(list(set([g.strip() for s in df['genres'].dropna() for g in s.split(',')])))
        all_genres_translated = traduzir_generos(all_genres)
        selected_genres_translated = st.sidebar.multiselect("🎭 Gêneros", all_genres_translated, default=all_genres_translated)
        selected_genres_english = [REVERSE_TRADUCOES_GENEROS.get(g, g) for g in selected_genres_translated]
        
        min_budget, max_budget = float(df['budget'].min()), float(df['budget'].max())
        budget_range = st.sidebar.slider("💸 Orçamento (USD)", min_budget, max_budget, (min_budget, max_budget), format="$%.0f")
        
        min_revenue, max_revenue = float(df['revenue'].min()), float(df['revenue'].max())
        revenue_range = st.sidebar.slider("💰 Receita (USD)", min_revenue, max_revenue, (min_revenue, max_revenue), format="$%.0f")
        
        all_langs = sorted(df['original_language'].dropna().unique().tolist())
        lang_options = [f"{LANGUAGE_CODES_TO_PORTUGUESE.get(c, 'Desconhecido')} ({c})" for c in all_langs]
        selected_langs_display = st.sidebar.multiselect("🗣️ Idioma Original", lang_options, default=lang_options)
        selected_languages_codes = [opt[opt.rfind('(') + 1:opt.rfind(')')] for opt in selected_langs_display]
        
        df_filtered = df[
            (df['release_year'].between(year_range[0], year_range[1])) &
            (df['budget'].between(budget_range[0], budget_range[1])) &
            (df['revenue'].between(revenue_range[0], revenue_range[1]))
        ]
        if selected_genres_english:
            df_filtered = df_filtered[df_filtered['genres'].apply(lambda x: any(genre in str(x) for genre in selected_genres_english))]
        if selected_languages_codes:
            df_filtered = df_filtered[df_filtered['original_language'].isin(selected_languages_codes)]
        
        df_final_filtered = df_filtered.copy()

        render_main_plots(df_final_filtered)
        
        st.divider()
        st.header("Análise de Lucro e Prejuízo")
        
        profit_limit = st.slider("Limitar exibição da porcentagem de lucro (%)", 10, 5000, 500, 50, help="Este filtro afeta apenas o gráfico de histograma de lucros.")
        
        col1, col2 = st.columns(2)
        with col1:
            plot_profit_histogram(df_final_filtered, profit_limit)
        with col2:
            plot_loss_histogram(df_final_filtered)
            
        col3, col4 = st.columns(2)
        with col3:
            plot_profit_boxplot(df_final_filtered)
        with col4:
            plot_loss_boxplot(df_final_filtered)
        
        render_corr(df_final_filtered)
    else:
        st.error("Não foi possível carregar os dados para a análise.")

# ==============================================================================
# === PÁGINA 2: MACHINE LEARNING ==============================================
# ==============================================================================
elif page == "🤖 Modelos de Machine Learning":

    st.sidebar.header("🤖 Filtros do Recomendador")
    num_recommendations = st.sidebar.slider(
        "Número de Recomendações",
        min_value=3, max_value=20, value=5, step=1,
        help="Selecione quantos filmes você deseja que o sistema recomende."
    )
    
    st.header("🤖 Modelos de Machine Learning")
    if ml_artifacts is None:
        pass
    else:
        st.subheader("🍿 Sistema de Recomendação de Filmes")
        st.markdown("Selecione um filme e veja recomendações aleatórias baseadas no conteúdo.")
        df_rec = ml_artifacts['df_rec']
        cosine_sim = ml_artifacts['cosine_sim']
        indices = pd.Series(df_rec.index, index=df_rec['title']).drop_duplicates()
        def get_recommendations(title, num_recs, cosine_sim=cosine_sim):
            idx = indices[title]
            sim_scores_pool = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:51]
            num_to_sample = min(num_recs, len(sim_scores_pool))
            random_sim_scores = random.sample(sim_scores_pool, num_to_sample)
            movie_indices = [i[0] for i in random_sim_scores]
            return df_rec['title'].iloc[movie_indices]
        movie_list = df_rec['title'].unique()
        selected_movie = st.selectbox("Escolha um filme:", movie_list)
        if st.button("Recomendar Filmes Similares"):
            with st.spinner("Buscando recomendações..."):
                recommendations = get_recommendations(selected_movie, num_recommendations)
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
                # popularity = st.number_input("Popularidade (TMDb)", min_value=0.0, value=50.0, step=0.5)
                runtime = st.number_input("Duração (minutos)", min_value=60, value=120, step=5)
                genres = st.multiselect(label="Gêneros (separados por vírgula)", options=generos_unicos ,placeholder="Action, Adventure, Science Fiction")
            with col_form2:
                production_companies = st.multiselect(label="Produtora(s)", options= produtoras_unicas,placeholder= "Warner Bros. Pictures, Legendary Pictures")
                cast = st.multiselect(label="Elenco Principal", options= atores_unicos,placeholder= "Leonardo DiCaprio, Joseph Gordon-Levitt, Elliot Page")
                director = st.multiselect(label="Diretor(es)", options= diretores_unicos,placeholder= "Christopher Nolan")
            submitted = st.form_submit_button("Prever Receita")
            if submitted:

                genres_formatted = ", ".join(genres)
                production_companies_formatted = ", ".join(production_companies)
                cast_formatted = ", ".join(cast)
                director_formatted = ", ".join(director)

                input_data = pd.DataFrame({'budget': [budget], 'popularity': [popularity], 'runtime': [runtime], 'genres': [genres_formatted], 'production_companies': [production_companies_formatted], 'cast': [cast_formatted], 'director': [director_formatted]})
                with st.spinner("Processando..."):
                    pipeline = ml_artifacts['regression_pipeline']
                    prediction = pipeline.predict(input_data)
                    predicted_revenue = prediction[0]
                st.success("Previsão Concluída!")
                st.metric(label="Receita Estimada (USD)", value=f"$ {predicted_revenue:,.2f}")
