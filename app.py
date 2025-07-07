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

def prepare_data_for_boxplot(df, top_n=10):
    """
    Prepara os dados para o boxplot de lucro/prejuízo por gênero.
    - 'Explode' os gêneros para ter uma linha por gênero por filme.
    - Identifica os N gêneros mais frequentes.
    - Filtra o DataFrame para incluir apenas esses top N gêneros.
    - Traduz os nomes dos gêneros para português.
    """
    # Garante que a coluna de gêneros não é nula e a 'explode'
    df_exploded = df.dropna(subset=['genres']).copy()
    df_exploded['genres'] = df_exploded['genres'].str.split(', ')
    df_exploded = df_exploded.explode('genres')

    # Encontra os top N gêneros mais comuns
    top_genres = df_exploded['genres'].value_counts().nlargest(top_n).index

    # Filtra o DataFrame para conter apenas os filmes desses gêneros
    df_filtered = df_exploded[df_exploded['genres'].isin(top_genres)]
    
    # Traduz os nomes dos gêneros para a plotagem
    df_filtered['genres_translated'] = df_filtered['genres'].map(TRADUCOES_GENEROS)
    
    return df_filtered

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

# --- Abas para Organização ---
tab1, tab2 = st.tabs(["📊 Análise Exploratória", "🤖 Modelos de Machine Learning"])

# ==============================================================================
# === LÓGICA DA ABA 1: Múltiplos Fragmentos para UI e Performance =============
# ==============================================================================

@st.fragment
def render_main_plots(df_final_filtered):
    """
    Este fragmento renderiza os gráficos principais que dependem apenas dos filtros da sidebar.
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
    df_budget_revenue = df_final_filtered[df_final_filtered['budget'] > 1000]

    df = df_final_filtered[(df_final_filtered['budget'] > -1e9) & (df_final_filtered['revenue'] > -1e9)]  # só filtro básico

    fig = go.Figure(go.Histogram2dContour(
        x=df['budget'],
        y=df['revenue'],
        contours=dict(coloring='fill'),
        colorscale='Blues',
        reversescale=True,
        ncontours=20,
        hoverinfo='x+y+z',
    ))

    fig.update_layout(
        title='',
        xaxis=dict(
            title=dict(text="Orçamento (USD)", font=dict(size=36)),
            tickfont=dict(size=32),
            range=[df_budget_revenue['budget'].min(), df_budget_revenue['budget'].quantile(0.75)],
        ),
        yaxis=dict(
            title=dict(text="Receita (USD)", font=dict(size=36)),
            tickfont=dict(size=32),
            range=[df_budget_revenue['revenue'].min(), df_budget_revenue['revenue'].quantile(0.85)],
        ),
        width=1500,
        height=1000,
        font=dict(size=36),
        title_font_size=40,
    )

    st.plotly_chart(fig, use_container_width=False)


    st.subheader("🌍 Nota Média por Idioma (Top 10)")
    lang_counts = df_final_filtered['original_language'].value_counts()
    frequent_langs = lang_counts[lang_counts > 20].index
    filtered_df_lang = df_final_filtered[df_final_filtered['original_language'].isin(frequent_langs)]
    fig2, ax2 = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI)
    if not filtered_df_lang.empty:
        language_ratings = filtered_df_lang.groupby('original_language')['vote_average'].mean().sort_values(ascending=False).head(10)
        languages_pt = [LANGUAGE_CODES_TO_PORTUGUESE.get(lang, lang) for lang in language_ratings.index]

        fig2 = px.bar(
            x=language_ratings.values,
            y=languages_pt,
            orientation='h',
            color=language_ratings.values,
            color_continuous_scale='Viridis_r',
            height=1000,
            width=1500,
            labels={'x': 'Nota Média', 'y': 'Idioma'}
        )

        fig2.update_layout(
            title='',
            yaxis=dict(autorange='reversed', tickfont=dict(size=32), title=dict(text='Idioma', font=dict(size=36))),
            xaxis=dict(tickfont=dict(size=32), title=dict(text='Nota Média', font=dict(size=36))),
            font=dict(size=36),
            title_font_size=40,
        )

        st.plotly_chart(fig2, use_container_width=False)
    else:
        st.write("Dados insuficientes.")
    
    st.subheader("🎭 Top 10 Gêneros por Número de Filmes")
    genre_counts = Counter([g.strip() for genre_str in df_final_filtered['genres'].dropna() for g in genre_str.split(',')])
    top_genres = genre_counts.most_common(10)
    if top_genres:
        genres_names, genres_vals = zip(*top_genres)
        genres_names_traduzidos = traduzir_generos(list(genres_names))

        fig3 = px.bar(
            top_genres,
            x=genres_vals,
            y=genres_names_traduzidos,
            orientation='h',
            color=genres_vals,
            color_continuous_scale='Blues',
        )

        fig3.update_layout(
            title='',
            yaxis=dict(autorange="reversed", tickfont=dict(size=32), title=dict(text='Gênero', font=dict(size=36))),
            xaxis=dict(tickfont=dict(size=32), title=dict(text='Número de Filmes', font=dict(size=36))),
            font=dict(size=36),
            title_font_size=40,
            height=1000,
            width=1500,
        )

        st.plotly_chart(fig3, use_container_width=False)
    else:
        st.write("Nenhum gênero encontrado.")

    st.subheader("💎 Top 10 'Joias Escondidas'")
    mediana_pop = df_final_filtered['popularity'].median()
    undervalued = df_final_filtered[(df_final_filtered['popularity'] < mediana_pop) & (df_final_filtered['vote_average'] >= 7.5) & (df_final_filtered['vote_count'] >= 100)]
    top_pearl = undervalued.sort_values(['vote_average', 'vote_count'], ascending=[False, False]).head(10)
    if not top_pearl.empty:
        fig6 = px.bar(
            top_pearl,
            x='vote_average',
            y='title',
            orientation='h',
            color='vote_average',
            color_continuous_scale='magma',
            labels={'vote_average': 'Média de Votos', 'title': 'Título do Filme'},
            height=1000,
            width=1500,
        )
        fig6.update_layout(
            title='',
            yaxis=dict(
                autorange="reversed",
                tickfont=dict(size=32),
                title=dict(text='Título do Filme', font=dict(size=36))
            ),
            xaxis=dict(
                tickfont=dict(size=32),
                title=dict(text='Nota média pelos votos', font=dict(size=36))
            ),
            font=dict(size=36),
            title_font_size=40,
        )
        st.plotly_chart(fig6, use_container_width=False)
    else:
        st.write("Nenhuma 'joia escondida' encontrada.")

    st.subheader("⏱️ Distribuição da Duração dos Filmes (Runtime)")
    fig_runtime = px.histogram(
        df_final_filtered,
        x='runtime',
        nbins=50,
        color_discrete_sequence=['purple']
    )

    fig_runtime.update_layout(
        title="",
        bargap=0.1,
        xaxis=dict(
            title=dict(text="Duração (minutos)", font=dict(size=36)),
            tickfont=dict(size=32),
        ),
        yaxis=dict(
            title=dict(text="Frequência", font=dict(size=36)),
            tickfont=dict(size=32),
        ),
        width=1500,
        height=1000,
        font=dict(size=36),
        title_font_size=40,
    )

    st.plotly_chart(fig_runtime, use_container_width=False)

    st.subheader("⭐ Popularidade vs. Nota Média")

    fig_pop = go.Figure(data=go.Histogram2dContour(
        x=df_final_filtered['popularity'],
        y=df_final_filtered['vote_average'],
        colorscale='Blues',
        reversescale=True,
        contours=dict(
            coloring='fill',
            showlines=True
        ),
        ncontours=20,
    ))

    fig_pop.update_layout(
        title='',
        xaxis=dict(
            title=dict(text="Popularidade", font=dict(size=36)),
            tickfont=dict(size=32),
            range=[0, df_final_filtered['popularity'].quantile(0.95)]
        ),
        yaxis=dict(
            title=dict(text="Nota Média", font=dict(size=36)),
            tickfont=dict(size=32)
        ),
        width=2000,
        height=1000,
        font=dict(size=36),
        title_font_size=40,
    )

    st.plotly_chart(fig_pop, use_container_width=False)


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
    
    profit_limit = st.slider("Limitar exibição de lucro (%)", 10, 5000, 500, 50)
    
    lucros_filtrados = lucros[lucros['profit_percentage'] < profit_limit]
    
    fig_lucros = px.histogram(
    lucros_filtrados,
    x='profit_percentage',
    nbins=50,
    color_discrete_sequence=['green']
    )

    fig_lucros.update_layout(
        title="",
        bargap=0.1,
        xaxis=dict(
            title=dict(text="Porcentagem de Lucro", font=dict(size=36)),
            tickfont=dict(size=32)
        ),
        yaxis=dict(
            title=dict(text="Quantidade", font=dict(size=36)),
            tickfont=dict(size=32)
        ),
        width=1500,
        height=1000,
        font=dict(size=36),
        title_font_size=40
    )

    st.plotly_chart(fig_lucros, use_container_width=False)


@st.fragment
def render_profit_boxplots(df_final_filtered):
    """
    Este fragmento renderiza os boxplot de lucro. Ele foi separado do prejuizo e dos restantes para ser colocado em uma coluna ao lado
    """
    FIG_SIZE = (8, 5) # Um pouco maior para acomodar melhor os boxplots
    FIG_DPI = 75

    # --- NOVO: Gráfico de Boxplot para Lucro por Gênero ---
    st.subheader("📊 Distribuição de Lucro Percentual por Gênero (top 10 mais frequentes)")
    df_lucro = df_final_filtered[df_final_filtered['profit_percentage'].between(0.01, 5000)] # Filtro para lucros razoáveis
    if not df_lucro.empty:
        df_lucro_box = prepare_data_for_boxplot(df_lucro)
        # Ordena os gêneros pela mediana do lucro
        order = (
            df_lucro_box.groupby('genres_translated')['profit_percentage']
            .median()
            .sort_values(ascending=False)
            .index
        )
        fig_lucro_gen = px.box(
            df_lucro_box,
            x='profit_percentage',
            y='genres_translated',
            category_orders={'genres_translated': list(order)},
            color_discrete_sequence=['lightgreen'],
            
        )
        fig_lucro_gen.update_traces(marker=dict(size=5, opacity=0.2))

        fig_lucro_gen.update_layout(
            title='',
            font=dict(size=36),
            xaxis=dict(
                title=dict(text="Lucro (%)", font=dict(size=36)),
                tickfont=dict(size=32),
                range=[0, 1300]
            ),
            yaxis=dict(
                title=dict(text="Gênero", font=dict(size=36)),
                tickfont=dict(size=32)
            ),
            width=1500,
            height=1000,
        )
        st.plotly_chart(fig_lucro_gen, use_container_width=False)
    else:
        st.write("Não há dados de lucro para exibir com os filtros atuais.")
def render_profit_loss_boxplots(df_final_filtered):
    """
    Este fragmento renderiza os novos boxplots para prejuízo por gênero.
    """
    FIG_SIZE = (8, 5) # Um pouco maior para acomodar melhor os boxplots
    FIG_DPI = 75

    st.subheader("📉 Distribuição de Prejuízos")
    prejuizos = df_final_filtered[df_final_filtered['profit_percentage'] < 0]
    st.markdown("<div style='height: 80px;'></div>", unsafe_allow_html=True) #melhora o alinhamento dos gráficos, apesar de não ficar perfeito
    fig_prejuizo = px.histogram(
        prejuizos,
        x='profit_percentage',
        nbins=50,
        color_discrete_sequence=['red']
    )
    fig_prejuizo.update_layout(
        title="",
        bargap=0.1,
        xaxis=dict(
            title=dict(text="Porcentagem de Prejuízo", font=dict(size=36)),
            tickfont=dict(size=32)
            ),
        yaxis=dict(
            title=dict(text="Quantidade", font=dict(size=36)),
            tickfont=dict(size=32)
            ),
        width=1500,
        height=1000,
        font=dict(size=36),
        title_font_size=40, 
    )
    st.plotly_chart(fig_prejuizo, use_container_width=False)

    # --- NOVO: Gráfico de Boxplot para Prejuízo por Gênero ---
    st.subheader("📊 Distribuição de Prejuízo Percentual por Gênero (top 10 mais frequentes)")
    df_prejuizo = df_final_filtered[df_final_filtered['profit_percentage'] < 0]
    if not df_prejuizo.empty:
        df_prejuizo_box = prepare_data_for_boxplot(df_prejuizo)
        order = (
            df_prejuizo_box.groupby('genres_translated')['profit_percentage']
            .median()
            .sort_values(ascending=True)
            .index
        )
        fig_prejuizo_gen = px.box(
            df_prejuizo_box,
            x='profit_percentage',
            y='genres_translated',
            category_orders={'genres_translated': list(order)},
            color_discrete_sequence=['lightcoral'],
        )
        fig_prejuizo_gen.update_layout(
            title="",
            xaxis=dict(
                title=dict(text="Prejuízo (%)", font=dict(size=36)),
                tickfont=dict(size=32),
                range=[-100, 0]
                ),
            yaxis=dict(
                title=dict(text="Gênero", font=dict(size=36)),
                tickfont=dict(size=32)
                ),
            width=1500,
            height=1000,
            font=dict(size=36),
            title_font_size=40, 
        )
        fig_prejuizo_gen.update_yaxes(tickfont=dict(size=32))
        fig_prejuizo_gen.update_xaxes(tickfont=dict(size=32))
        st.plotly_chart(fig_prejuizo_gen, use_container_width=False)
    else:
        st.write("Não há dados de prejuízo para exibir com os filtros atuais.")

@st.fragment
def render_corr(df_final_filtered):
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

    mask = np.tril(np.ones(correlation_matrix.shape), k=-1).astype(bool)
    correlation_matrix = correlation_matrix.mask(~mask)

    filtered_corr = correlation_matrix.dropna(axis=0, how='all').dropna(axis=1, how='all')

    fig_corr = px.imshow(
    filtered_corr,
    text_auto=".2f",
    color_continuous_scale="RdBu",
    zmin=-1,
    zmax=1,
    labels=dict(color="Correlação")
    )
    fig_corr.update_layout(
        title='',
        xaxis=dict(
            title=dict(text="Variáveis", font=dict(size=36)),
            tickfont=dict(size=32)
            ),
        yaxis=dict(
            title=dict(text="Variáveis", font=dict(size=36)),
            tickfont=dict(size=32)
            ),
        width=1500,
        height=1000,
        font=dict(size=36),
        title_font_size=40, 
        )
    st.plotly_chart(fig_corr, use_container_width=False)


with tab1:
    if df is not None:
        # --- Controles Ficam na Sidebar ---
        st.sidebar.header("⚙️ Filtros de Análise")

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
        colLucro, colPrejuizo = st.columns(2)
        with colLucro:
            render_profit_distribution_plot(df_final_filtered)
            render_profit_boxplots(df_final_filtered)
        with colPrejuizo:
            render_profit_loss_boxplots(df_final_filtered)
        render_corr(df_final_filtered)

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

                input_data = pd.DataFrame({
                    'budget': [budget], 'popularity': [popularity], 'runtime': [runtime],
                    'genres': [genres_formatted], 'production_companies': [production_companies_formatted],
                    'cast': [cast_formatted], 'director': [director_formatted]
                })

                with st.spinner("Processando..."):
                    pipeline = ml_artifacts['regression_pipeline']
                    prediction = pipeline.predict(input_data)
                    predicted_revenue = prediction[0]

                st.success("Previsão Concluída!")
                st.metric(label="Receita Estimada (USD)", value=f"$ {predicted_revenue:,.2f}")
