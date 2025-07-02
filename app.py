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
        # Adicionar coluna de ano de lançamento para facilitar o filtro
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
        # Este retorno None será usado para exibir a mensagem de erro na aba de ML
        return None

# --- Mapeamento de Códigos de Idioma para Nomes em Português ---
LANGUAGE_CODES_TO_PORTUGUESE = {
    'en': 'Inglês',
    'fr': 'Francês',
    'ko': 'Coreano',
    'ja': 'Japonês',
    'zh': 'Chinês',
    'es': 'Espanhol',
    'de': 'Alemão',
    'hi': 'Hindi',
    'ru': 'Russo',
    'it': 'Italiano',
    'pt': 'Português',
    'ar': 'Árabe',
    'cn': 'Cantonês',
    'sv': 'Sueco',
    'da': 'Dinamarquês',
    'no': 'Norueguês',
    'fi': 'Finlandês',
    'nl': 'Holandês',
    'pl': 'Polonês',
    'th': 'Tailandês',
    'id': 'Indonésio',
    'cs': 'Checo',
    'hu': 'Húngaro',
    'tr': 'Turco',
    'el': 'Grego',
    'fa': 'Persa',
    'he': 'Hebraico',
    'te': 'Telugo',
    'ml': 'Malaiala',
    'sr': 'Sérvio',
    'bg': 'Búlgaro',
    'uk': 'Ucraniano',
    'ta': 'Tâmil',
    'ab': 'Abcázio',
    'az': 'Azerbaijano',
    'bm': 'Bâmbara',
    'bn': 'Bengali',
    'bs': 'Bósnio',
    'ca': 'Catalão',
    'dv': 'Diveí',
    'dz': 'Dzongkha',
    'et': 'Estoniano',
    'eu': 'Basco',
    'ff': 'Fula',
    'ga': 'Irlandês',
    'gl': 'Galego',
    'gu': 'Gujarati',
    'hr': 'Croata',
    'hy': 'Armênio',
    'ig': 'Ibo',
    'is': 'Islandês',
    'iu': 'Inuktitut',
    'km': 'Khmer',
    'kn': 'canarês',
    'ku': 'Curdo',
    'la': 'Latim',
    'lt': 'Lituano',
    'lv': 'Letão',
    'mn': 'Mongol',
    'mr': 'Marata',
    'ms': 'Malaio',
    'ne': 'Nepali',
    'pa': 'Panjabi',
    'ps': 'Pachto',
    'ro': 'Romeno',
    'si': 'Cingalês',
    'sk': 'Eslovaco',
    'sl': 'Esloveno',
    'sw': 'Suaíli',
    'tl': 'Tagalo',
    'tn': 'Tswana',
    'ur': 'Urdu',
    'vi': 'Vietnamita',
    'xx': 'Desconhecido',
}


# --- Funções Auxiliares para Gráficos ---
def traduzir_generos(lista_generos):
    traducoes = {
        "Action": "Ação", "Adventure": "Aventura", "Animation": "Animação", "Comedy": "Comédia",
        "Crime": "Crime", "Documentary": "Documentário", "Drama": "Drama", "Family": "Família",
        "Fantasy": "Fantasia", "History": "História", "Horror": "Terror", "Music": "Música",
        "Mystery": "Mistério", "Romance": "Romance", "Science Fiction": "Ficção Científica",
        "TV Movie": "Filme de TV", "Thriller": "Suspense", "War": "Guerra", "Western": "Faroeste"
    }
    return [traducoes.get(genero, genero) for genero in lista_generos]

def agrupar_outros(df_agrupado, top_n=8):
    if len(df_agrupado) <= top_n:
        return df_agrupado
    top = df_agrupado.sort_values('count', ascending=False).head(top_n)
    outros_df = df_agrupado.drop(top.index)
    if not outros_df.empty:
        if 'mean_profit' in outros_df.columns:
             mean_ponderada = np.average(outros_df['mean_profit'], weights=outros_df['count'])
             count_soma = outros_df['count'].sum()
             top.loc['Outros'] = {'mean_profit': mean_ponderada, 'count': count_soma}
        elif 'med_profit' in outros_df.columns:
            mediana_outros = outros_df['med_profit'].median()
            count_soma = outros_df['count'].sum()
            top.loc['Outros'] = {'med_profit': mediana_outros, 'count': count_soma}
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

# Carregando os dados e modelos no início
df = load_base_data()
ml_artifacts = load_ml_artifacts()

# --- Abas para Organização ---
tab1, tab2 = st.tabs(["📊 Análise Exploratória", "🤖 Modelos de Machine Learning"])

# ==============================================================================
# === ABA 1: ANÁLISE EXPLORATÓRIA (COMPLETA) ===================================
# ==============================================================================
with tab1:
    if df is not None:
        st.header("📄 Análise Exploratória dos Dados")
        st.write(f"**Dimensões do conjunto de dados:** `{df.shape[0]}` linhas × `{df.shape[1]}` colunas")

        # --- Filtros na Sidebar ---
        st.sidebar.header("⚙️ Filtros de Análise")

        # 1. Filtro de Intervalo de Anos de Lançamento
        df_for_filters = df.dropna(subset=['release_year']).copy()
        df_for_filters['release_year'] = df_for_filters['release_year'].astype(int)

        df_filtered_by_year = df.copy() # Inicializa com o df completo
        if not df_for_filters.empty:
            min_year_data = int(df_for_filters['release_year'].min())
            max_year_data = int(df_for_filters['release_year'].max())
            
            year_range = st.sidebar.slider(
                "📅 Intervalo de Anos de Lançamento",
                min_value=min_year_data,
                max_value=max_year_data,
                value=(min_year_data, max_year_data),
                step=1
            )
            df_filtered_by_year = df[(df['release_year'] >= year_range[0]) & (df['release_year'] <= year_range[1])]
        
        # 2. Filtro de Seleção de Gêneros
        all_genres = sorted(list(set([g.strip() for genre_str in df['genres'].dropna() for g in genre_str.split(',') if g.strip()])))
        all_genres_translated = traduzir_generos(all_genres)
        
        selected_genres_translated = st.sidebar.multiselect(
            "🎭 Selecionar Gêneros",
            options=all_genres_translated,
            default=all_genres_translated # Seleciona todos por padrão
        )
        
        reverse_traducoes = {v: k for k, v in {
            "Action": "Ação", "Adventure": "Aventura", "Animation": "Animação", "Comedy": "Comédia",
            "Crime": "Crime", "Documentary": "Documentário", "Drama": "Drama", "Family": "Família",
            "Fantasy": "Fantasia", "History": "História", "Horror": "Terror", "Music": "Música",
            "Mystery": "Mistério", "Romance": "Romance", "Science Fiction": "Ficção Científica",
            "TV Movie": "Filme de TV", "Thriller": "Suspense", "War": "Guerra", "Western": "Faroeste"
        }.items()}
        selected_genres_english = [reverse_traducoes.get(g, g) for g in selected_genres_translated]

        if selected_genres_english:
            df_filtered_by_genre = df_filtered_by_year[
                df_filtered_by_year['genres'].apply(
                    lambda x: any(genre in [g.strip() for g in str(x).split(',')] for genre in selected_genres_english) if pd.notna(x) else False
                )
            ]
        else:
            df_filtered_by_genre = df_filtered_by_year.copy()

        # 3. Filtro de Intervalo de Orçamento
        df_for_budget_filter = df_filtered_by_genre.dropna(subset=['budget']).copy()
        df_for_budget_filter['budget'] = df_for_budget_filter['budget'].astype(float)
        
        df_filtered_by_budget = df_filtered_by_genre.copy()
        if not df_for_budget_filter.empty and df_for_budget_filter['budget'].max() > 0:
            min_budget_data = float(df_for_budget_filter['budget'].min())
            max_budget_data = float(df_for_budget_filter['budget'].max())
            
            budget_range = st.sidebar.slider(
                "💸 Intervalo de Orçamento (USD)",
                min_value=min_budget_data,
                max_value=max_budget_data,
                value=(min_budget_data, max_budget_data),
                format="$%.0f"
            )
            df_filtered_by_budget = df_filtered_by_genre[
                (df_filtered_by_genre['budget'] >= budget_range[0]) &
                (df_filtered_by_genre['budget'] <= budget_range[1])
            ]
        
        # 4. Filtro de Intervalo de Receita
        df_for_revenue_filter = df_filtered_by_budget.dropna(subset=['revenue']).copy()
        df_for_revenue_filter['revenue'] = df_for_revenue_filter['revenue'].astype(float)

        df_filtered_by_revenue = df_filtered_by_budget.copy()
        if not df_for_revenue_filter.empty and df_for_revenue_filter['revenue'].max() > 0:
            min_revenue_data = float(df_for_revenue_filter['revenue'].min())
            max_revenue_data = float(df_for_revenue_filter['revenue'].max())
            
            revenue_range = st.sidebar.slider(
                "💰 Intervalo de Receita (USD)",
                min_value=min_revenue_data,
                max_value=max_revenue_data,
                value=(min_revenue_data, max_revenue_data),
                format="$%.0f"
            )
            df_filtered_by_revenue = df_filtered_by_budget[
                (df_filtered_by_budget['revenue'] >= revenue_range[0]) &
                (df_filtered_by_budget['revenue'] <= revenue_range[1])
            ]

        # 5. Filtro de Idioma Original
        # Obter todos os idiomas únicos do DataFrame filtrado até agora
        all_languages_codes = sorted(df_filtered_by_revenue['original_language'].dropna().unique().tolist())
        
        # Criar a lista de opções para o multiselect (Nome do Idioma (código))
        language_display_options = []
        for code in all_languages_codes:
            full_name = LANGUAGE_CODES_TO_PORTUGUESE.get(code, "Desconhecido") # Usa "Desconhecido" se não encontrar
            language_display_options.append(f"{full_name} ({code})")

        selected_languages_display = st.sidebar.multiselect(
            "🗣️ Idioma Original",
            options=language_display_options,
            default=language_display_options # Seleciona todos por padrão
        )

        # Mapear as seleções do usuário de volta para os códigos para filtragem
        selected_languages_codes = []
        for display_option in selected_languages_display:
            # Extrair o código entre parênteses
            code_start = display_option.rfind('(') + 1
            code_end = display_option.rfind(')')
            if code_start != -1 and code_end != -1:
                selected_languages_codes.append(display_option[code_start:code_end])
            # Se não encontrar o padrão (código), apenas tenta usar o nome completo como código, embora menos preciso
            else:
                selected_languages_codes.append(display_option) 


        if selected_languages_codes:
            df_filtered_by_language = df_filtered_by_revenue[
                df_filtered_by_revenue['original_language'].isin(selected_languages_codes)
            ]
        else:
            df_filtered_by_language = df_filtered_by_revenue.copy()

        # O DataFrame final a ser usado nos gráficos é df_filtered_by_language
        df_final_filtered = df_filtered_by_language.copy()

        # Aviso se o filtro resultar em dados vazios
        if df_final_filtered.empty:
            st.warning("Nenhum dado encontrado para os filtros selecionados. Ajuste o intervalo de anos, gêneros, orçamento, receita e/ou idiomas.")
        
        # Continuar a análise apenas se df_final_filtered não estiver vazio
        if not df_final_filtered.empty:
            with st.expander("🔍 Visualizar Amostra dos Dados Filtrados"):
                st.dataframe(df_final_filtered.head(10))

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("💰 Receita vs. Orçamento")
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                sns.scatterplot(data=df_final_filtered[df_final_filtered['budget'] > 1000], x='budget', y='revenue', alpha=0.4, ax=ax1, color='royalblue')
                ax1.set_xlabel('Orçamento (USD)')
                ax1.set_ylabel('Receita (USD)')
                st.pyplot(fig1)

                st.subheader("🎭 Top 10 Gêneros por Número de Filmes")
                genre_counts = Counter()
                for genre_str in df_final_filtered['genres'].dropna():
                    genres_list = [g.strip() for g in genre_str.split(',')]
                    genre_counts.update(genres_list)
                top_genres = genre_counts.most_common(10)
                genres_names, genres_vals = zip(*top_genres)
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                sns.barplot(x=list(genres_vals), y=traduzir_generos(list(genres_names)), palette='mako', ax=ax3)
                ax3.set_xlabel("Número de Filmes")
                ax3.set_ylabel("Gênero")
                st.pyplot(fig3)
                
                st.subheader("⏱️ Distribuição da Duração dos Filmes (Runtime)")
                fig_runtime, ax_runtime = plt.subplots(figsize=(10, 6))
                sns.histplot(df_final_filtered['runtime'].dropna(), bins=50, kde=True, ax=ax_runtime, color='purple')
                ax_runtime.set_xlabel("Duração (minutos)")
                ax_runtime.set_ylabel("Frequência")
                st.pyplot(fig_runtime)

            with col2:
                st.subheader("🌍 Nota Média por Idioma (Top 10)")
                # Aqui df_final_filtered já foi filtrado por idioma selecionado
                # A lógica abaixo é para os 10 mais frequentes DENTRO do filtro, se o filtro retornar muitos
                valid_langs = df_final_filtered['original_language'].dropna()
                valid_langs = valid_langs[valid_langs.str.isalpha()]
                
                # Se houver idiomas selecionados, filtramos apenas eles aqui
                if selected_languages_codes:
                    valid_langs = valid_langs[valid_langs.isin(selected_languages_codes)]
                
                lang_counts = valid_langs.value_counts()
                
                # O limite de 20 filmes para entrar no 'Top 10' será sobre o df_final_filtered
                # (já com os idiomas selecionados)
                frequent_langs = lang_counts[lang_counts > 20].index if not lang_counts.empty else []
                filtered_df_lang = df_final_filtered[df_final_filtered['original_language'].isin(frequent_langs)]
                
                # Se filtered_df_lang estiver vazia (e.g., nenhum idioma selecionado teve mais de 20 filmes)
                if not filtered_df_lang.empty:
                    language_ratings = filtered_df_lang.groupby('original_language')['vote_average'].mean().sort_values(ascending=False).head(10)
                else:
                    language_ratings = pd.Series() # Cria uma série vazia para evitar erro no sns.barplot

                fig2, ax2 = plt.subplots(figsize=(10, 6))
                if not language_ratings.empty: # Só plota se houver dados para o gráfico
                    sns.barplot(x=language_ratings.values, y=[LANGUAGE_CODES_TO_PORTUGUESE.get(lang, lang) for lang in language_ratings.index], palette='viridis', ax=ax2)
                    ax2.set_xlabel("Nota Média")
                    ax2.set_ylabel("Idioma")
                else:
                    ax2.text(0.5, 0.5, "Dados insuficientes para este gráfico com os filtros aplicados.", 
                             horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
                    ax2.set_xticks([])
                    ax2.set_yticks([])
                st.pyplot(fig2)

                st.subheader("💎 Top 10 'Joias Escondidas'")
                mediana_pop = df_final_filtered['popularity'].median()
                undervalued = df_final_filtered[(df_final_filtered['popularity'] < mediana_pop) & (df_final_filtered['vote_average'] >= 7.5) & (df_final_filtered['vote_count'] >= 100)]
                top_pearl = undervalued.sort_values(['vote_average', 'vote_count'], ascending=[False, False]).head(10)
                fig6, ax6 = plt.subplots(figsize=(10, 6))
                sns.barplot(data=top_pearl, x='vote_average', y='title', palette='magma', ax=ax6)
                ax6.set_xlabel("Média de Votos")
                ax6.set_ylabel("Título do Filme")
                st.pyplot(fig6)
                
                st.subheader("⭐ Popularidade vs. Nota Média")
                fig_pop, ax_pop = plt.subplots(figsize=(10, 6))
                sns.scatterplot(data=df_final_filtered, x='popularity', y='vote_average', alpha=0.3, ax=ax_pop, color='gold')
                ax_pop.set_xlabel("Popularidade")
                ax_pop.set_ylabel("Nota Média")
                ax_pop.set_xlim(0, df_final_filtered['popularity'].quantile(0.95))
                st.pyplot(fig_pop)

            st.divider()
            st.header("Análise de Lucro e Prejuízo")
            col3, col4 = st.columns(2)
            with col3:
                st.subheader("📈 Distribuição de Lucros Positivos")
                lucros = df_final_filtered[df_final_filtered['profit_percentage'] > 0]
                max_lucro = int(min(5000, lucros['profit_percentage'].max())) if not lucros.empty else 500
                limite = st.slider("Limitar exibição de lucro (%)", 10, max_lucro, value=min(500, max_lucro), step=50) # min(500, max_lucro) para valor inicial
                lucros_filtrados = lucros[lucros['profit_percentage'] < limite]
                fig_lucros, ax_lucros = plt.subplots(figsize=(10, 6))
                sns.histplot(lucros_filtrados['profit_percentage'], bins=50, kde=True, ax=ax_lucros, color='green')
                st.pyplot(fig_lucros)

                st.subheader("💹 Mediana de Lucro (%) por Gênero")
                df_lucro = df_final_filtered[df_final_filtered['profit_percentage'] > 0]
                df_lucro_agrupado = processar_por_genero(df_lucro, lucro=True)
                fig_lucro_gen, ax_lucro_gen = plt.subplots(figsize=(10, 6))
                sns.barplot(x=df_lucro_agrupado.index, y=df_lucro_agrupado['med_profit'], color='green', ax=ax_lucro_gen)
                ax_lucro_gen.set_xticklabels(ax_lucro_gen.get_xticklabels(), rotation=45, ha='right')
                st.pyplot(fig_lucro_gen)

            with col4:
                st.subheader("📉 Distribuição de Prejuízos")
                prejuizos = df_final_filtered[df_final_filtered['profit_percentage'] < 0]
                fig_prejuizo, ax_prejuizo = plt.subplots(figsize=(10, 6))
                sns.histplot(prejuizos['profit_percentage'], bins=50, kde=True, ax=ax_prejuizo, color='red')
                st.pyplot(fig_prejuizo)

                st.subheader("📉 Média de Prejuízo (%) por Gênero")
                df_prejuizo = df_final_filtered[df_final_filtered['profit_percentage'] < 0]
                df_prejuizo_agrupado = processar_por_genero(df_prejuizo, lucro=False)
                fig_prejuizo_gen, ax_prejuizo_gen = plt.subplots(figsize=(10, 6))
                sns.barplot(x=df_prejuizo_agrupado.index, y=-df_prejuizo_agrupado['mean_profit'], color='red', ax=ax_prejuizo_gen)
                ax_prejuizo_gen.set_xticklabels(ax_prejuizo_gen.get_xticklabels(), rotation=45, ha='right')
                st.pyplot(fig_prejuizo_gen)

            st.divider()
            st.header("Análise de Correlações")
            numeric_cols = df_final_filtered.select_dtypes(include=np.number).columns.tolist()
            correlation_matrix = df_final_filtered[numeric_cols].corr()
            fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
            sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax_corr)
            st.pyplot(fig_corr)

# ==============================================================================
# === ABA 2: MACHINE LEARNING ==================================================
# ==============================================================================
with tab2:
    st.header("🤖 Modelos de Machine Learning")

    if ml_artifacts is None:
        st.error(
            "**Arquivos dos modelos não encontrados!** Por favor, execute os scripts de treinamento para gerá-los."
        )
    else:
        # --- Seção 1: Sistema de Recomendação ---
        st.subheader("🍿 Sistema de Recomendação de Filmes")
        st.markdown("Selecione um filme e veja 5 recomendações baseadas no conteúdo (gênero, elenco, diretor, etc.).")
        
        df_rec = ml_artifacts['df_rec']
        cosine_sim = ml_artifacts['cosine_sim']
        indices = pd.Series(df_rec.index, index=df_rec['title']).drop_duplicates()

        def get_recommendations(title, cosine_sim=cosine_sim):
            idx = indices[title]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:6]
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

        # --- Seção 2: Previsão de Receita ---
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
                    'budget': [budget],
                    'popularity': [popularity],
                    'runtime': [runtime],
                    'genres': [genres],
                    'production_companies': [production_companies],
                    'cast': [cast],
                    'director': [director]
                })

                with st.spinner("Processando..."):
                    pipeline = ml_artifacts['regression_pipeline']
                    prediction = pipeline.predict(input_data)
                    predicted_revenue = prediction[0]

                st.success("Previsão Concluída!")
                st.metric(label="Receita Estimada (USD)", value=f"$ {predicted_revenue:,.2f}")