import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def generate_recommendation_artifacts():
    """
    Função para gerar e salvar os arquivos necessários para o sistema de recomendação.
    """
    print("--- Iniciando a criação dos artefatos do sistema de recomendação ---")

    # 1. Carregar os dados base
    try:
        df = pd.read_csv("tmdb_new.csv")
        print("Arquivo 'tmdb_new.csv' carregado com sucesso.")
    except FileNotFoundError:
        print("ERRO: 'tmdb_new.csv' não encontrado. Certifique-se de que está na mesma pasta.")
        return

    # 2. Preparar os dados para recomendação
    # Selecionar colunas de texto que descrevem o filme
    features_rec = ['genres', 'director', 'cast', 'production_companies', 'writers', 'title']
    
    df_rec = df.copy()

    # Tratar valores nulos nessas colunas, preenchendo com string vazia
    for feature in features_rec:
        df_rec[feature] = df_rec[feature].fillna('')

    # 3. Criar a "sopa de metadados" - VERSÃO CORRIGIDA
    # Apenas concatenamos as strings, o vetorizador cuidará de separar as palavras.
    def create_soup(x):
        return (str(x['genres']) + ' ' + 
                str(x['director']) + ' ' + 
                str(x['cast']) + ' ' + 
                str(x['production_companies']) + ' ' + 
                str(x['writers']) + ' ' + 
                str(x['title']))

    df_rec['soup'] = df_rec.apply(create_soup, axis=1)
    print("Coluna 'soup' de metadados criada.")

    # 4. Usar TF-IDF para vetorizar a "sopa"
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_rec['soup'])
    print("Matriz TF-IDF criada.")

    # 5. Calcular a matriz de similaridade de cossenos
    print("Calculando a matriz de similaridade de cossenos... (Isso pode levar um momento)")
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    print("Cálculo de similaridade concluído.")

    # 6. Preparar o DataFrame final para o app e salvar os artefatos
    df_to_save = df[['title']].copy().reset_index()

    output_df_filename = 'df_rec.csv'
    output_matrix_filename = 'cosine_sim.pkl'

    df_to_save.to_csv(output_df_filename, index=False)
    joblib.dump(cosine_sim, output_matrix_filename)

    print(f"\nArquivos '{output_df_filename}' e '{output_matrix_filename}' salvos com sucesso!")
    print("--- Processo de recomendação finalizado ---")


if __name__ == '__main__':
    generate_recommendation_artifacts()