import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def train_and_save_recommendation_artifacts():
    """
    Lê os dados, treina o modelo de recomendação e salva os artefatos.
    Aplica um tokenizer personalizado para tratar nomes compostos corretamente.
    """
    # Carrega os dados
    print("Carregando os dados de tmdb_new.csv...")
    df = pd.read_csv("tmdb_new.csv")
    
    # Seleciona as colunas relevantes e preenche valores nulos
    df = df[['id', 'title', 'genres', 'cast', 'director']]
    df.dropna(inplace=True)
    print("Dados carregados e limpos.")

    # Função de processamento de tags simplificada
    def process_tags(row):
        cast_list = row['cast'].split(', ')[:3]
        cast_str = ", ".join(cast_list)
        return f"{row['genres']}, {cast_str}, {row['director']}"

    df['tags'] = df.apply(process_tags, axis=1)
    
    # Seleciona as colunas finais
    df_rec = df[['id', 'title', 'tags']].copy()
    
    def space_remover_tokenizer(text):
        tokens = text.split(',')
        cleaned_tokens = [token.replace(" ", "").lower() for token in tokens if token.strip()]
        return cleaned_tokens

    cv = CountVectorizer(
        max_features=5000,
        tokenizer=space_remover_tokenizer 
    )
    
    print("Criando a matriz de vetores a partir das tags...")
    vectors = cv.fit_transform(df_rec['tags']).toarray()
    print("Matriz de vetores criada.")

   
    print("\nIniciando o cálculo de similaridade de cosseno... Isso pode levar alguns minutos. Por favor, aguarde.\n")
    similarity = cosine_similarity(vectors)
    print("Cálculo de similaridade concluído com sucesso!")

    # Salva os artefatos
    df_rec.to_csv('df_rec.csv', index=False)
    joblib.dump(similarity, 'cosine_sim.pkl')
    print("\nArtefatos do sistema de recomendação foram recriados e salvos com sucesso.")
    print("O problema de tokenização de nomes compostos foi corrigido.")
    print("df_rec.csv e cosine_sim.pkl estão prontos.")

if __name__ == '__main__':
    train_and_save_recommendation_artifacts()
