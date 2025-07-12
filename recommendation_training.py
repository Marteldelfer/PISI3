import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import joblib




def train_and_save_recommendation_artifacts():


    """
    Lê os dados, treina o modelo de KMeans e salva os artefatos.
    Aplica um tokenizer personalizado para tratar nomes compostos corretamente.
    """
    print("Carregando os dados de tmdb_new.csv...")
    df = pd.read_csv("tmdb_new.csv")
    
    df = df[['id', 'title', 'genres', 'cast', 'director']]
    df.dropna(inplace=True)
    print("Dados carregados e limpos.")

    def process_tags(row):
        cast_list = row['cast'].split(', ')[:3]
        cast_str = ", ".join(cast_list)
        return f"{row['genres']}, {cast_str}, {row['director']}"

    df['tags'] = df.apply(process_tags, axis=1)
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

    print("\nIniciando o treinamento com KMeans...\n")
    k = 20  # número de clusters, pode ser ajustado
    kmeans = KMeans(n_clusters=k, random_state=42)
    df_rec['cluster'] = kmeans.fit_predict(vectors)
    print("Treinamento com KMeans concluído com sucesso!")

    # Salva os artefatos
    df_rec.to_csv('df_rec.csv', index=False)
    joblib.dump(kmeans, 'kmeans_model.pkl')
    joblib.dump(cv, 'vectorizer.pkl')  # salva o vectorizer para uso futuro
    print("\nArtefatos do sistema de recomendação com KMeans foram salvos com sucesso.")

if __name__ == '__main__':
    train_and_save_recommendation_artifacts()
