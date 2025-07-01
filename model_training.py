import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

def train_and_save_pipeline():
    """
    Função principal para carregar dados, treinar o modelo e salvar o pipeline completo.
    """
    print("--- Iniciando o processo de treinamento ---")

    # 1. Carregar os dados
    try:
        df = pd.read_csv("tmdb_new.csv")
        print("Arquivo 'tmdb_new.csv' carregado com sucesso.")
    except FileNotFoundError:
        print("ERRO: 'tmdb_new.csv' não encontrado. Certifique-se de que está na mesma pasta.")
        return

    # 2. Limpeza e Seleção de Features
    # Usaremos apenas as colunas necessárias para o modelo e removeremos linhas com dados faltantes nelas.
    features = [
        'budget', 'popularity', 'runtime', 'genres', 'production_companies',
        'cast', 'director', 'revenue'
    ]
    df_model = df[features].copy()
    df_model.dropna(inplace=True)
    print(f"Dados limpos. O modelo será treinado com {len(df_model)} amostras.")


    # 3. Definir Features (X) e Alvo (y)
    X = df_model.drop('revenue', axis=1)
    y = df_model['revenue']


    # 4. Definir o pré-processador com ColumnTransformer
    # Isso garante que cada tipo de coluna receba o tratamento correto.
    numeric_features = ['budget', 'popularity', 'runtime']
    text_features = ['genres', 'production_companies', 'cast', 'director']

    # Criamos um transformador para colunas numéricas e um para cada coluna de texto
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            # Usamos CountVectorizer para transformar texto em vetores numéricos
            ('genres', CountVectorizer(), 'genres'),
            ('companies', CountVectorizer(), 'production_companies'),
            # Limitamos o número de atores para evitar uma matriz muito grande
            ('cast', CountVectorizer(max_features=200, stop_words='english'), 'cast'),
            ('director', CountVectorizer(), 'director')
        ],
        remainder='drop'  # Ignora colunas não especificadas
    )

    # 5. Criar o Pipeline Completo
    # O Pipeline encadeia o pré-processamento e o modelo de regressão.
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])


    # 6. Treinar o Pipeline
    print("\nIniciando o treinamento do pipeline... (Isso pode levar alguns minutos)")
    # O pipeline inteiro é treinado com os dados brutos. Ele aplica o pré-processamento internamente.
    model_pipeline.fit(X, y)
    print("Treinamento concluído com sucesso!")


    # 7. Salvar o Pipeline
    output_filename = 'full_model_pipeline.pkl'
    joblib.dump(model_pipeline, output_filename)
    print(f"\nPipeline completo salvo com sucesso como '{output_filename}'")
    print("--- Processo de treinamento finalizado ---")


if __name__ == '__main__':
    train_and_save_pipeline()