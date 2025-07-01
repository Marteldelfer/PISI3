# Projeto Interdisciplinar de Sistemas de Informação III (PISI III)

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-1.5.3-lightgrey?logo=pandas&logoColor=black)
![NumPy](https://img.shields.io/badge/NumPy-1.24-purple?logo=numpy&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12-cyan)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-1.2.2-orange?logo=scikitlearn&logoColor=white)

  <p align="center">
  <img src="https://github.com/user-attachments/assets/7856b8d7-3557-4d11-8b24-aee33b202075" width="200"/>
  <img src="https://github.com/user-attachments/assets/84f061f4-12eb-48fd-9872-fe48e744f082" width="200"/>
</p>


# Descrição

Este é o repositório visa registrar nossas contribuições para a disciplina de PISI III matéria ministrada pelo professor Dr Gabriel Alves no curso de sistemas de Informação na Universidade Federal Rural de pernambuco UFRPE. 
Esta disciplina consiste no uso de bases de dados públicos para a mineração de dados e criação de modelos (preditivos ou descritivos) de Inteligência artificial.

A base escolhida para esta empreitada é a base pública do Kaggle "The Ultimate 1 Million Movies Dataset (TMDB + IMDb)". A base escolhida trata de filmes trazendo informações relevantes tais como avaliação, diretores, orçamento, arrecadação entre outros.
A base possui ao todo mais de um milhão de registros e 28 atributos. Veja a descrição detalhada dos atributos abaixo

# Dicionário de dados

| **Atributo**             | **Descrição**                                                                 | **Tipo de Dado**      | **Exemplo**                                             |
|--------------------------|------------------------------------------------------------------------------|------------------------|----------------------------------------------------------|
| `id`                     | Identificador único do filme                                                 | Numérico (`int64`)     | 157336                                                   |
| `title`                  | Título do filme                                                              | Texto (`object`)       | Interstellar                                             |
| `vote_average`           | Média das avaliações (TMDB)                                                  | Decimal (`float64`)    | 8.454                                                    |
| `vote_count`             | Total de votos recebidos (TMDB)                                              | Decimal (`float64`)    | 37035.0                                                  |
| `release_date`           | Data de lançamento do filme                                                  | Texto, data (`object`) | 2014-11-05                                               |
| `revenue`                | Arrecadação total do filme (em dólares)                                      | Decimal (`float64`)    | 746606706.00                                             |
| `runtime`                | Duração do filme (em minutos)                                                | Decimal (`float64`)    | 169.0                                                    |
| `budget`                 | Orçamento de produção do filme (em dólares)                                  | Decimal (`float64`)    | 165000000.00                                             |
| `original_language`      | Idioma original do filme                                                     | Texto (`object`)       | en                                                       |
| `popularity`             | Índice de popularidade gerado pelo TMDB                                      | Decimal (`float64`)    | 36.0922                                                  |
| `genres`                 | Gênero(s) do filme                                                           | Texto (`object`)       | Adventure, Drama, Science Fiction                        |
| `production_companies`   | Companhias responsáveis pela produção                                        | Texto (`object`)       | Legendary Pictures, Syncopy, Lynda Obst Productions      |
| `production_countries`   | País(es) de produção dos filmes                                              | Texto (`object`)       | United Kingdom, United States of America                 |
| `cast`                   | Principais atores do filme                                                   | Texto (`object`)       | Brooke Smith, Leah Cairns, Benjamin Hardy, ...           |
| `director`               | Diretor(es) do filme                                                         | Texto (`object`)       | Christopher Nolan                                        |
| `director_of_photography`| Diretor(es) de fotografia                                                    | Texto (`object`)       | Hoyte van Hoytema                                        |
| `writers`                | Roteiristas do filme                                                         | Texto (`object`)       | Jonathan Nolan, Christopher Nolan                        |
| `producers`              | Produtores do filme                                                          | Texto (`object`)       | Kip Thorne, Emma Thomas, Jake Myers, ...                 |
| `music_composer`         | Compositor(es) da trilha sonora                                              | Texto (`object`)       | Hans Zimmer                                              |
| `imdb_rating`            | Nota de avaliação no IMDb                                                    | Decimal (`float64`)    | 8.7                                                      |
| `imdb_votes`             | Número de votos recebidos no IMDb                                            | Decimal (`float64`)    | 2337587.0                                                |
| `profit_percentage`      | Porcentagem de lucro (calculada com base em receita e orçamento)             | Decimal (`float64`)    | 352.49                                                   |

# Configurando o ambiente

Siga os passos abaixo para clonar este repositório e configurar o ambiente virtual Python:

## 1. Clone o repositório
  ```
  git clone https://github.com/Marteldelfer/PISI3.git
  cd PISI3  
```

## 2. Crie o ambiente virtual
```
  python -m venv venv 
```

## 3. Ative o ambiente virtual
```
    venv\Scripts\activate
```

## 4. Instalar dependências
```
    pip install -r requirements.txt
```

