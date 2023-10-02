from fastapi import FastAPI
import pandas as pd
import uvicorn
from sklearn.metrics.pairwise import cosine_similarity

app=FastAPI(title='PI n°1:Steam games', description='Betiana Lopez Andueza')

@app.get('/')
async def read_root():
    return {'Dirígite a /docs'}

df_generos_horas_jugadas=pd.read_csv('Data/generos_horas_jugadas.csv')
df_usuario_horas_jugadas=pd.read_parquet('Data/usuario_horas_jugadas.parquet')
df_recomendaciones_positivas=pd.read_csv('Data/recomendaciones_positivas.csv')
df_recomendaciones_negativas=pd.read_csv('Data/recomendaciones_negativas.csv')
df_sentiment_analysis=pd.read_csv('Data/sentiment_analysis.csv')
df_ml=pd.read_csv('Data/modelo_ml.csv')


@app.get("/PlayTimeGenre/{genero:str}")
def PlayTimeGenre(genero: str):
    '''
    Debe devolver año con mas horas jugadas para dicho género.
    '''
    #Vamos a utilizar el dataframe df_generos_horas_jugadas, que recordar que ya se encuentra agrupado en el archivo 04_tratamiento_datasets
    
    genero=genero.lower()

    # Filtramos el DataFrame por el género especificado
    df_filtrado = df_generos_horas_jugadas[df_generos_horas_jugadas['genres'].str.lower() == genero]

    # Encontrar el año con la máxima suma de horas jugadas
    año_max_horas = df_filtrado.loc[df_filtrado['playtime_forever'].idxmax()]

    return("El año con más horas jugadas para el genero " + str(genero).capitalize() + " es " + str(año_max_horas['release_year']))


@app.get("/UserForGenre/{genero:str}")
def UserForGenre(genero: str):
    
    '''
    Debe devolver el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año.
    
    '''

    genero=genero.lower()
    # Filtra el dataframe df_final2 por el género solicitado
    df_filtrado = df_usuario_horas_jugadas.loc[df_usuario_horas_jugadas["genres"].str.lower() == genero]
    # Comprueba si hay datos para el género solicitado

    
    if df_filtrado.empty:
        return "El genero ingresado no es válido"
    
    # Agrupa el dataframe filtrado por el usuario y suma las horas jugadas
    df_user = df_filtrado.groupby("user_id")["playtime_forever"].sum()

    # Obtiene el usuario con más horas jugadas para el género solicitado
    user = df_user.idxmax()

    # Filtra el dataframe filtrado por el usuario obtenido
    df_user_genre = df_filtrado.loc[df_filtrado["user_id"] == user]

    # Agrupa el dataframe filtrado por el año y suma las horas jugadas
    df_user_year = df_user_genre.groupby("release_year")["playtime_forever"].sum()

    # Crea una lista con la acumulación de horas jugadas por año
    hours = [{"Año": year, "Horas": hours} for year, hours in df_user_year.items()]
    
    # Devuelve el usuario con más horas jugadas para el género solicitado y la lista de la acumulación de horas jugadas por año en formato JSON
    return {f"Usuario con más horas jugadas para el género {genero}": user, "Horas jugadas": hours}
  


@app.get("/UsersRecommend/{año:int}")
def UsersRecommend(año: int):

    '''
    Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado. (reviews.recommend = True y comentarios positivos/neutrales)
    '''

    #Se utiliza el dataframe df_recomendaciones_positivas que ya se encuentra filtrado por recommend y sentimiento en el archivo 04_tratamiento_datasets.

    df_filtrado = df_recomendaciones_positivas[df_recomendaciones_positivas['year_review'] == año]

    if df_filtrado.empty:
        return f"No se encontraron datos para el año ingresado"

    # Agrupo las filas contando las recomendaciones por titulo
    recomendaciones = df_filtrado.groupby('title')['recommend'].count().reset_index()

    # Ordeno en orden descendente
    top_games = recomendaciones.sort_values(by='recommend', ascending=False)

    # Selecciono los 3 juegos principales
    top_3_games = top_games.head(3)

    # resultado
    resultado = [{"Puesto {}: {}".format(i + 1, juego)} for i, juego in enumerate(top_3_games['title'])]

    return resultado
    


@app.get("/UsersNotRecommend/{año:int}")
def UsersNotRecommend(año: int):

    '''
    Devuelve el top 3 de juegos MENOS recomendados por usuarios para el año dado. (reviews.recommend = False y comentarios negativos)
    
    '''
    #Se utiliza el dataframe df_recomendaciones_negativas que ya se encuentra filtrado por recommend y sentimiento en el archivo 04_tratamiento_dataset
    
    df_filtrado = df_recomendaciones_negativas[df_recomendaciones_negativas['year_review'] == año]

    if df_filtrado.empty:
        return f"No se encontraron juegos para el año ingresado"

    # Agrupo las filas por el nombre del juego y cuento las no recomendaciones
    recomendaciones = df_filtrado.groupby('title')['recommend'].count().reset_index()

    # Ordeno en orden descendente
    bottom_games = recomendaciones.sort_values(by='recommend', ascending=True)

    # Selecciono los 3 juegos principales
    bottom_3_games = bottom_games.head(3)

    # resultado
    resultado = [{"Puesto {}: {}".format(i + 1, juego)} for i, juego in enumerate(bottom_3_games['title'])]

    return resultado


@app.get("/sentiment_analysis/{año:int}")
def sentiment_analysis(año):
    '''
    Según el año de lanzamiento, se devuelve una lista con la cantidad de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento.
    '''

    #Se utiliza el dataframe df_sentiment_analysis que ya se encuentra filtrado por recommend y sentimiento en el archivo 04_tratamiento_para_funciones.ipynb
    
    # Filtrar el DataFrame por el año de lanzamiento especificado
    df_filtrado = df_sentiment_analysis[df_sentiment_analysis['release_year'] == int(año)]

    if df_filtrado.empty:
        return 'No hay datos para el año ingresado'
    
    # Contar la cantidad de registros de cada categoría de sentimiento
    conteo_sentimiento = df_filtrado['sentimiento'].value_counts().to_dict()

    # Crear un diccionario con etiquetas descriptivas
    resultado = {
        'Negative': conteo_sentimiento.get(0, 0),
        'Neutral': conteo_sentimiento.get(1, 0),
        'Positive': conteo_sentimiento.get(2, 0)
    }

    return resultado

@app.get("/recomendar_juegos/{id_juego:int}")
def recomendar_juegos(id_juego: int):

    '''
    Ingresando el id del juego, deberíamos recibir una lista con 5 juegos recomendados similares al ingresado.
    
    '''
    
    if id_juego not in df_ml['id'].values:
        return "El id ingresado no se encuentra en la base de datos."

    # Encuentra el índice del juego en la matriz de similitud
    juego_index = df_ml[df_ml['id'] == id_juego].index[0]

    # Selecciona las columnas relevantes (las que contienen 0 y 1 para géneros)
    columnas_generos = df_ml.columns[2:]

    # Calcula la similitud de coseno entre juegos basada en las características de género (item a item)
    similarity_matrix = cosine_similarity(df_ml[columnas_generos].T)

    # Obtén los juegos más similares (excluyendo el juego en sí mismo)
    similar_juegos_indices = similarity_matrix[juego_index].argsort()[::-1][1:]

    # Obtén los títulos de los juegos recomendados (los primeros 5)
    juegos_recomendados = df_ml['title'].iloc[similar_juegos_indices[:5]].values.tolist()

    return juegos_recomendados