import csv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv
import requests
from models import Base, Movie, MovieCritic, User # Assurez-vous que models.py est dans le même répertoire

# =========================
# Configuration de la BDD
# =========================

load_dotenv()
DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_HOST = os.getenv("POSTGRES_HOST")
DB_PORT = os.getenv("POSTGRES_PORT")
DB_NAME = os.getenv("POSTGRES_DB")

# Assurez-vous d'ajouter votre clé API TMDb à votre fichier .env
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

# =========================
# Fonctions d'aide TMDb
# =========================

def get_movie_details_from_tmdb(imdb_id: str, api_key: str) -> tuple:
    """
    Récupère les informations détaillées d'un film depuis l'API de TMDb
    en utilisant son ID IMDb.

    Args:
        imdb_id (str): L'ID IMDb du film (ex: "tt0369610").
        api_key (str): La clé API de TMDb.

    Returns:
        tuple: Un tuple contenant (tmdb_id, synopsis) ou (None, None) en cas d'échec.
    """
    if not api_key:
        print("Erreur : La clé API TMDb est manquante.")
        return None, None

    # Endpoint de recherche par ID externe
    url = f"https://api.themoviedb.org/3/find/{imdb_id}?api_key={api_key}&language=en-US&external_source=imdb_id"

    try:
        response = requests.get(url)
        response.raise_for_status() # Lève une exception pour les codes d'état d'erreur
        data = response.json()

        # L'API TMDb retourne les films dans le champ "movie_results"
        if data.get("movie_results"):
            movie_result = data["movie_results"][0]
            tmdb_id = movie_result.get("id")
            synopsis = movie_result.get("overview", "")
            return tmdb_id, synopsis
        else:
            print(f"Avertissement : Aucun résultat trouvé sur TMDb pour l'ID IMDb {imdb_id}.")
            return None, None

    except requests.exceptions.RequestException as e:
        print(f"Erreur de requête à l'API TMDb : {e}")
        return None, None
    except IndexError:
        print(f"Avertissement : Aucun résultat valide trouvé pour l'ID IMDb {imdb_id}.")
        return None, None

# =========================
# Script d'importation principal
# =========================

def import_movies_from_csv(csv_filepath, session):
    """
    Importe les films depuis un fichier CSV et enrichit les données
    avec la synopsis de TMDb.

    Args:
        csv_filepath (str): Chemin d'accès au fichier CSV contenant les films.
        session (Session): Session SQLAlchemy pour l'interaction avec la base de données.
    """
    if not TMDB_API_KEY:
        print("Erreur : Impossible de continuer sans la clé API TMDb.")
        return

    try:
        with open(csv_filepath, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            movies_to_import = list(reader)

        print(f"Début de l'importation de {len(movies_to_import)} films...")

        imported_count = 0

        existing_movies = session.query(Movie).all()
        existing_movie_ids = {movie.movie_id for movie in existing_movies}

        for row in movies_to_import:
            imdb_id = row.get("id")
            title = row.get("title")
            genre = row.get("genre")
            year = row.get("year")

            # Récupérer les informations de TMDb
            tmdb_id, synopsis = get_movie_details_from_tmdb(imdb_id, TMDB_API_KEY)

            if tmdb_id and not tmdb_id in existing_movie_ids:
                # Créer une instance de Movie
                new_movie = Movie(
                    movie_id=tmdb_id, # On stocke l'ID TMDb dans tmdb_id
                    title=title,
                    year=year,
                    synopsis=synopsis,
                    genre=genre
                )
                session.add(new_movie)
                imported_count += 1

        session.commit()
        print(f"Importation terminée. {imported_count} films importés avec succès.")

    except FileNotFoundError:
        print(f"Erreur : Le fichier {csv_filepath} n'a pas été trouvé.")
    except Exception as e:
        session.rollback()
        print(f"Une erreur s'est produite lors de l'importation : {e}")
    finally:
        session.close()

if __name__ == "__main__":
    db_session = SessionLocal()
    import_movies_from_csv("imdb_list.csv", db_session)
