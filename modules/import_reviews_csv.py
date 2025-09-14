import csv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv
from models import Base, MovieCritic, Movie, User # Assurez-vous que models.py est dans le même répertoire
import requests

# =========================
# Configuration de la BDD
# =========================

load_dotenv()
DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_HOST = os.getenv("POSTGRES_HOST")
DB_PORT = os.getenv("POSTGRES_PORT")
DB_NAME = os.getenv("POSTGRES_DB")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

def get_movie_id_from_tmdb(imdb_id: str, api_key: str) -> tuple:
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
        return None

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
            return tmdb_id
        else:
            print(f"Avertissement : Aucun résultat trouvé sur TMDb pour l'ID IMDb {imdb_id}.")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Erreur de requête à l'API TMDb : {e}")
        return None
    except IndexError:
        print(f"Avertissement : Aucun résultat valide trouvé pour l'ID IMDb {imdb_id}.")
        return None

# =========================
# Script d'importation
# =========================

def import_reviews_from_csv(csv_filepath, session):
    """
    Importe les critiques de films depuis un fichier CSV.

    Args:
        csv_filepath (str): Chemin d'accès au fichier CSV contenant les critiques.
        session (Session): Session SQLAlchemy pour l'interaction avec la base de données.
    """
    try:
        # Lire les données du fichier CSV
        with open(csv_filepath, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            reviews_to_import = list(reader)

        print(f"Début de l'importation de {len(reviews_to_import)} critiques...")

        # Récupérer les identifiants de films existants pour la validation
        existing_movies = session.query(Movie).all()
        existing_movie_ids = {movie.movie_id for movie in existing_movies}
        existing_critics = session.query(MovieCritic).all()
        existing_critic_titles = {review.critic_title for review in existing_critics}

        id_dict={}
        imdb_id_set=set()

        for row in reviews_to_import:
            imdb_id_set.add(row.get("imdb_id"))
        for e in imdb_id_set:
            id_dict[e]=get_movie_id_from_tmdb(e, TMDB_API_KEY)

        # print(existing_movie_ids)

        # Initialiser le compteur de critiques importées
        imported_count = 0

        # Parcourir chaque ligne du fichier CSV
        for row in reviews_to_import:
            imdb_id = row.get("imdb_id")
            review_title = row.get("review title")
            review_text = row.get("review")
            rating = row.get("review_rating")
            source = "IMDb"  # Source fixe pour toutes les critiques IMDb
            tmdb_id=id_dict[imdb_id]

            # Vérifier si l'ID IMDb existe dans notre table de films
            if tmdb_id in existing_movie_ids and not review_title in existing_critic_titles:

                # Créer une instance de MovieCritic
                new_critic = MovieCritic(
                    movie_id=tmdb_id,
                    critic_title=review_title,
                    review_text=review_text,
                    rating=int(rating) if rating else None,
                    source=source
                )
                session.add(new_critic)
                imported_count += 1
            else:
                # print(f"Avertissement : Film avec l'ID IMDb {imdb_id} non trouvé dans la BDD. Critique ignorée.")
                pass

        # Commettre les changements à la base de données
        session.commit()
        print(f"Importation terminée. {imported_count} critiques de films importées avec succès.")

    except FileNotFoundError:
        print(f"Erreur : Le fichier {csv_filepath} n'a pas été trouvé.")
    except Exception as e:
        session.rollback()
        print(f"Une erreur s'est produite lors de l'importation : {e}")
    finally:
        session.close()

if __name__ == "__main__":
    db_session = SessionLocal()
    import_reviews_from_csv("imdb_reviews.csv", db_session)
