import os
import requests
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base, Movie
from dotenv import dotenv_values,load_dotenv

# ========================
# Config
# ========================
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"


DB_USER=os.getenv("POSTGRES_USER")
DB_PASSWORD=os.getenv("POSTGRES_PASSWORD")
DB_HOST=os.getenv("POSTGRES_HOST")
DB_PORT=os.getenv("POSTGRES_PORT")
DB_NAME=os.getenv("POSTGRES_DB")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
session = SessionLocal()

# ========================
# TMDB helpers
# ========================
def fetch_genres():
    """Fetch genre list from TMDB and return mapping {id: name}"""
    url = f"{TMDB_BASE_URL}/genre/movie/list"
    params = {"api_key": TMDB_API_KEY, "language": "en-US"}
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    genres = resp.json()["genres"]
    return {g["id"]: g["name"] for g in genres}

def fetch_popular_movies(page=1):
    url = f"{TMDB_BASE_URL}/movie/popular"
    params = {"api_key": TMDB_API_KEY, "language": "en-US", "page": page}
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()["results"]

def fetch_movie_details(movie_id):
    url = f"{TMDB_BASE_URL}/movie/{movie_id}"
    params = {"api_key": TMDB_API_KEY, "language": "en-US"}
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()

# ========================
# Insert into DB
# ========================
def insert_movie(movie_data, genre_map):
    existing = session.query(Movie).filter_by(movie_id=movie_data["id"]).first()
    if existing:
        print(f"Movie {movie_data['title']} already exists, skipping.")
        return

    # Map genre IDs → names
    genre_names = [genre_map.get(gid, str(gid)) for gid in movie_data.get("genre_ids", [])]

    movie = Movie(
        movie_id=movie_data["id"],
        title=movie_data["title"],
        year=int(movie_data["release_date"].split("-")[0]) if movie_data.get("release_date") else None,
        synopsis=movie_data.get("overview", ""),
        genre=", ".join(genre_names) if genre_names else None
    )
    session.add(movie)
    session.commit()
    print(f"Inserted {movie.title} ({movie.year}) | Genres: {movie.genre}")

# ========================
# Main
# ========================
def import_popular_movies(pages=1):
    genre_map = fetch_genres()
    for page in range(1, pages + 1):
        print(f"Fetching page {page}...")
        movies = fetch_popular_movies(page=page)
        for m in movies:
            insert_movie(m, genre_map)

if __name__ == "__main__":
    Base.metadata.create_all(engine)  # make sure tables exist
    import_popular_movies(pages=100)    # import first 100 pages (~2000 movies)
