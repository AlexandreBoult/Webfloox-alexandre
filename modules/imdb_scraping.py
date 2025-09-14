import os
import time
import re
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import requests
from bs4 import BeautifulSoup
from models import Movie, Base, MovieCritic
from dotenv import dotenv_values,load_dotenv

# ========================
# Config
# ========================
load_dotenv()
DB_USER=os.getenv("POSTGRES_USER")
DB_PASSWORD=os.getenv("POSTGRES_PASSWORD")
DB_HOST=os.getenv("POSTGRES_HOST")
DB_PORT=os.getenv("POSTGRES_PORT")
DB_NAME=os.getenv("POSTGRES_DB")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ========================
# DB connection
# ========================
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
session = SessionLocal()


def search_imdb(movie_title):
    """
    Recherche un film sur IMDb et retourne l'URL de sa page.
    """
    search_url = f"https://www.imdb.com/find/?s=tt&q={movie_title.replace(' ', '%20')}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        response = requests.get(search_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Le premier résultat de recherche est généralement le plus pertinent
        first_result = soup.find('a', {'class': 'ipc-metadata-list-summary-item__t'})
        if first_result:
            return "https://www.imdb.com" + first_result['href'].split("/?ref")[0]
        else:
            return None

    except requests.exceptions.RequestException as e:
        print(f"Erreur de requête : {e}")
        return None

def get_movie_reviews(movie_url):
    """
    Extrait les critiques d'un film à partir de l'URL de sa page IMDb.
    """
    if not movie_url:
        print("URL du film non trouvée.")
        return []

    # Construction de l'URL de la page des critiques
    if not movie_url.endswith('/'):
        movie_url += '/'
    reviews_url = movie_url + "reviews"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        print(f"Extraction des critiques de : {reviews_url}")
        response = requests.get(reviews_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        reviews_data = []
        review_divs = soup.find_all('div', class_='ipc-list-card--border-speech ipc-list-card--hasActions ipc-list-card--base ipc-list-card sc-b8d6d2b6-0 CHbIK')

        for review in review_divs:
            try:
                review_title_tag = review.find('h3', class_='ipc-title__text ipc-title__text--reduced')
                review_title = review_title_tag.get_text(strip=True) if review_title_tag else None

                review_content_tag = review.find('div', class_='ipc-html-content-inner-div')
                review_content = review_content_tag.get_text(strip=True) if review_content_tag else None

                review_rating_tag = review.find('span', class_='ipc-rating-star--rating')
                review_rating = review_rating_tag.get_text(strip=True) if review_rating_tag else None
                if review_title != None and review_content != None and review_title != None:
                    reviews_data.append({
                        'title': review_title,
                        'content': review_content,
                        'rating' : review_rating
                    })
            except Exception as e:
                print(f"Erreur lors de l'extraction d'une critique : {e}")
                continue

        return reviews_data

    except requests.exceptions.RequestException as e:
        print(f"Erreur de requête : {e}")
        return []

def insert_review(review_dict: dict, source: str = "external"):
    """
    Inserts a review into movie_critics table.
    review_dict should have keys: title, content, rating, movie_id
    """
    review = MovieCritic(
        movie_id=int(review_dict["movie_id"]),
        critic_title=review_dict["title"],
        review_text=review_dict["content"],
        rating=int(review_dict["rating"]),
        source=source
    )
    session.add(review)
    session.commit()
    print(f"✅ Inserted review for movie_id={review.movie_id}: '{review.critic_title}'")

# ========================
# Extract titles
# ========================
def get_all_movie_titles():
    titles = session.query(Movie.movie_id, Movie.title).all()
    # SQLAlchemy returns a list of tuples -> flatten to list of strings
    return [t for t in titles]

if __name__ == "__main__":
    titles = get_all_movie_titles()
    for t in titles:
        print(t[1])
        movie_url = search_imdb(t[1])
        reviews = get_movie_reviews(movie_url)
        k=0
        for r in reviews:
            if k > 10 : break
            print(r)
            r_dict=r.copy()
            r_dict['movie_id']=t[0]
            try :
                time.sleep(0.1)
                insert_review(r_dict,"IMDb")
                k+=1
            except :
                print("Invalid review skipped !")
