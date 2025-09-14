from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from typing import List
from uuid import uuid4
from pydantic import BaseModel
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
import os
from dotenv import load_dotenv
import sys
sys.path.append("modules")
from models import Base, User, Movie, LikedMovie, MovieCritic  # SQLAlchemy models
import hashlib
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np
from sklearn.cluster import KMeans
import random as rd
from collections import Counter

def remove_infrequent_elements(lst, n):
  """
  Removes elements from a list that appear less than n times.

  Args:
    lst: The input list.
    n: The minimum number of occurrences for an element to be kept.

  Returns:
    A new list with the infrequent elements removed.
  """
  # Count the frequency of each element in the list
  counts = Counter(lst)

  # Create a new list containing only elements that appear at least n times
  filtered_list = [item for item in lst if counts[item] >= n]

  return filtered_list

def add_noise_to_vector(embedding_vector, noise_level: float, rng):
    """
    Adds Gaussian noise to an embedding vector.

    :param embedding_vector: A NumPy array representing the embedding.
    :param noise_level: The standard deviation of the Gaussian noise.
    :return: The noisy embedding vector as a NumPy array.
    """
    noise = rng.normal(loc=0.0, scale=noise_level, size=embedding_vector.shape)
    return embedding_vector + noise

def sample_vectors_with_kmeans(vectors: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Samples a fixed number of vectors from a larger set, maintaining a similar
    distribution using K-Means clustering.

    This function treats the input vectors as data points and groups them into
    'n_samples' clusters. The center of each cluster is then returned as the
    representative vector for that group.

    Args:
        vectors (np.ndarray): A NumPy array of vectors, with shape (n_vectors, n_features).
        n_samples (int): The desired number of output vectors.

    Returns:
        np.ndarray: A NumPy array containing the cluster centers, representing the
                    sampled vectors, with shape (n_samples, n_features).
    """
    # Check if the number of samples requested is more than the total vectors available.
    if n_samples > len(vectors):
        print("Warning: n_samples is greater than the number of input vectors.")
        print(f"Returning the original {len(vectors)} vectors.")
        return vectors

    # Initialize the KMeans model with the desired number of clusters.
    # The random_state is set for reproducibility.
    kmeans = KMeans(n_clusters=n_samples, random_state=0, n_init='auto')

    # Fit the model to the vectors. This is where the clustering happens.
    kmeans.fit(vectors)

    # The cluster centers are the representatives of the distribution.
    sampled_vectors = kmeans.cluster_centers_

    return sampled_vectors

app = FastAPI(title="Movie Recommendation API", version="0.2")

# =====================
# Database setup
# =====================
load_dotenv()
DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_HOST = os.getenv("POSTGRES_HOST")
DB_PORT = os.getenv("POSTGRES_PORT")
DB_NAME = os.getenv("POSTGRES_DB")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = int(os.getenv("QDRANT_PORT"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")

qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

Base.metadata.create_all(bind=engine)

# =====================
# Auth
# =====================
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/login")
tokens = {}  # token -> user_id mapping

# =====================
# Pydantic Schemas
# =====================
class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class MovieOut(BaseModel):
    id: int
    title: str
    genre: str


class LikeRequest(BaseModel):
    movie_id: int

class RegisterRequest(BaseModel):
    username: str
    password: str

class Feedback(BaseModel):
    grade : int
    seed : int


class DeleteUser(BaseModel):
    yes: int

# =====================
# Dependencies
# =====================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def authenticate_user(db: Session, username: str, password: str):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return None
    # compare hashes
    hashed = hashlib.sha256((password + user.salt).encode()).hexdigest()
    if user.password_hash != hashed:
        return None
    return user

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    user_id = tokens.get(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

def find_similar_critics(db, current_user, rd_state=0):
    liked = db.query(LikedMovie).filter(LikedMovie.user_id == current_user.user_id).all()
    liked_ids = [l.movie_id for l in liked]

    if not liked_ids:
        # fallback: return first 3 movies
        movies = db.query(Movie).limit(20).all()
        return [m.movie_id for m in movies]

    all_critics=[]
    for l_id in liked_ids:
        critics=db.query(MovieCritic).filter(MovieCritic.movie_id == l_id).all()
        all_critics=all_critics+critics
    critic_ids=[c.critic_id for c in all_critics]

    points = qdrant_client.retrieve(collection_name=QDRANT_COLLECTION, ids=critic_ids, with_vectors=True)
    vectors = [p.vector for p in points if p.vector is not None]

    if not vectors:
        movies = db.query(Movie).limit(20).all()
        return [m.movie_id for m in movies]

    # Average liked embeddings
    query_vectors=sample_vectors_with_kmeans(vectors,10)

    rng=np.random.RandomState(rd_state)

    for i in range(len(query_vectors)):
        query_vectors[i]=add_noise_to_vector(np.array(query_vectors[i]), 0.03, rng).tolist()

    # query_vector = np.mean(query_vectors, axis=0).tolist()
    similar_critics=[]
    for query_vector in query_vectors:
        similar_critics = similar_critics + qdrant_client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            limit=20,
            # score_threshold=0.0,
            with_payload=True
        ).points
    # Filter out already liked
    similar_critics_filtered = [r for r in similar_critics if r.id not in critic_ids]
    similar_critics_filtered.sort(key=lambda point: point.score, reverse=True)
    new_critics=[]

    for e in similar_critics_filtered:
        critics=db.query(MovieCritic).filter(MovieCritic.critic_id == e.id).all()
        new_critics=new_critics+critics
    return new_critics,liked_ids

# =====================
# Endpoints
# =====================
@app.post("/api/v1/register")
def register_user(request: RegisterRequest, db: Session = Depends(get_db)):
    # Check if username or email already exists
    existing_user = db.query(User).filter(
        (User.username == request.username)
    ).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username or email already registered")

    # Generate salt & hash password
    salt = uuid4().hex
    password_hash = hashlib.sha256((request.password + salt).encode()).hexdigest()

    new_user = User(
        username=request.username,
        password_hash=password_hash,
        salt=salt
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {"message": f"User {new_user.username} registered successfully"}

@app.post("/api/v1/login", response_model=TokenResponse)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    token = str(uuid4())
    tokens[token] = user.user_id
    return {"access_token": token, "token_type": "bearer"}

@app.get("/api/v1/movies/{id}", response_model=MovieOut)
def get_movie(id: int, db: Session = Depends(get_db)):
    movie = db.query(Movie).filter(Movie.movie_id == id).first()
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")
    return MovieOut(id=movie.movie_id, title=movie.title, genre=movie.genre)

@app.get("/api/v1/movies_liked", response_model=List[MovieOut])
def liked_movies(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    movies = db.query(LikedMovie).filter(LikedMovie.user_id == current_user.user_id).all()
    results = [db.query(Movie).filter(Movie.movie_id == m.movie_id).first() for m in movies]
    return [MovieOut(id=m.movie_id, title=m.title, genre=m.genre) for m in results]

@app.post("/api/v1/movies/like")
def like_movie(like_request: LikeRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    movie = db.query(Movie).filter(Movie.movie_id == like_request.movie_id).first()
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")

    # Check if already liked
    existing = db.query(LikedMovie).filter(
        LikedMovie.user_id == current_user.user_id,
        LikedMovie.movie_id == like_request.movie_id
    ).first()
    if not existing:
        liked = LikedMovie(user_id=current_user.user_id, movie_id=like_request.movie_id)
        db.add(liked)
        db.commit()
    else :
        db.query(LikedMovie).filter(
            LikedMovie.user_id == current_user.user_id,
            LikedMovie.movie_id == like_request.movie_id
        ).delete()
        db.commit()
    return {"message": f"Movie {like_request.movie_id} liked"}

@app.get("/api/v1/search", response_model=List[MovieOut])
def search_movies(query: str, db: Session = Depends(get_db)):
    results = db.query(Movie).filter(Movie.title.ilike(f"%{query}%")).all()
    return [MovieOut(id=m.movie_id, title=m.title, genre=m.genre) for m in results]

@app.get("/api/v1/recommend", response_model=List[int])
def recommend_movies(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    seed=rd.randint(0,999)
    new_critics,liked_ids=find_similar_critics(db, current_user, rd_state=seed)
    recommended_ids=list(set(remove_infrequent_elements([c.movie_id for c in new_critics], int(1+4/len(liked_ids)))))
    result=recommended_ids[:20]+[seed]
    return result

@app.get("/api/v1/status")
def status():
    return "Everything is fine."

@app.post("/api/v1/delete_account")
def delete_account(del_u: DeleteUser, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if del_u.yes:
        liked = db.query(LikedMovie).filter(LikedMovie.user_id == current_user.user_id).all()
        user = db.query(User).filter(User.user_id == current_user.user_id).first()
        for e in liked:
            db.delete(e)
        db.delete(user)
        db.commit()
        return "User deleted"
    return "User not deleted"

@app.post("/api/v1/feedback")
def feedback(feedback: Feedback, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    new_critics,liked_ids=find_similar_critics(db, current_user, rd_state=feedback.seed)
    points_ids=list(set([e.critic_id for e in new_critics]))

    retrieved_points = qdrant_client.retrieve(
        collection_name=QDRANT_COLLECTION,
        ids=points_ids,
        with_payload=True,
        with_vectors=False
    )

    new_payloads=[]

    for p in retrieved_points:
        new_payloads.append(p.payload)

    for i in range(len(new_payloads)):
        new_payloads[i]["grade_count"]=new_payloads[i]["grade_count"]+1
        new_payloads[i]["embedding_quality_grade"]=(new_payloads[i]["embedding_quality_grade"]*(new_payloads[i]["grade_count"]-1)+feedback.grade)/new_payloads[i]["grade_count"]

    # print(len(new_payloads),len(points_ids))
    points_to_update = [models.SetPayloadOperation(set_payload=models.SetPayload(payload=new_payloads[i],points=[points_ids[i]])) for i in range(len(points_ids))]

    qdrant_client.batch_update_points(
        collection_name=QDRANT_COLLECTION,
        update_operations=points_to_update
    )

    return "Ok"
