import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import MovieCritic, Movie
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import dotenv_values,load_dotenv
import numpy as np

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

QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = int(os.getenv("QDRANT_PORT"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")

# ========================
# DB connection
# ========================
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
session = SessionLocal()

# ========================
# Qdrant setup
# ========================
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# ========================
# Load embedding model
# ========================
print("🔹 Loading embedding model (all-MiniLM-L6-v2)...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Ensure collection exists
client.recreate_collection(
    collection_name=QDRANT_COLLECTION,
    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
)
print(len(model.encode("hello world")))
# ========================
# Embedding + insert
# ========================
def embed_and_insert_reviews(batch_size=50):
    reviews = session.query(MovieCritic).all()
    # Debug print 1: Check how many reviews were retrieved.
    print(f"Found {len(reviews)} reviews to embed.")

    points = []
    for i, review in enumerate(reviews, 1):
        movie = session.query(Movie).filter(Movie.movie_id == review.movie_id).first()

        if movie.genre is None: genre = ""
        else : genre = movie.genre
        # if movie.synopsis is None: embedding3 = [.0]*384
        # else : embedding3 = model.encode(movie.synopsis)

        embedding = model.encode(genre + " " + review.review_text).tolist()
        # Debug print 2: Check if each embedding is being generated and appended.
        # print(f"Embedding review {i} with vector size: {len(embedding)}")

        payload = {
            "movie_id": review.movie_id,
            "critic_id": review.critic_id,
            "embedding_quality_grade": 1,
            "grade_count": 0
        }

        point = models.PointStruct(
            id=review.critic_id,
            vector=embedding,
            payload=payload,
        )
        # print(point)
        points.append(point)

        if len(points) >= batch_size:
            # Debug print 3: Check batch size before upsert.
            print(f"Upserting batch of {len(points)} points...")
            client.upsert(collection_name=QDRANT_COLLECTION, points=points)
            print(f"Inserted {i} reviews so far...")
            points = []

    if points:
        # Debug print 4: Check remaining points before final upsert.
        print(f"Upserting remaining {len(points)} points...")
        client.upsert(collection_name=QDRANT_COLLECTION, points=points)

    print("✅ All reviews embedded and inserted into Qdrant!")

# ========================
# Run
# ========================
if __name__ == "__main__":
    embed_and_insert_reviews()
