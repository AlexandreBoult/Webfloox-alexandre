from sqlalchemy import (
    create_engine, Column, Integer, String, Text, Float, DateTime, ForeignKey
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
import os
from dotenv import dotenv_values,load_dotenv

# =========================
# Configuration DB
# =========================

load_dotenv()
DB_USER=os.getenv("POSTGRES_USER")
DB_PASSWORD=os.getenv("POSTGRES_PASSWORD")
DB_HOST=os.getenv("POSTGRES_HOST")
DB_PORT=os.getenv("POSTGRES_PORT")
DB_NAME=os.getenv("POSTGRES_DB")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# =========================
# Tables
# =========================
class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False)
    # email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    salt = Column(String(255), nullable=False)

    liked_movies = relationship("LikedMovie", back_populates="user")
    # critics = relationship("MovieCritic", back_populates="user")


class Movie(Base):
    __tablename__ = "movies"

    movie_id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    year = Column(Integer, nullable=True)
    synopsis = Column(Text, nullable=True)
    genre = Column(String(150), nullable=True)

    liked_by = relationship("LikedMovie", back_populates="movie")
    # embeddings = relationship("MovieEmbedding", back_populates="movie")
    critics = relationship("MovieCritic", back_populates="movie")


class LikedMovie(Base):
    __tablename__ = "liked_movies"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    movie_id = Column(Integer, ForeignKey("movies.movie_id"), nullable=False)
    liked_date = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="liked_movies")
    movie = relationship("Movie", back_populates="liked_by")


class MovieCritic(Base):
    __tablename__ = "movie_critics"

    critic_id = Column(Integer, primary_key=True, index=True)
    # user_id = Column(Integer, ForeignKey("users.user_id"), nullable=True)  # facultatif si critique externe
    movie_id = Column(Integer, ForeignKey("movies.movie_id"), nullable=False)
    critic_title = Column(String(255), nullable=True)
    review_text = Column(Text, nullable=False)
    rating = Column(Float, nullable=True)
    source = Column(String(255), nullable=True)

    # user = relationship("User", back_populates="critics")
    movie = relationship("Movie", back_populates="critics")


# =========================
# Création de la BDD
# =========================
def create_db():
    Base.metadata.create_all(bind=engine)
    print(f"Database and tables created successfully at {DB_HOST}")


if __name__ == "__main__":
    create_db()
