import streamlit as st
import requests
import json
from dotenv import load_dotenv
import os
from uuid import uuid4
from streamlit_modal import Modal
from streamlit_star_rating import st_star_rating

# Config file
try:
    with open('config.json', 'r') as file:
        config_dict = json.load(file)
except FileNotFoundError:
    st.error("config.json not found. Please create one with 'API_URL'.")
    st.stop()

# -----------------------
# Backend config
# -----------------------
API_URL = config_dict["API_URL"]
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
# SECRET_KEY = os.getenv("SECRET_KEY")
# COOKIE_PREFIX = "webfloox-"
#
# if not SECRET_KEY:
#     st.error("SECRET_KEY not found in environment variables. Please set it for secure cookie management.")
#     st.stop()

# -----------------------
# Authentication functions
# -----------------------
def authenticate(username, password):
    try:
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        response = requests.post(f"{API_URL}/api/v1/login", data={"username": username, "password": password}, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            return {}
    except Exception as e:
        st.error(f"Erreur de connexion au serveur : {e}")
        return {}

def create_user(username, password):
    try:
        response = requests.post(
            f"{API_URL}/api/v1/register",
            json={"username": username, "password": password},
        )
        if response.status_code == 201:
            return True, "Compte créé avec succès 🎉"
        elif response.status_code == 400:
            return False, response.json().get("detail", "Erreur : informations invalides")
        else:
            return False, "Impossible de créer le compte."
    except Exception as e:
        return False, f"Erreur de connexion au serveur : {e}"

# -----------------------
# Page functions
# -----------------------
def login_page():
    st.title("🔑 Connexion")

    with st.form("login_form"):
        username = st.text_input("Nom d'utilisateur")
        password = st.text_input("Mot de passe", type="password")
        if st.form_submit_button("Se connecter"):
            auth_data = authenticate(username, password)
            if "access_token" in auth_data:
                st.session_state["page"] = "app"
                st.session_state["token"] = auth_data["access_token"]
                st.success("Connexion réussie ✅")
                st.rerun()
            else:
                st.error("Nom d'utilisateur ou mot de passe incorrect ❌")

    if st.button("Créer un compte"):
        st.session_state["page"] = "register"
        st.rerun()

def register_page():
    st.title("🆕 Créer un compte")
    with st.form("register_form"):
        username = st.text_input("Nom d'utilisateur")
        password = st.text_input("Mot de passe", type="password")
        confirm = st.text_input("Confirmer le mot de passe", type="password")
        submitted = st.form_submit_button("Créer le compte")
        if submitted:
            if password != confirm:
                st.error("Les mots de passe ne correspondent pas ❌")
            else:
                success, msg = create_user(username, password)
                if success:
                    st.success(msg)
                    st.session_state["page"] = "login"
                    st.rerun()
                else:
                    st.error(msg)
    st.text("Les données stockés sont : votre pseudonyme et vos préférences de films.")
    if st.button("Déjà un compte ? Se connecter"):
        st.session_state["page"] = "login"
        st.rerun()

def app_page():
    try : token = st.session_state["token"]
    except :
        st.session_state["page"] = "login"
        st.rerun()
    headers = {"Authorization": f"Bearer {token}"}

    st.title("Webfloox 🎬")
    # st.write("Cookies disponibles :", cookie_controller.getAll())
    if st.sidebar.button("🚪 Se déconnecter"):
        st.session_state.clear()
        st.session_state["page"] = "login"
        st.rerun()
    if st.sidebar.button("🗑️ Supprimer son compte"):
        modal.open()
    if modal.is_open():
        with modal.container():
            st.write("Voulez vraiment supprimer votre compte ?")
            if st.button("Oui"):
                requests.post(f"{API_URL}/api/v1/delete_account", json={"del_u":1}, headers=headers)
                st.session_state.clear()
                st.session_state["page"] = "login"
                st.rerun()
            if st.button("Non"):
                modal.close()


    search_query = st.text_input("Rechercher un film", "")

    cols_per_row = 5

    movies = []
    try:
        r = requests.get(f"{API_URL}/api/v1/recommend", headers=headers)
        if r.status_code == 200:
            st.session_state["seed"]=r.json()[-1]
            for movie_id in r.json()[:-1]:
                resp = requests.get(f"{TMDB_BASE_URL}/movie/{movie_id}?api_key={TMDB_API_KEY}")
                resp.raise_for_status()
                movies.append({"poster": f"https://image.tmdb.org/t/p/w300/{resp.json()['poster_path']}", "title": resp.json()["title"], "id": movie_id})
        else:
            st.warning("Impossible de récupérer les recommandations du backend.")
    except requests.exceptions.RequestException as e:
        st.warning(f"Impossible de contacter le backend ou l'API TMDB. Erreur: {e}")

    stars = st_star_rating(label = "Notez les recommandations", maxValue = 5, defaultValue = 0, key = "rating", dark_theme = True )
    if stars :
        resp = requests.post(f"{API_URL}/api/v1/feedback", json={"grade":stars,"ssed":st.session_state["seed"]}, headers=headers)
        resp.raise_for_status()

    if search_query:
        r = requests.get(f"{API_URL}/api/v1/search?query={search_query}", headers=headers)
        if r.status_code == 200:
            movies = []
            for movie in r.json():
                resp = requests.get(f"{TMDB_BASE_URL}/movie/{movie["id"]}?api_key={TMDB_API_KEY}")
                resp.raise_for_status()
                # print(resp.json())
                movies.append({"poster": f"https://image.tmdb.org/t/p/w300/{resp.json()['poster_path']}", "title": resp.json()["title"], "id": movie["id"]})
        else:
            st.warning("Impossible de récupérer les résultats du backend.")
    r = requests.get(f"{API_URL}/api/v1/movies_liked", headers=headers)
    if r.status_code == 200:
        liked_movies=r.json()
    else:
        st.warning("Impossible de récupérer les résultats du backend.")
    if st.sidebar.button("🏠 Acceuil"):
        st.session_state["liked_only"]="0"
        st.session_state["page"]="app"
    if st.sidebar.button("❤️ Films aimés"):
        st.session_state["liked_only"]="1"
    if st.session_state["liked_only"] == "1":
        movies = []
        for movie in r.json():
            resp = requests.get(f"{TMDB_BASE_URL}/movie/{movie["id"]}?api_key={TMDB_API_KEY}")
            resp.raise_for_status()
            # print(resp.json())
            movies.append({"poster": f"https://image.tmdb.org/t/p/w300/{resp.json()['poster_path']}", "title": resp.json()["title"], "id": movie["id"]})
    if movies:
        for i in range(0, len(movies), cols_per_row):
            cols = st.columns(cols_per_row)
            for col, movie in zip(cols, movies[i:i+cols_per_row]):
                with col:
                    st.image(movie["poster"], width='content')
                    st.markdown(f"**{movie['title']}**")
                    # st.markdown(f"{movie["id"], movie["title"]}") debug
                    if movie["id"] in [e["id"] for e in liked_movies]:
                        heart='💙'
                    else:
                        heart='❤️'
                    if st.button(heart, key=movie["id"]):
                        try:
                            resp = requests.post(f"{API_URL}/api/v1/movies/like", json={"movie_id": movie["id"]}, headers=headers)
                            resp.raise_for_status()
                            st.rerun()
                        except requests.exceptions.RequestException as e:
                            st.error(f"Erreur lors de l'envoi de la requête de 'like'. Erreur: {e}")
    else:
        st.info("Aucun film à afficher. Lancez une recherche ou connectez-vous pour voir les recommandations.")

# -----------------------
# Main App logic
# -----------------------
st.set_page_config(page_title="Webfloox", layout="wide", page_icon="🎬")

modal = Modal(
    "Suppression du compte",
    key="demo-modal",

    # Optional
    padding=20,    # default value
    max_width=744  # default value
)

if "page" not in st.session_state:
    st.session_state["page"] = "login"

if st.session_state["page"] == "login":
    login_page()
    st.session_state["liked_only"] = "0"
    st.session_state["seed"]=0

elif st.session_state["page"] == "register":
    register_page()
else:
    app_page()
