import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
from streamlit_lottie import st_lottie

# Page configuration
st.set_page_config(
    page_title="Movie Recommender",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS ---
st.markdown("""
    <style>
        /* Limit max width and center */
        .css-18e3th9 {  /* main container */
            max-width: 700px;
            margin: 0 auto;
            padding: 1rem 1.5rem;
        }

        /* Disable vertical scrolling */
        html, body, .main {
            height: 100vh;
            overflow: hidden;
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            color: white;
            font-family: 'Segoe UI', sans-serif;
        }

        /* Fade in animation for app */
        .stApp {
            animation: fadeIn 1.5s ease-in;
        }

        /* Header and text */
        h1, h2, h3, .stMarkdown {
            font-size: 1.3rem;
            color: #1DB954;
            text-align: center;
            margin-bottom: 0.2rem;
        }

        p {
            text-align: center;
            font-size: 1rem;
            margin-top: 0;
            margin-bottom: 1.5rem;
            color: #d0f0c0;
        }

        # Replace the existing selectbox styling in your CSS with this:

        /* Selectbox styling */
        div[data-baseweb="select"] > div {
            text-align: left !important;
            font-size: 1.1rem !important;
            font-weight: 500 !important;
            border-radius: 10px !important;
            padding: 8px 10px !important;
            background-color: #203a43 !important;
            color: white !important;
            overflow: visible !important;
            white-space: normal !important;
            text-overflow: unset !important;
        }
        
        /* Dropdown menu styling */
        div[role="listbox"] ul {
            background-color: #203a43 !important;
            color: white !important;
        }
        
        div[role="listbox"] li {
            padding: 10px !important;
            white-space: normal !important;
        }
        
        div[role="listbox"] li:hover {
            background-color: #1DB954 !important;
        }

        /* Button styling */
        button[kind="primary"] {
            background-color: #1DB954 !important;
            color: white !important;
            font-size: 1.1rem !important;
            padding: 10px 15px !important;
            border-radius: 10px !important;
            margin-top: 10px;
            width: 100%;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        button[kind="primary"]:hover {
            background-color: #17a84b !important;
        }

        /* Recommendations card */
        .recommendation-card {
            background-color: rgba(29, 185, 84, 0.15);
            border-radius: 12px;
            padding: 12px 16px;
            margin: 8px 0;
            font-weight: 600;
            color: #c4f0a4;
            box-shadow: 0 0 8px rgba(29,185,84,0.4);
            transition: background-color 0.3s ease;
        }
        .recommendation-card:hover {
            background-color: rgba(29, 185, 84, 0.3);
        }

        /* FadeIn animation */
        @keyframes fadeIn {
            0% { opacity: 0; transform: translateY(-10px); }
            100% { opacity: 1; transform: translateY(0); }
        }

        /* Footer */
        footer {
            margin-top: 40px;
            text-align: center;
            font-size: 14px;
            color: #a6d785;
        }
    </style>
""", unsafe_allow_html=True)

# Load animation
def load_lottie(url: str):
    try:
        r = requests.get(url, timeout=3)
        return r.json() if r.status_code == 200 else None
    except:
        return None

# Data loading
@st.cache_data
def load_movies():
    return pd.read_csv('ml-1m/movies.dat', sep='::', engine='python',
                     names=['MovieID', 'Title', 'Genres'],
                     encoding='ISO-8859-1')

# Content-based recommendations
@st.cache_data
def create_content_sim(movies):
    tfidf = TfidfVectorizer(token_pattern='[a-zA-Z0-9-]+')
    tfidf_matrix = tfidf.fit_transform(movies['Genres'])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_recommendations(selected_title, movies, content_sim, indices):
    idx = indices[selected_title]
    sim_scores = list(enumerate(content_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    return [movies.iloc[i[0]]['Title'] for i in sim_scores[1:6]]

def main():
    st.title("🎬 Movie Recommender")
    st.markdown("Discover films you'll love based on genres")
    
    # Animation
    lottie = load_lottie("https://assets4.lottiefiles.com/packages/lf20_q6qezkqk.json")
    if lottie:
        st_lottie(lottie, height=150, key="movieAnim")
    
    # Load data
    movies = load_movies()
    content_sim = create_content_sim(movies)
    indices = pd.Series(movies.index, index=movies['Title']).drop_duplicates()
    
    # User input
    selected_movie = st.selectbox(
        "Choose a movie you like:", 
        movies['Title'].tolist()
    )
    
    if st.button("Get Recommendations"):
        with st.spinner("Finding similar movies..."):
            recommendations = get_recommendations(
                selected_title=selected_movie,
                movies=movies,
                content_sim=content_sim,
                indices=indices
            )
            
            st.subheader("Recommended For You")
            for i, title in enumerate(recommendations, 1):
                st.markdown(f"{i}. 🎬 **{title}**")

if __name__ == "__main__":
    main()