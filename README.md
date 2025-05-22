# Movie-Recommendation-System-ElevateLabs-Internship-Project
A **Movie Recommendation System** built using **Python**, **Pandas**, **Scikit-learn**, and **Streamlit**. This hybrid model combines **collaborative filtering** (user-based KNN) and **content-based filtering** (genre similarity) to provide personalized movie suggestions.

---

## 📌 Project Objective

To recommend top 5 movies based on a user's selected movie preferences using machine learning and data analysis techniques.

---

## 📂 Dataset

- **Source:** [MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1m/)
- **Files Used:**
  - `movies.dat`: Contains movie titles and genres
  - `ratings.dat`: Contains user ratings for movies

---

## 🛠️ Tools & Technologies

- Python 3
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- SciPy

---

## 🚀 Features

- 📊 **Collaborative Filtering** using KNN (User-Item matrix)
- 🎭 **Content-Based Filtering** using Genre similarity (TF-IDF + Cosine Similarity)
- 🎯 **Hybrid Recommendation System**: Combines both techniques for better accuracy
- 🖥️ **Interactive Streamlit UI** for live recommendations
- 📥 Returns **Top 5 movie recommendations** based on user selection

---

## 🧰 Installation

1. **Clone this repository:**
   ```bash
   git clone https://github.com/yourusername/movie-recommender
   cd movie-recommender

## Download the MovieLens 1M dataset:
- Go to https://grouplens.org/datasets/movielens/1m/
- Extract and place movies.dat and ratings.dat in the root directory.

## Run the Streamlit App:
- streamlit run app.py

