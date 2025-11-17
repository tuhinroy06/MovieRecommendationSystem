#!/usr/bin/env python3
"""
movie_recommender.py

Simple Movie Recommendation System
- Content-based: TF-IDF on movie titles + genres -> cosine similarity
- Item-based Collaborative Filtering: cosine similarity on item-user ratings matrix

Dataset:
- Uses MovieLens "ml-latest-small" (automatically downloads & extracts)

Usage examples:
$ python movie_recommender.py --method content --title "Toy Story (1995)" --topn 10
$ python movie_recommender.py --method collab --user 1 --topn 10

Dependencies:
pip install pandas numpy scipy scikit-learn requests tqdm

Notes:
- Collaborative method here is item-based: it computes item-item similarity and
  ranks unseen movies by estimated score (weighted sum of user's rated items).
- Content-based uses TF-IDF on "title + genres" text.
"""
import argparse
import os
import zipfile
import io
import requests
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_ZIP = "ml-latest-small.zip"
DATA_DIR = "ml-latest-small"


def download_movielens(target_dir=DATA_DIR):
    """Download and extract MovieLens ml-latest-small if not present."""
    if os.path.isdir(target_dir):
        print(f"Found existing dataset directory: {target_dir}")
        return target_dir

    print("Downloading MovieLens ml-latest-small dataset...")
    r = requests.get(DATA_URL, stream=True)
    r.raise_for_status()
    total = int(r.headers.get('content-length', 0))
    with open(DATA_ZIP, 'wb') as f:
        for chunk in tqdm(r.iter_content(chunk_size=8192), total=(total // 8192) + 1, unit='KB'):
            if chunk:
                f.write(chunk)
    print("Extracting...")
    with zipfile.ZipFile(DATA_ZIP, 'r') as z:
        z.extractall()
    os.remove(DATA_ZIP)
    print(f"Dataset downloaded and extracted to ./{target_dir}")
    return target_dir


def load_data(data_dir=DATA_DIR):
    """Load movies and ratings into pandas DataFrames."""
    movies_path = os.path.join(data_dir, "movies.csv")
    ratings_path = os.path.join(data_dir, "ratings.csv")
    if not (os.path.exists(movies_path) and os.path.exists(ratings_path)):
        download_movielens(data_dir)
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    # Basic cleaning: ensure movieId is int, title string, genres string
    movies['title'] = movies['title'].astype(str)
    movies['genres'] = movies['genres'].astype(str)
    return movies, ratings


# -------- Content-based model --------
def build_content_model(movies, max_features=5000):
    """
    Build TF-IDF matrix from combined title + genres.
    Returns: tfidf matrix (sparse), vectorizer, index->movieId map, movieId->index map
    """
    # Create a simple text field: title + " " + genres (replace | with space)
    texts = (movies['title'].fillna('') + " " + movies['genres'].fillna('').str.replace('|', ' '))
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
    tfidf = vectorizer.fit_transform(texts.values)
    idx_to_movieid = movies['movieId'].reset_index(drop=True)
    movieid_to_idx = {mid: idx for idx, mid in enumerate(idx_to_movieid)}
    return tfidf, vectorizer, idx_to_movieid, movieid_to_idx


def recommend_content(movies, tfidf, idx_to_movieid, movieid_to_idx, title_query, topn=10):
    """
    Recommend movies similar to a movie title (content-based).
    title_query: exact title string or partial; function finds best match by title substring.
    """
    # find movie by exact or partial match (case-insensitive)
    matches = movies[movies['title'].str.contains(title_query, case=False, na=False)]
    if matches.empty:
        # fallback: exact lower-case match
        matches = movies[movies['title'].str.lower() == title_query.lower()]

    if matches.empty:
        raise ValueError(f"No movie matching '{title_query}' found in the dataset.")

    # pick the first match
    movie_row = matches.iloc[0]
    movie_id = movie_row['movieId']
    idx = movieid_to_idx[movie_id]

    # cosine similarity between this movie and all others
    sim_scores = cosine_similarity(tfidf[idx], tfidf).flatten()
    # exclude the movie itself
    sim_scores[idx] = -1

    top_idx = np.argsort(sim_scores)[::-1][:topn]
    recommendations = []
    for i in top_idx:
        recommendations.append({
            'movieId': int(idx_to_movieid.iloc[i]),
            'title': movies.iloc[i]['title'],
            'genres': movies.iloc[i]['genres'],
            'score': float(sim_scores[i])
        })
    return recommendations


# -------- Item-based collaborative filtering --------
def build_item_user_matrix(movies, ratings):
    """
    Build item-user matrix with rows = movieId idx, columns = userId idx
    Returns sparse csr_matrix, movieid<->idx maps, userid<->idx maps
    """
    # create index maps
    unique_movieids = ratings['movieId'].unique()
    unique_userids = ratings['userId'].unique()
    movieid_to_idx = {mid: i for i, mid in enumerate(unique_movieids)}
    idx_to_movieid = {i: mid for mid, i in movieid_to_idx.items()}
    userid_to_idx = {uid: i for i, uid in enumerate(unique_userids)}
    idx_to_userid = {i: uid for uid, i in userid_to_idx.items()}

    # build sparse matrix
    rows = ratings['movieId'].map(movieid_to_idx)
    cols = ratings['userId'].map(userid_to_idx)
    data = ratings['rating']
    mat = csr_matrix((data, (rows, cols)), shape=(len(unique_movieids), len(unique_userids)))
    return mat, idx_to_movieid, movieid_to_idx, idx_to_userid, userid_to_idx


def compute_item_similarity(item_user_matrix):
    """
    Compute item-item cosine similarity (dense). For ml-latest-small it's ok.
    Returns a dense numpy array similarity matrix (n_items x n_items).
    """
    # Normalize rows to unit length to compute cosine via dot-product
    # But sklearn's cosine_similarity handles sparse; use it.
    print("Computing item-item similarity matrix (this may take a moment)...")
    sim = cosine_similarity(item_user_matrix, dense_output=True)
    return sim


def recommend_collab(movies, ratings, item_user_matrix, idx_to_movieid, movieid_to_idx,
                    idx_to_userid, userid_to_idx, item_similarity, user_id, topn=10):
    """
    Item-based CF recommendation for a given user_id.
    Approach:
    - For items the user has rated, look up similar items and compute weighted score:
        score[j] = sum_over_i ( similarity[j,i] * rating_i )
      then normalize by sum of abs(similarity).
    - Exclude items the user has already rated.
    """
    if user_id not in userid_to_idx:
        raise ValueError(f"Unknown user id: {user_id}")

    uidx = userid_to_idx[user_id]
    # get user's ratings: from sparse matrix columns
    user_ratings = item_user_matrix[:, uidx].toarray().flatten()  # shape (n_items,)

    rated_mask = user_ratings > 0
    if not rated_mask.any():
        raise ValueError(f"User {user_id} has no ratings in the dataset; cannot recommend.")

    # Compute estimated scores for all items: weighted sum over user's rated items
    # item_similarity is (n_items x n_items)
    # We'll compute: scores = item_similarity[:, rated_items] dot user_ratings[rated_items]
    rated_indices = np.where(rated_mask)[0]
    sim_sub = item_similarity[:, rated_indices]  # shape (n_items, n_rated)
    weights = user_ratings[rated_indices]       # shape (n_rated,)

    raw_scores = sim_sub.dot(weights)  # shape (n_items,)
    denom = np.abs(sim_sub).sum(axis=1)  # sum of abs similarities per item
    # avoid division by zero
    denom[denom == 0] = 1e-8
    scores = raw_scores / denom

    # Do not recommend items user already rated
    scores[rated_indices] = -np.inf

    top_idx = np.argsort(scores)[::-1][:topn]
    recommendations = []
    for i in top_idx:
        mid = int(idx_to_movieid[i])
        title = movies[movies['movieId'] == mid]['title'].values
        title = title[0] if len(title) > 0 else "Unknown"
        recommendations.append({
            'movieId': mid,
            'title': title,
            'estimated_score': float(scores[i])
        })
    return recommendations


# -------- CLI and main flow --------
def main():
    parser = argparse.ArgumentParser(description="Simple Movie Recommendation System")
    parser.add_argument('--method', choices=['content', 'collab'], required=True,
                        help="Recommendation method: 'content' or 'collab'")
    parser.add_argument('--title', type=str, default=None, help="Movie title (for content-based)")
    parser.add_argument('--user', type=int, default=None, help="User ID (for collaborative)")
    parser.add_argument('--topn', type=int, default=10, help="Number of recommendations to return")
    args = parser.parse_args()

    movies, ratings = load_data()

    if args.method == 'content':
        if not args.title:
            parser.error("Content-based method requires --title")
        print("Building content model...")
        tfidf, vectorizer, idx_to_movieid, movieid_to_idx = build_content_model(movies)
        try:
            recs = recommend_content(movies, tfidf, idx_to_movieid, movieid_to_idx, args.title, topn=args.topn)
        except ValueError as e:
            print("Error:", e)
            return
        print(f"\nContent-based recommendations for '{args.title}' (top {args.topn}):\n")
        for i, r in enumerate(recs, start=1):
            print(f"{i:2d}. {r['title']}  (genres: {r['genres']})  score={r['score']:.4f}")

    else:  # collab
        if args.user is None:
            parser.error("Collaborative method requires --user")
        print("Building item-user matrix...")
        item_user_matrix, idx_to_movieid, movieid_to_idx, idx_to_userid, userid_to_idx = build_item_user_matrix(movies, ratings)
        item_similarity = compute_item_similarity(item_user_matrix)
        try:
            recs = recommend_collab(movies, ratings, item_user_matrix, idx_to_movieid, movieid_to_idx,
                                   idx_to_userid, userid_to_idx, item_similarity, args.user, topn=args.topn)
        except ValueError as e:
            print("Error:", e)
            return
        print(f"\nItem-based collaborative recommendations for user {args.user} (top {args.topn}):\n")
        for i, r in enumerate(recs, start=1):
            print(f"{i:2d}. {r['title']}  estimated_score={r['estimated_score']:.4f}")


if __name__ == "__main__":
    main()
