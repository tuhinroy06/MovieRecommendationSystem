# Movie Recommendation System

This project includes:
- Content-based recommender (TF-IDF + cosine similarity)
- Collaborative filtering recommender (item-based)

## Usage
```bash
python movie_recommender.py --method content --title "Toy Story (1995)" --topn 10
python movie_recommender.py --method collab --user 1 --topn 10
```

## Requirements
Install dependencies with:

```bash
pip install pandas numpy scipy scikit-learn requests tqdm
```
