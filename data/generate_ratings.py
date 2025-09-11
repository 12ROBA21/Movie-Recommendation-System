import argparse, numpy as np, pandas as pd, random, os

GENRES = ["Action","Adventure","Animation","Comedy","Crime","Documentary","Drama","Fantasy","Horror","Mystery","Romance","Sci-Fi","Thriller","War","Western"]

def make_movies(n_movies: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    movies = []
    for mid in range(1, n_movies+1):
        title = f"Movie {mid:04d}"
        n_g = rng.integers(1, 4)
        genres = ",".join(sorted(rng.choice(GENRES, size=n_g, replace=False).tolist()))
        movies.append([mid, title, genres])
    return pd.DataFrame(movies, columns=["movie_id","title","genres"])

def make_ratings(n_users: int, movies: pd.DataFrame, density: float = 0.06, seed: int = 42):
    rng = np.random.default_rng(seed)
    k = 8
    U = rng.normal(0, 1, size=(n_users, k))
    V = rng.normal(0, 1, size=(len(movies), k))
    genre_to_vec = {g: rng.normal(0, 0.6, size=k) for g in set(",".join(movies.genres.tolist()).split(","))}
    item_bias = np.zeros(len(movies))
    for i, gs in enumerate(movies.genres):
        gvec = np.mean([genre_to_vec[g] for g in gs.split(",")], axis=0)
        V[i] += gvec * 0.7
        item_bias[i] = rng.normal(0, 0.3)

    rows = []
    for uid in range(1, n_users+1):
        u = U[uid-1]
        n_r = max(5, int(density * len(movies) + rng.integers(-10, 10)))
        items = rng.choice(len(movies), size=min(len(movies), n_r), replace=False)
        for i in items:
            score = u @ V[i] + item_bias[i] + rng.normal(0, 1.0)
            prob_pos = 1 / (1 + np.exp(-score/2))
            rating = 1 + int(np.clip(np.round(prob_pos*4), 0, 4))
            rows.append([uid, int(movies.movie_id.iloc[i]), rating])
    df = pd.DataFrame(rows, columns=["user_id","movie_id","rating"])
    df = df.groupby(["user_id","movie_id"], as_index=False)["rating"].mean()
    df["rating"] = df["rating"].round().astype(int)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=800)
    ap.add_argument("--movies", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="data")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    movies = make_movies(args.movies, args.seed)
    ratings = make_ratings(args.users, movies, seed=args.seed)

    movies.to_csv(os.path.join(args.outdir, "movies.csv"), index=False)
    ratings.to_csv(os.path.join(args.outdir, "ratings.csv"), index=False)
    print(f"[OK] wrote {len(movies)} movies to {args.outdir}/movies.csv")
    print(f"[OK] wrote {len(ratings)} ratings to {args.outdir}/ratings.csv")

if __name__ == "__main__":
    main()
