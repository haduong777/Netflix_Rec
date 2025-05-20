import numpy as np
import pandas as pd
import os
from typing import List, Dict, Tuple, Optional, Callable
from scipy.sparse import csr_matrix
from tqdm import tqdm

def init_factors(num_users: int, num_movies: int,
                 num_factors: int,
                 scale: float = 0.01):
  """
  Initialize user and movie latent factors (small non zeroes)

  Args
    num_users: number of users
    num_movies: number of movies
    num_factors: number of latent factors

  Returns
    (U, V): user and movie latent factors
  """

  U = np.random.rand(num_users, num_factors) * scale
  V = np.random.rand(num_movies, num_factors) * scale

  return U, V

def solve_user_factors(ratings: csr_matrix, movie_factors: np.ndarray,
                       lambda_reg: float = 0.01):
  """
  Solve for user latent factors, fixed movie factors

  Args
    ratings: sparse matrix of user-movie ratings
    movie_factors: latent factors for movies
    lambda_reg: regularization parameter

  Returns
    user_factors: array of user latent factors
  """

  num_users, num_factors = ratings.shape[0], movie_factors.shape[1]
  user_factors = np.zeros((num_users, num_factors))

  reg = lambda_reg * np.eye(num_factors)

  for user in range(num_users):
    movie_idx = ratings[user].indices

    if len(movie_idx) == 0:  # users w no ratings
      continue

    user_ratings = ratings[user].data
    V = movie_factors[movie_idx]

    A = V.T @ V + reg
    b = V.T @ user_ratings

    try:
      user_factors[user] = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
      # fall back to singular value decomposition when matrix is singular
      user_factors[user] = np.linalg.lstsq(A, b, rcond=None)[0]

  return user_factors

def solve_movie_factors(ratings: csr_matrix, user_factors: np.ndarray,
                       lambda_reg: float = 0.01):
  """
  Solve for movie latent factors, fixed user factors

  Args
    ratings: sparse matrix of user-movie ratings (transposed)
    movie_factors: latent factors for movies
    lambda_reg: regularization parameter

  Returns
    movie_factors: array of movies latent factors
  """

  num_movies, num_factors = ratings.shape[0], user_factors.shape[1]
  movie_factors = np.zeros((num_movies, num_factors))

  reg = lambda_reg * np.eye(num_factors)

  for movie in range(num_movies):
    # idx of users who rated this movie
    user_idx = ratings[movie].indices

    if len(user_idx) == 0:  # movies w no ratings
      continue

    user_ratings = ratings[movie].data
    U = user_factors[user_idx]

    A = U.T @ U + reg
    b = U.T @ user_ratings

    try:
      user_factors[movie] = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
      # fall back to singular value decomposition when matrix is singular
      user_factors[movie] = np.linalg.lstsq(A, b, rcond=None)[0]

  return user_factors

def sparse_matrix(partitions: List[Dict],
                  user_map: Dict[int, int],
                  movie_map: Dict[int, int],
                  columns: List[str] = ['userId', 'movieId', 'rating']):
  """
  Builds sparse matrix from partitions

  Args
    partitions: List of partition files metadata
    user_map, movie_id: Mapping from original to dense user/movie id
    columns: List of columns to include in sparse matrix

  Returns
    sparse matrix of ratings (csr_matrix)
  """

  num_users = len(user_map)
  num_movies = len(movie_map)

  rows, cols, data = [], [], []

  for partition in tqdm(partitions, desc="Building sparse matrix..."):
    df = pd.read_parquet(partition['path'], columns=columns)

    # map user_id, movie_id -> dense indices
    df_users = [user_map.get(u, -1) for u in df['user_id']]
    df_movies = [movie_map.get(m, -1) for m in df['movie_id']]

    valid_idx = [(i,u,m) for i, (u, m) in enumerate(zip(df_users, df_movies))
                 if u != -1 and m != -1]

    if not valid_idx:
      print(f"WARNING: Skipping partition {partition['path']} due to no valid mappings found")
      continue

    # Append to COO format data
    idx, mapped_users, mapped_movies = zip(*valid_idx)
    rows.extend(mapped_users)
    cols.extend(mapped_movies)

    data.extend(df['rating'].iloc[list(idx)].values)

  # construct sparse matrix in CSR format
  return csr_matrix((data, (rows, cols)), shape=(num_users, num_movies))

def save_checkpoint(checkpoint_dir: str, iteration: int, U: np.ndarray, V: np.ndarray):
  """
  Args
    checkpoint_dir: Directory to save checkpoints
    iteration: Current iteration
    U, V: User and movie latent matrices
  """
  os.makedirs(checkpoint_dir, exist_ok=True)

  checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{iteration}.npz")

  np.savez_compressed(checkpoint_path, U=U, V=V, iteration=iteration)

  print(f"Saved checkpoint to {checkpoint_path}")

def load_checkpoint(checkpoint_path: str):
  """
  Args
    checkpoint_path: Path to checkpoint file

  Returns
    U, V: User and movie latent matrices
  """
  print(f"Loading checkpoint from {checkpoint_path}")

  checkpoint = np.load(checkpoint_path)

  return {
      'U': checkpoint['U'],
      'V': checkpoint['V'],
      'iteration': checkpoint['iteration']
  }

def calculate_rmse(ratings: csr_matrix, U: np.ndarray, V: np.ndarray):
  """
  Args
    ratings: sparse matrix of user-movie ratings
    U, V: User and movie latent matrices

  Returns
    RMSE
  """
  #### COULD CHANGE SAMPLE SIZE
  sample_size = min(100000,ratings.nnz)
  sample_idx = np.random.choice(ratings.nnz, size=sample_size, replace=False)

  rows, cols = ratings.nonzero()
  row_sample = rows[sample_idx]
  col_sample = cols[sample_idx]

  ratings_sample = np.array([ratings[i, j] for i, j in zip(row_sample, col_sample)])

  predictions = np.sum(U[row_sample] * V[col_sample], axis=1)

  rmse = np.sqrt(np.mean((predictions - ratings_sample) ** 2))

  return rmse


def als_mf(partitions: List[Dict],
           user_map: Dict[int, int],
           movie_map: Dict[int, int],
           num_factors: int=100,
           lambda_reg: float=0.1,
           num_iters: int=20,
           checkpoint_dir: Optional[str]=None,
           val_interval: int=10):
  """
  Alternating Least Squares Matrix Factorization

  Args
    partitions: List of partition files metadata
    user_map, movie_id: Mapping from original to dense user/movie id
    num_factors: Number of latent factors
    lambda_reg: Regularization
    num_iters: Number of iterations
    checkpoint_dir: Directory to save checkpoints

  Returns
    U, V: User and movie latent matrices
  """

  num_users = len(user_map)
  num_movies = len(movie_map)

  print(f"Building sparse matrices for {num_users} users, and {num_movies} movies")

  ratings = sparse_matrix(partitions, user_map, movie_map)
  ratings_t = ratings.transpose().tocsr()

  # Optimization
  U, V = init_factors(num_users, num_movies, num_factors)

  for iter in tqdm(range(1, num_iters + 1), desc="ALS Optimization"):

    # Update user latent factors
    U = solve_user_factors(ratings, V, lambda_reg)

    # Update movie latent factors
    V = solve_movie_factors(ratings_t, U, lambda_reg)

    if checkpoint_dir:
      save_checkpoint(checkpoint_dir, iteration, U, V)

    if iter % val_interval == 0:
      rmse = calculate_rmse(ratings, U, V)
      print(f"Iteration {iter} -- RMSE: {rmse:.4f}")

  return U, V

def als_predict(user_id: int, movie_id: int,
                ratings: csr_matrix, U: np.ndarray, V: np.ndarray,
                user_map: Dict[int, int], movie_map: Dict[int, int]):
  """
  Singular predictions

  Args
    ratings: sparse matrix of user-movie ratings
    U, V: User and movie latent matrices
    user_map, movie_map: Mapping from original to dense user/movie id

  Returns
    1 prediction
  """

  dense_user_id = user_map.get(user_id)
  dense_movie_id = movie_map.get(movie_id)

  # if the user and movie doesn't yet exist in the model --> use average
  if dense_user_id is None or dense_movie_id is None:
    print(f"Cold start for user {user_id} and movie {movie_id}, returning default prediction 3.0 (average rating)")
    return 3.0

  # otherwise make prediction
  prediction = float(np.dot(U[dense_user_id], V[dense_movie_id]))

  return prediction


def als_batch_predict(data: pd.DataFrame, U: np.ndarray, V: np.ndarray,
                      user_map: Dict[int, int], movie_map: Dict[int, int]):
  """
  Make batch predictions from DataFrame without ratings

  Args
    data: DataFrame of user-movie pairs
    U, V: User and movie latent matrices
    user_map, movie_map: Mapping from original to dense user/movie id

  Returns
    Array of predictions
  """

  dense_user_ids = [user_map.get(u, -1) for u in data['user_id']]
  dense_movie_ids = [movie_map.get(m, -1) for m in data['movie_id']]

  predictions = np.full(len(data), 3.0)

  valid_idx = [(i,u,m) for i, (u, m) in enumerate(zip(dense_user_ids, dense_movie_ids))
                if u != -1 and m != -1]

  if not valid_idx:
    print(f"WARNING: Batch prediction failed due to no valid mappings found, returning default prediction = 3.0")
    return predictions

  valid_users = [dense_user_ids[i] for i in valid_idx]
  valid_movies = [dense_movie_ids[i] for i in valid_idx]

  # make all predictions at once
  valid_predictions = np.sum(U[valid_users] * V[valid_movies], axis=1)

  # convert to array
  for i, pred in zip(valid_idx, valid_predictions):
    predictions[i[0]] = pred

  return predictions

