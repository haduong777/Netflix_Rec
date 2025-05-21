import numpy as np
import pandas as pd
import os
from typing import List, Dict, Tuple, Optional, Callable
from scipy.sparse import csr_matrix
from tqdm import tqdm
import time
import json

class ALS:
  def __init__(self, 
              num_factors: int = 100, 
              lambda_reg: float = 0.1,
              num_iters: int = 20,
              val_interval: int = 10,
              checkpoint_interval: int=5,
              default_prediction: float = 3.0):
  
    self.num_factors = num_factors
    self.lambda_reg = lambda_reg
    self.num_iters = num_iters
    self.val_interval = val_interval
    self.checkpoint_interval = checkpoint_interval
    self.default_prediction = default_prediction

    self.user_map = None
    self.movie_map = None
    self.U = None
    self.V = None

  def fit(self, partitions: List[Dict],
          user_map: Dict[int, int],
          movie_map: Dict[int, int],
          checkpoint_dir: Optional[str]=None):
    """
    Train model

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
    self.user_map = user_map
    self.movie_map = movie_map

    num_users = len(user_map)
    num_movies = len(movie_map)

    print(f"Building sparse matrices for {num_users} users, and {num_movies} movies")

    ratings = self._sparse_matrix(partitions)
    ratings_t = ratings.transpose().tocsr()

    # Optimization
    self.U, self.V = self._init_factors(num_users, num_movies)

    for iter in tqdm(range(1, self.num_iters + 1), desc="ALS Optimization"):

      # Update user latent factors
      self.U = self._solve_user_factors(ratings)

      # Update movie latent factors
      self.V = self._solve_movie_factors(ratings_t)

      if iter % self.checkpoint_interval == 0:   
        if checkpoint_dir:
          self.save_checkpoint(checkpoint_dir, iter)

      if iter % self.val_interval == 0:
        rmse = self._calculate_rmse(ratings)
        print(f"\nIteration {iter} -- RMSE: {rmse:.4f}")

    return self.U, self.V

  def predict(self, user_id: int, movie_id: int):
    """
    Singular predictions

    Args
      ratings: sparse matrix of user-movie ratings
      U, V: User and movie latent matrices
      user_map, movie_map: Mapping from original to dense user/movie id

    Returns
      1 prediction
    """

    if self.U is None or self.V is None:
      raise ValueError("Model hasn't been trained, call fit() first")

    dense_user_id = self.user_map.get(user_id)
    dense_movie_id = self.movie_map.get(movie_id)

    # if the user and movie doesn't yet exist in the model --> use average
    if dense_user_id is None or dense_movie_id is None:
      print(f"Cold start for user {user_id} and movie {movie_id}, returning default prediction 3.0 (average rating)")
      return 3.0

    # otherwise make prediction
    prediction = float(np.dot(self.U[dense_user_id], self.V[dense_movie_id]))

    return prediction


  def batch_predict(self, data: pd.DataFrame):
    """
    Make batch predictions from DataFrame without ratings

    Args
      data: DataFrame of user-movie pairs
      U, V: User and movie latent matrices
      user_map, movie_map: Mapping from original to dense user/movie id

    Returns
      Array of predictions
    """

    dense_user_ids = [self.user_map.get(u, -1) for u in data['user_id']]
    dense_movie_ids = [self.movie_map.get(m, -1) for m in data['movie_id']]

    predictions = np.full(len(data), 3.0)

    valid_indices = [(i,u,m) for i, (u, m) in enumerate(zip(dense_user_ids, dense_movie_ids))
                  if u != -1 and m != -1]

    if not valid_indices:
      print(f"WARNING: Batch prediction failed due to no valid mappings found, returning default prediction = 3.0")
      return predictions

    valid_idx = [i for i, _, _ in valid_indices]
    valid_users = [u for _, u, _ in valid_indices]
    valid_movies = [v for _, _, v in valid_indices]

    # make all predictions at once
    valid_predictions = np.sum(self.U[valid_users] * self.V[valid_movies], axis=1)

    # convert to array
    for i, pred in zip(valid_idx, valid_predictions):
      predictions[i] = pred

    return predictions
  




  # ----------- INTERNAL METHODS --------------
    
  def _init_factors(self, num_users: int, num_movies: int):
    """
    Initialize user and movie latent factors (small non zeroes)
    """
    scale: float = 0.01

    U = np.random.rand(num_users, self.num_factors) * scale
    V = np.random.rand(num_movies, self.num_factors) * scale

    return U, V

  def _solve_user_factors(self, ratings: csr_matrix):
    """
    Solve for user latent factors, fixed movie factors

    Args
      ratings: sparse matrix of user-movie ratings

    Returns
      user_factors: array of user latent factors
    """

    num_users = ratings.shape[0]
    user_factors = np.zeros((num_users, self.num_factors))

    reg = self.lambda_reg * np.eye(self.num_factors)

    for user in range(num_users):
      movie_idx = ratings[user].indices

      if len(movie_idx) == 0:  # users w no ratings
        continue

      user_ratings = ratings[user].data
      V_m = self.V[movie_idx]

      A = V_m.T @ V_m + reg
      b = V_m.T @ user_ratings

      try:
        user_factors[user] = np.linalg.solve(A, b)
      except np.linalg.LinAlgError:
        # fall back to singular value decomposition when matrix is singular
        user_factors[user] = np.linalg.lstsq(A, b, rcond=None)[0]

    return user_factors

  def _solve_movie_factors(self, ratings_t: csr_matrix):
    """
    Solve for movie latent factors, fixed user factors

    Args
      ratings_t: sparse matrix of user-movie ratings (transposed)

    Returns
      movie_factors: array of movies latent factors
    """

    num_movies= ratings_t.shape[0]
    movie_factors = np.zeros((num_movies, self.num_factors))

    reg = self.lambda_reg * np.eye(self.num_factors)

    for movie in range(num_movies):
      # idx of users who rated this movie
      user_idx = ratings_t[movie].indices

      if len(user_idx) == 0:  # movies w no ratings
        continue

      movie_ratings = ratings_t[movie].data
      U_m = self.U[user_idx]

      A = U_m.T @ U_m + reg
      b = U_m.T @ movie_ratings

      try:
        movie_factors[movie] = np.linalg.solve(A, b)
      except np.linalg.LinAlgError:
        # fall back to singular value decomposition when matrix is singular
        movie_factors[movie] = np.linalg.lstsq(A, b, rcond=None)[0]

    return movie_factors

  def _sparse_matrix(self, partitions: List[Dict]):
    """
    Builds sparse matrix from partitions

    Args
      partitions: List of partition files metadata

    Returns
      sparse matrix of ratings (csr_matrix)
    """

    columns = ['user_id', 'movie_id', 'rating']

    num_users = len(self.user_map)
    num_movies = len(self.movie_map)

    rows, cols, data = [], [], []

    for partition in tqdm(partitions, desc="Building sparse matrix"):
      df = pd.read_parquet(partition['path'], columns=columns)

      # map user_id, movie_id -> dense indices
      df_users = [self.user_map.get(u, -1) for u in df['user_id']]
      df_movies = [self.movie_map.get(m, -1) for m in df['movie_id']]

      valid_idx = [(i,u,m) for i, (u, m) in enumerate(zip(df_users, df_movies))
                  if u != -1 and m != -1]

      if not valid_idx:
        print(f"\nWARNING: Skipping partition {partition['path']} due to no valid mappings found")
        continue

      # Append to COO format data
      idx, mapped_users, mapped_movies = zip(*valid_idx)
      rows.extend(mapped_users)
      cols.extend(mapped_movies)

      data.extend(df['rating'].iloc[list(idx)].values)

    # construct sparse matrix in CSR format
    return csr_matrix((data, (rows, cols)), shape=(num_users, num_movies))
  
  def _calculate_rmse(self, ratings: csr_matrix):
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

    predictions = np.sum(self.U[row_sample] * self.V[col_sample], axis=1)

    rmse = np.sqrt(np.mean((predictions - ratings_sample) ** 2))

    return rmse






  # ----------- SAVING AND LOADING --------------

  def save_checkpoint(self, checkpoint_dir: str, iteration: int):
    """
    Args
      checkpoint_dir: Directory to save checkpoints
      iteration: Current iteration
      U, V: User and movie latent matrices
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{iteration}.npz")

    start_time = time.time()

    np.savez_compressed(checkpoint_path, 
                        U=self.U, 
                        V=self.V, 
                        iteration=iteration,
                        num_factors=self.num_factors,
                        lambda_reg=self.lambda_reg,
                        default_prediction=self.default_prediction)

    print(f"\nSaved checkpoint to {checkpoint_path}, in {time.time()-start_time} seconds")

  def save_model(self, model_path: str, val_rmse: Optional[float]=None):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    start_time = time.time()

    np.savez_compressed(model_path, 
                        U=self.U, 
                        V=self.V,
                        num_factors=self.num_factors,
                        lambda_reg=self.lambda_reg,
                        default_prediction=self.default_prediction)
    
    print(f"\nTrained Model saved at {model_path}, in {time.time()-start_time} seconds")

    metadata = {
        "model_type": "ALS",
        "num_factors": self.num_factors,
        "lambda_reg": self.lambda_reg,
        "num_iterations": self.num_iters,
        "val_rmse": val_rmse,
        "default_prediction": self.default_prediction,
        "users_count": self.U.shape[0],
        "movies_count": self.V.shape[0],
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    metadata_path = f"{os.path.splitext(model_path)[0]}_metadata.json"
    with open(metadata_path, 'w') as f:
      json.dump(metadata, f, indent=2)

  @classmethod
  def load_model(cls, checkpoint_path: str, 
                 user_map: Dict[int, int], 
                 movie_map: Dict[int, int]):
    """
    Args
      checkpoint_path: Path to checkpoint file
      user_map, movie_map: Mapping from original ID to dense user/movie ID

    Returns
      ALS instance
    """
    print(f"Loading model from {checkpoint_path}")

    checkpoint = np.load(checkpoint_path)

    model = cls(
      num_factors = int(checkpoint['num_factors']),
      lambda_reg = float(checkpoint['lambda_reg']),
      default_prediction = float(checkpoint['default_prediction'])
    )

    model.U = checkpoint['U']
    model.V = checkpoint['V']
    model.user_map = user_map
    model.movie_map = movie_map

    return model