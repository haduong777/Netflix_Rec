import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import random
from pathlib import Path
import pickle
from tqdm import tqdm

class AutoEncoder(nn.Module):
    def __init__(self, num_movies: int,
                 hidden_dims: List[int] = [512, 256, 128],
                 dropout: float = 0.5,
                 l2: float = 0.001,
                 global_mean: float = 3):
      super().__init__()
      self.num_movies = num_movies
      self.global_mean = global_mean
      self.l2 = l2
      self.global_mean = global_mean

      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

      encoder_layers = []
      prev_dim = num_movies

      for i, dim in enumerate(hidden_dims):
        encoder_layers.append(nn.Linear(prev_dim, dim))
        encoder_layers.append(nn.ReLU())

        if i < len(hidden_dims) - 1:
          encoder_layers.append(nn.Dropout(dropout)) # dropouts except bottleneck layer

        prev_dim = dim

      self.encoder = nn.Sequential(*encoder_layers)

      decoder_layers = []
      hidden_dims = list(reversed(hidden_dims))

      for i, dim in enumerate(hidden_dims):
        decoder_layers.append(nn.Linear(prev_dim, dim))
        decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Dropout(dropout))
        prev_dim = dim

      decoder_layers.append(nn.Linear(prev_dim, num_movies))
      self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
      """
      Define forward pass
      """
      masked_input = x * mask # zero for unobserved ratings

      z = self.encoder(masked_input)
      x_hat = self.decoder(z)
      return x_hat

    def encode(self, x: torch.Tensor, mask: torch.Tensor):
      masked_input = x * mask
      return self.encoder(masked_input)

    def decode(self, z: torch.Tensor):
      return self.decoder(z)

    def masked_loss(self, predictions: torch.Tensor,
                  targets: torch.Tensor,
                  mask: torch.Tensor):
      """
      Calculates the masked MSE loss + regularization

      Args:
        predictions: (batch_size, num_movies)
        targets: (batch_size, num_movies)
        mask: (batch_size, num_movies)

      returns
        mse loss with regularization
      """
      masked_pred = predictions * mask
      masked_target = targets * mask

      mse = torch.sum((masked_pred - masked_target) ** 2) / torch.sum(mask)

      l2_loss = sum(torch.norm(param, 2) ** 2 for param in self.parameters())

      return mse + self.l2 * l2_loss

    def create_user_ratings(self, user_ratings: List[Tuple[int, float]],
                            movie_map: Dict[int, int]):
      """
      User rating vector and mask from maps of (movie_id, rating)

      Args
        user_ratings: List of (movie_id, rating) tuples
        movie_map: Dict of {movie_id: index}

      returns
        user_ratings_vec: (num_movies,)
        mask: (num_movies,)
      """
      mask = np.zeros(self.num_movies, dtype=np.float32)

      user_ratings_vec = np.full(self.num_movies, self.global_mean, dtype=np.float32)

      for movie_id, rating in user_ratings:
        if movie_id in movie_map:
          movie_idx = movie_map[movie_id]
          user_ratings_vec[movie_idx] = rating
          mask[movie_idx] = 1.0

      return user_ratings_vec, mask

    def predict(self, user_ratings: List[Tuple[int, float]],
                movie_map: Dict[int, int],
                movie_id: Optional[int]=None):
      """
      Predicts ratings for all movies if id isn't provided, if it is predict for that movie_id

      Args
        user_ratings: List of (movie_id, rating) tuples
        movie_map: Dict of {movie_id: index}
        movie_id: Optional movie_id to predict for (singular)

      returns
        predictions: (num_movies,)
      """
      user_ratings_vec, mask = self.create_user_ratings(user_ratings, movie_map)

      with torch.no_grad():
        user_ratings_vec = torch.tensor(user_ratings_vec, dtype=torch.float32).unsqueeze(0).to_device(self.device)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to_device(self.device)

        predictions = self.forward(user_ratings_vec, mask)

      if movie_id is None:
        return predictions.cpu().numpy().squeeze()

      if movie_id not in movie_map:
        print(f"WARNING: Movie {movie_id} not found in movie map, returning global mean")
        return self.global_mean

      predictions = self.predict_all_ratings(user_ratings, movie_map)

      return float(predictions[movie_map[movie_id]])

    def batch_predict(self, df: pd.DataFrame,
                      user_data: Dict[int, List[Tuple[int, float]]],
                      user_map: Dict[int, int],
                      movie_map: Dict[int, int]):
      """
      Batch predict for val and test

      Args
        df: pd.DataFrame of user_id, movie_id
        user_data: Dict of {user_id: List of (movie_id, rating)}

      returns
        predictions: np.array of predictions
      """

      predictions = []

      for _, row in df.iterrows():
        user_id = row['user_id']
        movie_id = row['movie_id']

        if user_id not in user_map or movie_id not in movie_map:
          predictions.append(self.global_mean)
          continue

        mapped_user = user_map[user_id]
        if mapped_user in user_data:
          # filter out ratings that exist in the training data -- prevent leakage
          user_ratings = [(movie_train_id, rating)
                          for movie_train_id, rating in user_data[mapped_user]
                          if movie_train_id != movie_id]

          predictions.append(self.predict(user_ratings, movie_map, movie_id))
        else:
          predictions.append(self.global_mean)

      return np.array(predictions)

    def load_user_data(self, partitions: List[Dict],
                       user_map: Dict[int, int]):
      """
      Loads user data from partitions ***CURRENTLY MEMORY INTENSIVE***

      Args
        partitions: List of dicts with 'path' and 'users' keys
        user_map: Dict of {user_id: mapped_user_id}

      returns
        data: Dict of {user_id: List of (movie_id, rating)}
      """

      data = {}

      for f in tqdm(partitions, desc="Loading user data"):
        df = pd.read_parquet(f['path'])

        for user_id in f['users'].unique():
          if user_id in user_map:
            mapped_user = user_map[user_id]
            ratings = df[df['user_id'] == user_id][['movie_id','rating']]

            if mapped_user not in data:
              data[mapped_user] = []

            data[mapped_user].extend([(int(movie_id), float(rating))
                                       for movie_id, rating in ratings])

      return data

    def get_latent(self, user_data: Dict[int, List[Tuple[int, float]]],
                   movie_map: Dict[int, int],
                   batch_size: int = 512):
      """
      Gets the latent representation of the user

      Args
        user_data: Dict of {user_id: List of (movie_id, rating)}
        movie_map: Dict of {movie_id: index}

      returns
        latent_reps: Dict of {user_id: latent_representation}
      """

      self.eval() # no dropouts use all neurons

      latent_reps = {}

      users = list(user_data.keys())

      with torch.no_grad():
        for i in tqdm(range(0, len(users), batch_size), desc="Computing latent representations"):
          batch_users = users[i:i + batch_size]
          batch_ratings = []
          batch_mask = []

          for user_id in batch_users:
            rating_vec, mask = self.create_user_ratings(user_data[user_id], movie_map)

            batch_ratings.append(rating_vec)
            batch_mask.append(mask)

          batch_ratings = torch.tensor(np.array(batch_ratings), dtype=torch.float32).to(self.device)
          batch_mask = torch.tensor(np.array(batch_mask), dtype=torch.float32).to(self.device)

          latent_reps_batch = self.encode(batch_ratings, batch_mask)

          for j, user_id in enumerate(batch_users):
            latent_reps[user_id] = latent_reps_batch[j].cpu().numpy()

      return latent_reps

    # ---------------- SAVE AND LOADING ------------------

    def save(self, path: Path, metadata: Optional[Dict] = None):
      """
      Saves the model
      """
      save_dict = {
          'model_state_dict': self.state_dict(),
          'num_movies': self.num_movies,
          'global_mean': self.global_mean,
          'l2': self.l2,
      }

      if metadata:
        self.save_metadata(metadata)

      torch.save(save_dict, path)
      print(f"Model saved to {path}")

    def save_metadata(self, metadata: Dict):

      ###
      return

    @classmethod
    def load(cls, path: Path,
             hidden_dims: List[int] = [512, 256, 128],
             dropout: float = 0.5):
      """
      Loads the model
      """
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      checkpoint = torch.load(path, map_location=device)

      model = cls(num_movies = checkpoint['num_movies'],
                  hidden_dims = checkpoint['hidden_dims'],
                  dropout = dropout,
                  l2 = checkpoint.get('l2', 0.001),
                  global_mean = checkpoint.get('global_mean', 3.0))

      model.load_state_dict(checkpoint['model_state_dict'])
      model.to(device)

      metadata = checkpoint.get('metadata', None)

      return model, metadata


    ### -------------------------------------------------------


    def _calculate_global_mean(self, partitions: List[Dict],
                               sample_size: int = 10):
      """
      Calculates the global mean of the ratings in the partitions (sampled)
      """
      global_sum = 0
      global_count = 0

      sample = random.sample(partitions, sample_size)

      for f in tqdm(sample, desc="Calculating global mean"):
        df = pd.read_parquet(f['path'])
        global_sum += df['rating'].sum()
        global_count += len(df)

      if global_count == 0:
        raise ValueError("No ratings found in partitions")

      return global_sum / global_count

