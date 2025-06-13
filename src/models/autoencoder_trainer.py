import os
import torch
import torch.optim as optim
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import random
import gzip
from .autoencoder import AutoEncoder
import optuna

def train_autoencoder(
    train_partitions: List[Dict],
    user_map: Dict[int, int],
    movie_map: Dict[int, int],
    num_epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    hidden_dims: List[int] = [512, 256, 128],
    dropout: float = 0.5,
    l2_reg: float = 0.001,
    checkpoint_dir: Optional[str]=None,
    checkpoint_interval: int = 5,
    eval_interval: int = 5,
    validation_partitions: Optional[List[Dict]] = None,
    resume_path: Optional[str]=None,
    trial: Optional[optuna.Trial]=None,
    user_data: Optional[Dict]=None,
    validation_data: Optional[Dict]=None
):
    """
    Train autoencoder on Netflix partitioned data.
    
    Args:
        train_partitions: List of partition file dicts
        user_map: User ID mapping
        movie_map: Movie ID mapping  
        num_epochs: Training epochs
        batch_size: Users per batch
        learning_rate: Optimizer learning rate
        hidden_dims: Hidden layer dimensions
        dropout: Dropout rate
        l2_reg: L2 regularization strength
        checkpoint_dir: Directory for checkpoints
        checkpoint_interval: Checkpoint frequency
        validation_partitions: Optional validation data
        resume_path: Path to latest checkpoint
        trial: Optional optuna trial for tuning
        user_data: Optional preloaded train user data
        validation_data: Optional preloaded validation user data
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # calculate global mean for default predictions
    global_mean = calculate_global_mean(train_partitions)
    print(f"Global mean rating: {global_mean:.3f}")
    
    # init
    model = AutoEncoder(
        num_movies=len(movie_map),
        hidden_dims=hidden_dims,
        dropout=dropout,
        l2=l2_reg,
        global_mean=global_mean
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # use passed user data or load new
    if user_data is None:
        user_data = model.load_user_data(train_partitions, user_map)
    
    if validation_partitions and validation_data is None:
        validation_data = model.load_validation_data(validation_partitions, user_map)

    # check for checkpoint to resume
    start_epoch = 0 
    if resume_path and os.path.exists(resume_path):
        try:
            start_epoch, last_loss, metadata = load_checkpoint(
                resume_path, model, optimizer, device
            )
            print(f"Resuming training from epoch {start_epoch}")
        except Exception as e:
            raise ValueError(e)
        
    
    # train loop
    model.train()
    users = list(user_data.keys())
    val_loss = None
    
    for epoch in range(start_epoch,num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        random.shuffle(users)
        
        progress_bar = tqdm(
            range(0, len(users), batch_size), 
            desc=f"Epoch {epoch+1}/{num_epochs}"
        )
        
        for i in progress_bar:
            batch_users = users[i:i + batch_size]
            
            # create tensors
            batch_ratings, batch_masks = create_batch_tensors(
                batch_users, user_data, movie_map, len(movie_map), global_mean
            )
            
            batch_ratings = batch_ratings.to(device)
            batch_masks = batch_masks.to(device)
            
            # forward
            optimizer.zero_grad()
            predictions = model(batch_ratings, batch_masks)
            loss = model.masked_loss(predictions, batch_ratings, batch_masks)
            
            # backward
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # validation
        if validation_data and epoch % eval_interval == 0:
            #val_loss = evaluate_model(model, validation_data, movie_map, device)
            eval_results = evaluate_model(model=model, 
                                          validation_data=validation_data, 
                                          training_data=user_data, 
                                          user_map=user_map, 
                                          movie_map=movie_map, 
                                          device=device)
            
            val_loss = eval_results['loss']
            val_rmse = eval_results['rmse']

            # check if tuning, if yes report back if rmse is bad
            if trial is not None:
                trial.report(val_rmse, epoch)
                if trial.should_prune(): 
                    raise optuna.TrialPruned() 
            
            model.train()
            print(f"Validation | Loss: {val_loss:.4f}, RMSE: {val_rmse:.4f}")
        
        # checkpointing
        if checkpoint_dir and (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth.gz")
            
            metadata = {
                'val_loss': val_loss,
                'hidden_dims': hidden_dims,
                'dropout': dropout,
                'learning_rate': learning_rate
            }
            
            model_data = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'current_epoch': epoch + 1,
                'train_loss': avg_loss,
                'metadata': metadata
            }
    
            with gzip.open(checkpoint_path, 'wb') as f:
                torch.save(model_data, f)

            print(f"Saved checkpoint at {checkpoint_path}")
    
    # save final model
    if checkpoint_dir:
        final_path = os.path.join(checkpoint_dir, "final_model.pth.gz")

        metadata = {
            'epochs': num_epochs,
            'train_loss': avg_loss,
            'val_loss': val_loss,
            'hidden_dims': hidden_dims,
            'dropout': dropout,
            'learning_rate': learning_rate
        }

        model_data = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metadata': metadata
        }

        with gzip.open(final_path, 'wb') as f:
            torch.save(model_data, f)

        print(f"Saved final model at {final_path}")
    
    return model, val_rmse

def load_checkpoint(path, model, optimizer, device):
    with gzip.open(path, 'rb') as f:
        checkpoint = torch.load(f, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['current_epoch']
    loss = checkpoint['train_loss']
    metadata = checkpoint.get('metadata', {})

    return epoch, loss, metadata

def create_batch_tensors(
    batch_users: List[int],
    user_data: Dict[int, List],
    movie_map: Dict[int, int],
    num_movies: int,
    global_mean: float
):
    """Create batch tensors for training."""
    batch_ratings = []
    batch_masks = []
    
    for user_id in batch_users:
        # create user rating vector
        ratings_vec = np.full(num_movies, global_mean, dtype=np.float32)
        mask = np.zeros(num_movies, dtype=np.float32)
        
        for movie_id, rating in user_data[user_id]:
            if movie_id in movie_map:
                movie_idx = movie_map[movie_id]
                ratings_vec[movie_idx] = rating
                mask[movie_idx] = 1.0
        
        batch_ratings.append(ratings_vec)
        batch_masks.append(mask)
    
    return (
        torch.tensor(np.array(batch_ratings), dtype=torch.float32),
        torch.tensor(np.array(batch_masks), dtype=torch.float32)
    )

def evaluate_model(
    model,
    validation_data: List[Tuple[int, int, float]],
    training_data: Dict[int, List[Tuple[int, float]]],
    user_map: Dict[int, int],
    movie_map: Dict[int, int],
    device: torch.device,
    batch_size: int = 512
):
    """Evaluate model on validation data."""
    model.eval()
    num_movies = len(movie_map)

    predictions = []
    targets = []

    with torch.no_grad():
        for i in tqdm(range(0, len(validation_data), batch_size), desc="Evaluating"):
            batch = validation_data[i:i+batch_size]
            batch_predictions = []
            batch_targets = []

            for user_id, movie_id, true_rating in batch:

                # cold start strategy predict global mean
                if user_id not in user_map or movie_id not in movie_map:
                    prediction = model.global_mean
                else:
                    mapped_user = user_map[user_id]
    
                    # user isn't in training data
                    if mapped_user not in training_data:
                        prediction = model.global_mean
                    else:
                        # get user data for all other movies except target
                        user_ratings = [(m_id, rating) for (m_id, rating) in training_data[mapped_user] 
                                        if m_id != movie_id]
        
                        if len(user_ratings) > 0:
                            # input vector
                            ratings_vec = np.full(num_movies, model.global_mean)
                            mask = np.zeros(num_movies)
        
                            for m_id, rating in user_ratings:
                                if m_id in movie_map:
                                    movie_idx = movie_map[m_id]
                                    ratings_vec[movie_idx] = rating
                                    mask[movie_idx] = 1.0
        
                            input_tensor = torch.tensor(ratings_vec, dtype=torch.float32).unsqueeze(0).to(device)
                            mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(device)
        
                            predictions_tensor = model(input_tensor, mask_tensor)
                            target_idx = movie_map[movie_id]
                            prediction = predictions_tensor[0, target_idx].item()
                        else:
                            # user doesnt have any training ratings
                            prediction = model.global_mean
                        
                batch_predictions.append(prediction)
                batch_targets.append(true_rating)
    
            predictions.extend(batch_predictions)
            targets.extend(batch_targets)

    model.train()

    predictions_arr = np.array(predictions)
    targets_arr = np.array(targets)

    mse = np.mean((predictions_arr - targets_arr) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions_arr - targets_arr))
    
    return {
        'loss': mse,
        'rmse': rmse,
        'mae': mae,
        'total_predictions': len(predictions)
    }

def evaluate_model_BAK(
    model,
    validation_data: List[Tuple[int, int, float]],
    movie_map: Dict[int, int],
    device: torch.device,
    batch_size: int = 512
):
    """Evaluate model on validation data."""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    total_predictions = 0
    num_batches = 0
    
    users = list(validation_data.keys())
    
    with torch.no_grad():
        for i in range(0, len(users), batch_size):
            batch_users = users[i:i + batch_size]
            
            batch_ratings, batch_masks = create_batch_tensors(
                batch_users, validation_data, movie_map, 
                len(movie_map), model.global_mean
            )
            
            batch_ratings = batch_ratings.to(device)
            batch_masks = batch_masks.to(device)
            
            predictions = model(batch_ratings, batch_masks)
            
            loss = model.masked_loss(predictions, batch_ratings, batch_masks)

            observed_mask = batch_masks.bool()
            observed_predictions = predictions[observed_mask]
            observed_targets = batch_ratings[observed_mask]
            
            if observed_predictions.size(0) > 0:
                mse = torch.mean((observed_predictions - observed_targets) ** 2)
                mae = torch.mean(torch.abs(observed_predictions - observed_targets))
                
                total_loss += loss.item()
                total_mae += mae.item() * observed_predictions.size(0)
                total_rmse += mse.item() * observed_predictions.size(0)
                total_predictions += observed_predictions.size(0)

                num_batches += 1
    
    model.train()

    avg_loss = total_loss / num_batches
    rmse = np.sqrt(total_rmse / total_predictions)
    mae = total_mae / total_predictions
    
    return {
        'loss': avg_loss,
        'rmse': rmse,
        'mae': mae
    }

def calculate_global_mean(partitions: List[Dict], sample_size: int = 10):
    """Calculate global mean from partition sample."""
    total_sum = 0.0
    total_count = 0
    
    sample = random.sample(partitions, min(sample_size, len(partitions)))
    
    for partition in tqdm(sample, desc="Calculating global mean"):
        df = pd.read_parquet(partition['path'])
        total_sum += df['rating'].sum()
        total_count += len(df)
    
    if total_count == 0:
        raise ValueError("No ratings found in partitions")
    
    return total_sum / total_count
