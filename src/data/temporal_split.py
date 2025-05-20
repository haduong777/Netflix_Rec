import os
import random
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.data.loader import load_partitions, read_partition


def sample_date_distribution(partition_files: List[Dict[str, Any]], 
                             sample_ratio: float = 0.15):
    """
    Sample dates from a subset of files to estimate date distribution.
    
    Args:
        partition_files: List of partition file information
        sample_ratio: Ratio of partition files to sample
    
    Returns:
        List of sampled dates
    """
    
    # sample
    num_files = int(len(partition_files) * sample_ratio)
    sampled_files = random.sample(partition_files, num_files)
    
    # DEBUG -----------------------------------------------------------------------
    print(f"Sampling dates from {num_files} files out of {len(partition_files)}")
    
    # get dates
    sampled_dates = []
    for file_info in tqdm(sampled_files, desc="Sampling dates"):
        df = read_partition(file_info['path'], columns=['date'])
        sampled_dates.extend(df['date'].tolist())

        del df  # free memory

    return sampled_dates


def determine_split_date(sampled_dates: List[datetime], 
                         test_ratio: float = 0.1):
    """
    Calculate date threshold for splitting based on sampled dates.
    
    Args:
        sampled_dates: List of sampled dates
        test_ratio: Ratio of validation date
    
    Returns:
        Split date
    """
    # find most recent % of dates based on test_ratio
    sorted_dates = sorted(sampled_dates)
    split_idx = int(len(sorted_dates) * (1 - test_ratio))
    split_date = sorted_dates[split_idx]
    
    print(f"Split date: {split_date}")
    return split_date


def split_partition_files(partition_files: List[Dict[str, Any]], 
                          split_date: datetime, 
                          output_dir: str):
    """
    Split each partition file into train/validation based on date.
    
    Args:
        partition_files: List of partition file information
        split_date: Date to use as threshold for splitting
        output_dir: Directory to save split files
    
    Returns:
        Tuple of (train_files, validation_files)
    """
    
    # ggdrive/S3
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'validation'), exist_ok=True)
    
    train_files = []
    validation_files = []
    
    for file_info in tqdm(partition_files, desc="Splitting files"):

        df = read_partition(file_info['path'])
        
        # temporal split -> most recent % == validation set
        train_df = df[df['date'] < split_date]
        validation_df = df[df['date'] >= split_date]

        train_file_info = None
        validation_file_info = None
        
        # create train/validation sets
        if len(train_df) > 0:
            train_path = os.path.join(output_dir, 'train', 
                                      f"part_{file_info['part']}_{file_info['group']}.parquet")

            train_df.to_parquet(train_path, compression='zstd')
            train_file_info = {
                'path': train_path,
                'part': file_info['part'],
                'group': file_info['group'],
                'movie_range': file_info['movie_range'],
                'count': len(train_df)
            }
            train_files.append(train_file_info)
        
        if len(validation_df) > 0:
            validation_path = os.path.join(output_dir, 'validation', f"part_{file_info['part']}_{file_info['group']}.parquet")
            validation_df.to_parquet(validation_path, compression='zstd')
            validation_file_info = {
                'path': validation_path,
                'part': file_info['part'],
                'group': file_info['group'],
                'movie_range': file_info['movie_range'],
                'count': len(validation_df)
            }
            validation_files.append(validation_file_info)
            
        # Free memory
        del df, train_df, validation_df

    return train_files, validation_files


def analyze_split_statistics(train_files: List[Dict[str, Any]], 
                             validation_files: List[Dict[str, Any]]):
    """
    Analyze split stats
    
    Args:
        train_files: List of training partition
        validation_files: List of validation partition
    
    Returns:
        Dictionary of split stats
    """

    train_count = sum(file_info.get('count', 0) for file_info in train_files)
    validation_count = sum(file_info.get('count', 0) for file_info in validation_files)
    total_count = train_count + validation_count
    
    train_movies = set()
    validation_movies = set()
    for file_info in train_files:
        if 'movie_range' in file_info:
            start, end = file_info['movie_range']
            train_movies.update(range(start, end + 1))
    
    for file_info in validation_files:
        if 'movie_range' in file_info:
            start, end = file_info['movie_range']
            validation_movies.update(range(start, end + 1))
    
    stats = {
        'train_count': train_count,
        'validation_count': validation_count,
        'total_count': total_count,
        'train_ratio': train_count / total_count if total_count > 0 else 0,
        'validation_ratio': validation_count / total_count if total_count > 0 else 0,
        'train_files': len(train_files),
        'validation_files': len(validation_files),
        'train_movies': len(train_movies),
        'validation_movies': len(validation_movies),
        'movies_in_both': len(train_movies.intersection(validation_movies)),
        'train_only_movies': len(train_movies - validation_movies),
        'validation_only_movies': len(validation_movies - train_movies)
    }
    
    return stats


def save_split_metadata(train_files: List[Dict[str, Any]], 
                        validation_files: List[Dict[str, Any]], 
                        split_date: datetime, 
                        stats: Dict[str, Any], 
                        output_dir: str):
    """
    Save metadata.
    
    Args:
        train_files: List of training partition
        validation_files: List of validation partition
        split_date: Date used for splitting
        stats: Statistics about the split
        output_dir: metadata directory
    """
    metadata_dir = os.path.join(output_dir, 'metadata')
    os.makedirs(metadata_dir, exist_ok=True)
    
    metadata = {
        'split_date': split_date.isoformat(),
        'statistics': stats,
        'created_at': datetime.now().isoformat()
    }
    
    # save to json
    with open(os.path.join(metadata_dir, 'split_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save file lists 
    def simplify_file_info(file_info):
        simple_info = file_info.copy()
        if 'path' in simple_info:
            simple_info['filename'] = os.path.basename(simple_info['path'])
            del simple_info['path']
        return simple_info
    
    with open(os.path.join(metadata_dir, 'train_files.json'), 'w') as f:
        json.dump([simplify_file_info(f) for f in train_files], f, indent=2)
        
    with open(os.path.join(metadata_dir, 'validation_files.json'), 'w') as f:
        json.dump([simplify_file_info(f) for f in validation_files], f, indent=2)


def create_temporal_split(data_dir: str, 
                          output_dir: str, 
                          test_ratio: float = 0.1, 
                          sample_ratio: float = 0.1, 
                          min_files: int = 5):
    """
    Execute full splitting process.
    
    Args:
        data_dir: Partitions directory
        output_dir: Output directory
        test_ratio: Ratio of data to use for validation
        sample_ratio: Ratio of partition files to sample for date distribution
        min_files: Minimum number of files to sample
    
    Returns:
        Dictionary with split information
    """

    partition_files = load_partitions(data_dir)
    print(f"Found {len(partition_files)} partition files")
    
    # sample dates -> dates distribution
    sampled_dates = sample_date_distribution(partition_files, sample_ratio, min_files)
    
    # determine split date
    split_date = determine_split_date(sampled_dates, test_ratio)
    
    # perform split
    train_files, validation_files = split_partition_files(partition_files, split_date, output_dir)
    
    stats = analyze_split_statistics(train_files, validation_files)
    
    # Save metadata
    save_split_metadata(train_files, validation_files, split_date, stats, output_dir)
    
    print("Completed temporal split")
    
    return {
        'train_files': train_files,
        'validation_files': validation_files,
        'split_date': split_date,
        'statistics': stats
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Split Netflix dataset into train/validation based on date")
    parser.add_argument("--data_dir", required=True, help="Directory containing raw partition files")
    parser.add_argument("--output_dir", required=True, help="Directory to save split files")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Ratio of data to use for validation")
    parser.add_argument("--sample_ratio", type=float, default=0.1, help="Ratio of partition files to sample")
    parser.add_argument("--min_files", type=int, default=10, help="Minimum number of files to sample")
    
    args = parser.parse_args()
    
    create_temporal_split(
        args.data_dir, 
        args.output_dir, 
        args.test_ratio,
        args.sample_ratio,
        args.min_files
    )