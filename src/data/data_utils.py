import os
import json
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm
import re

def map_id(partition_files: List[Dict], sample_size=None):
    """
    Map user and movie ids to denser indicies -> for matrix factorization
    """
    
    user_ids = set()
    movie_ids = set()
    
    if sample_size and sample_size < len(partition_files):
        partition_sample = random.sample(partition_files, sample_size)
    else:
        partition_sample = partition_files
    
    for partition in tqdm(partition_sample, desc="Mapping IDs"):
        df = pd.read_parquet(partition['path'], columns=['user_id', 'movie_id'])
        
        user_ids.update(df['user_id'].unique())
        movie_ids.update(df['movie_id'].unique())
    
    user_id_map = {user_id: idx for idx, user_id in enumerate(sorted(user_ids))}
    movie_id_map = {movie_id: idx for idx, movie_id in enumerate(sorted(movie_ids))}
    
    print(f"Map successful for {len(user_id_map)} users, {len(movie_id_map)} movies")
    
    return user_id_map, movie_id_map

def get_data(data_dir: str):
    """
    Get partitions metadata in the form part_X_X.parquet
    """
    partition_files = []
    
    pattern = re.compile(r'part_(\d+)_(\d+)\.parquet$')
    
    for filename in os.listdir(data_dir):
        match = pattern.match(filename)
        if match:
            file_path = os.path.join(data_dir, filename)
            part_num = int(match.group(1))
            group_num = int(match.group(2))
        
            partition_files.append({
                'path': file_path,
                'part': part_num,
                'group': group_num
            })
    
    # sorted partitions in order
    partition_files.sort(key=lambda x: (x['part'], x['group']))
    return partition_files

def create_id_mappings(partition_files: List[Dict[str, Any]]):
    """
    Build user and movie ID sequential mapping > better for Matrix Factorization

    Returns
      user_map: Dict[int, int]
      movie_map: Dict[int, int]
    """

    user_ids = set()
    movie_ids = set()

    for file in partition_files:
      df = pd.read_csv(file["path"], header=None, names=["user_id", "movie_id", "rating"])
      user_ids.update(df["user_id"].unique())
      movie_ids.update(df["movie_id"].unique())

    user_map = {user_id: idx for idx, user_id in enumerate(sorted(user_ids))}
    movie_map = {movie_id: idx for idx, movie_id in enumerate(sorted(movie_ids))}

    return user_map, movie_map


def convert_ids_to_indices(df: pd.DataFrame, user_map: Dict[int, int], movie_map: Dict[int, int]):
    """
    Use mapping to convert
    """
    result = df.copy()

    result["user_id"] = result["user_id"].map(user_map)
    result["movie_id"] = result["movie_id"].map(movie_map)

    missing_user_ids = result["user_id"].isna().sum()
    missing_movie_ids = result["movie_id"].isna().sum()

    if missing_user_ids > 0 or missing_movie_ids > 0:
        print(f"WARNING: MAPPING ISSUE")
        print(f"WARNING: {missing_user_ids} missing user_ids and {missing_movie_ids} missing movie_ids")

    return result


def get_partition_info(file_path: str):
    """
    Get metadata from file name
    """

    file_name = os.path.basename(file_path)
    file_name_parts = file_name.split("_")

    info = {
        'path': file_path,
        'name': file_name,
        'group': None,
        'part': None,
        'movie_range': None,
    }

    part_pattern = r'part[_-](\d+)'
    group_pattern = r'group[_-](\d+)'
    movie_range_pattern = r'movies[_-](\d+)[_-](\d+)'
    date_pattern = r'date[_-](\d{4}-\d{2}-\d{2})'

    part_match = re.search(part_pattern, file_name, re.IGNORECASE)
    if part_match:
        info['part'] = int(part_match.group(1))

    group_match = re.search(group_pattern, file_name, re.IGNORECASE)
    if group_match:
        info['group'] = int(group_match.group(1))

    movie_range_match = re.search(movie_range_pattern, file_name, re.IGNORECASE)
    if movie_range_match:
        start_id = int(movie_range_match.group(1))
        end_id = int(movie_range_match.group(2))
        info['movie_range'] = (start_id, end_id)

    return info