import os
import json
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import re

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


def save_mappings(user_map: Dict[int, int], movie_map: Dict[int, int], output_path: str):
    """
    Stores user and movie ID mappings in JSON format
    """

    user_map_str = {str(k): v for k, v in user_map.items()}
    movie_map_str = {str(k): v for k, v in movie_map.items()}

    with open(os.path.join(output_path, "user_map.json"), "w") as f:
        json.dump(user_map_str, f)

    with open(os.path.join(output_path, "movie_map.json"), "w") as f:
        json.dump(movie_map_str, f)



def load_mappings(mappings_dir: str):
    """
    Load ID mappings from json, fast lookup
    """

    with open(os.path.join(mappings_dir, "user_map.json"), "r") as f:
        user_map = json.load(f)

    with open(os.path.join(mappings_dir, "movie_map.json"), "r") as f:
        movie_map = json.load(f)

    return user_map, movie_map


def analyze_partition_metadata(partition_files: List[Dict[str, Any]]):
    """
    General stats analysis
    """

    ## MIGHT NOT NEED

    return stats


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