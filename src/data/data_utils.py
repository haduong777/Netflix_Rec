import os
import json
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np


def create_id_mappings(partition_files: List[Dict[str, Any]]):
    """
    Build user and movie ID sequential mapping > better for Matrix Factorization
    """

    return user_map, movie_map


def save_mappings(user_map: Dict[int, int], movie_map: Dict[int, int], output_path: str):
    """
    Stores user and movie ID mappings in JSON format 
    """



def load_mappings(mappings_dir: str):
    """
    Load ID mappings from json
    """

    return user_map, movie_map


def analyze_partition_metadata(partition_files: List[Dict[str, Any]]):
    """
    General stats analysis
    """

    return stats


def convert_ids_to_indices(df: pd.DataFrame, user_map: Dict[int, int], movie_map: Dict[int, int]):
    """
    Use mapping to convert
    """
    result = df.copy()
    ##
    
    return result


def get_partition_info(file_path: str):
    """
    Get metadata from file name
    """
    
    return info