import os
import re
import random
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional

def load_partitions(dir):
    """
    Find all partitions in directory and get metadata

    Args
        dir: directory

    Returns
        List of dict of file metadata (path, part, group, movie_range)
    """

    partition_files = []

    file_pattern = re.compile(r'(train|validation)_part_(\d+)_group_(\d+)_movies_(\d+)-(\d+)\.parquet')
    
    for filename in os.listdir(dir):
        match = file_pattern.match(filename)
        if match:
            file_path = os.path.join(data_dir, filename)
            dataset_type = match.group(1) 
            part_num = int(match.group(2))
            group_num = int(match.group(3))
            movie_start = int(match.group(4))
            movie_end = int(match.group(5))
            
            partition_files.append({
                'path': file_path,
                'dataset': dataset_type,
                'part': part_num,
                'group': group_num,
                'movie_range': (movie_start, movie_end)
            })
    
    # sorted partitions in order
    partition_files.sort(key=lambda x: (x['part'], x['group']))
    return partition_files

def read_partition(file_path: str, columns: Optional[List[str]] = None):
    """
    Read a partition file to pd dataframe
    """
    return pd.read_parquet(file_path, columns=columns)

def load_partition_sample(partition_files: List[Dict[str, Any]], 
                         sample_size: int = 5, 
                         columns: Optional[List[str]] = None):
    """
    Load and combine a sample of partition files for analysis.
    
    Args:
        partition_files: List of partition file metadata
        sample_size: Number of partition files to sample
        columns: Optional list of columns to load
        
    Returns:
        DataFrame containing sampled data
    """

    random.seed(42)

    sampled_files = random.sample(partition_files, sample_size)
    
    # combine samples
    dfs = [read_partition(file_info['path'], columns) for file_info in sampled_files]
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)

    return pd.DataFrame()

def get_partition_stats(partition_files: List[Dict[str, Any]]):
    """
    Compute statistics about the partition files without loading full data.
    
    Args:
        partition_files: List of partition file metadata
        
    Returns:
        Dictionary with statistics about partitions
    """
    num_partitions = len(partition_files)
    unique_parts = len(set(f['part'] for f in partition_files))
    unique_groups = len(set(f['group'] for f in partition_files))
    
    # Find overall movie ID range
    min_movie_id = min(f['movie_range'][0] for f in partition_files)
    max_movie_id = max(f['movie_range'][1] for f in partition_files)
    
    return {
        'num_partitions': num_partitions,
        'unique_parts': unique_parts,
        'unique_groups': unique_groups,
        'movie_id_range': (min_movie_id, max_movie_id)
    }
