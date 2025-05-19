import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import os
from tqdm import tqdm
from tqdm.notebook import tqdm

# 3 helper func parse_probe, process_parquet and process_batch
def parse_probe(path):
  """
  Parse probe.txt

  Returns (movie_id, user_id) pairs
  """
  probe_dict = {}
  curr_movie = None

  with open(path, 'r') as f:
    for line in f:
      line = line.strip()
      if line.endswith(':'):
        curr_movie = int(line[:-1])
      elif line and curr_movie is not None:
        probe_dict[(curr_movie, int(line))] = None

  return probe_dict

def process_batch(batch, probe_pairs, output, append=False):
  """
  Process batch of data. Updating probe_pairs

  Args
    batch: df batch to process
    probe_pairs: (movie_id, user_id) pairs
    output: path to save processed data
    append: whether to append to existing file

  Returns
    tuple: (train_count, probe_count)
  """

  # probe key now matches up to a column in the df batch: 'temp_key'
  batch['temp_key'] = batch['movie_id'].astype(str) + '_' + batch['user_id'].astype(str)

  probe_keys = {f"{movie_id}_{user_id}" for movie_id, user_id in probe_pairs.keys()}

  # mask to separate probe and train data
  mask = batch['temp_key'].isin(probe_keys)

  probe_rows = batch[mask]
  probe_count = len(probe_rows)

  # assign rating to movie_id, user_id pairs
  if probe_count > 0:
    for _, row in probe_rows.iterrows():
      movie_id = row['movie_id']
      user_id = row['user_id']
      probe_pairs[(movie_id, user_id)] = (row['rating'], row['date'])

  # filter out training rows using mask
  train_batch = batch[~mask].drop(columns=['temp_key','index']).reset_index(drop=True)
  train_count = len(train_batch)

  # saving training rows to parquet
  if train_count > 0:
    table = pa.Table.from_pandas(train_batch, preserve_index=False)

    if os.path.exists(output):

      # Append if pq file already exists
      with pq.ParquetWriter(output,
        table.schema,
        compression='zstd',
      ) as writer:
        writer.write_table(table)

    else:
      # otherwise make new with compression
      train_batch.to_parquet(output, compression='zstd', engine='pyarrow',index=False)

  return (train_count, probe_count)

def process_parquet(df, probe_pairs, output_base, batch_size=3):
  """
  Process parquet file. Get grouth truth ratings for each probe pair
  Remove probe datapoints from training data

  Args
    df: path to Parquet file
    probe_pairs: (movie_id, user_id) pairs
    output: path to save processed data
    batch_size: number of pq ROW GROUPS to process at a time

  Returns
    tuple: (training_count, probe_count)
  """
  probe_count = 0
  train_count = 0
  output_files = []

  #batch_size = 500000
  #first_batch = True

  if not isinstance(df, str):
    raise TypeError("df must be path to a Parquet file")

  pq_file = pq.ParquetFile(df)
  total_row_groups = pq_file.num_row_groups

  for group in tqdm(range(0, total_row_groups, batch_size)):
    #print(f'Processing row group {group+1}/{total_row}')

    row_groups_batch = list(range(group, min(group + batch_size, total_row_groups)))
    #print(f"Processing rowgroups {row_groups_batch}")

    batch = pd.concat([pq_file.read_row_group(i).to_pandas() for i in row_groups_batch], ignore_index=True)

    min_movie_id = batch['movie_id'].min()
    max_movie_id = batch['movie_id'].max()

    # build file name to describe partition's content
    output_path = f"{output_base}_group_{group//batch_size}_movies_{min_movie_id}-{max_movie_id}.parquet"
    output_files.append(output_path)

    batch_train_count, batch_probe_count = process_batch(batch, probe_pairs, output_path,
                                                          append=False) # append = not first_batch

    train_count += batch_train_count
    probe_count += batch_probe_count
    #first_batch = False

  return train_count, probe_count, output_files

def create_test_df(probe_pairs, output):
  """
  Create test dataframe from probe_pairs

  Args
    probe_pairs: dict with (movie_id, user_id) as key and rating as value
    output: path to save processed data

  Returns
    test_df: probe entries + rating
  """

  test_data = []
  none_count = 0

  for (movie_id, user_id), data in probe_pairs.items():
    if data is not None:
      test_data.append({'movie_id': movie_id,
                        'user_id': user_id,
                        'rating': data[0],
                        'date': data[1]})
    else:
      none_count += 1

  test_df = pd.DataFrame(test_data)
  print(f"Created test set with {len(test_df):,} entries")

  test_df.to_parquet(output, compression='zstd', engine='pyarrow')

  return none_count


def process_netflix(dfs, probe_path, save_dir):
  """
  Process netflix data. Get grouth truth ratings for each probe pair
  Remove probe datapoints from training data

  Args
    dfs: list of dataframes of Netflix data
    probe_path: path to probe.txt
    save_dir: directory to save processed data

  Returns
    train_df: dataframes with probe entries removed
    test_df: probe entries + rating
  """

  os.makedirs(save_dir, exist_ok=True)

  # Parse probe into dictionary > faster lookup
  print("Parsing probe file")
  probe_dict = parse_probe(probe_path)
  print(f"Found {len(probe_dict)} probe pairs")


  # Extract ratings from probe entries and remove from training set
  train_files = []

  total_train = 0
  total_matches = 0

  for i, df_source in tqdm(enumerate(dfs), total=len(dfs)):

    # construct save path -> saving to multiple files
    train_path_base = os.path.join(save_dir, f'train_part_{i+1}')

    part_train_count, part_probe_count, df_files = process_parquet(
        df_source, probe_dict, train_path_base
    )

    total_train += part_train_count
    total_matches += part_probe_count
    train_files.extend(df_files)

    print(f"Source {i+1} stats:")
    print(f"  - Added {part_train_count:,} ratings to training set")
    print(f"  - Found {part_probe_count:,} probe ratings")

  # Create test set from probe dict
  test_path = os.path.join(save_dir, 'test.parquet')
  none_count = create_test_df(probe_dict, test_path)
  print(f"Created test set with {none_count:,} missing ratings")

  print("\nProcessing complete!")
  print(f"Total training ratings: {total_train:,}")
  print(f"Total probe ratings found: {total_matches:,} out of {len(probe_dict):,}")

  return train_files, test_path