import boto3
import pandas as pd
from typing import List, Dict, Any, Iterator, Optional
from botocore.exceptions import ClientError
import io
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class S3Loader:
    """Handles streaming Netflix partitions from AWS S3."""
    
    def __init__(self, bucket_name: str, access_key: Optional[str] = None,
                 secret: Optional[str] = None, region: str = 'us-west-1'):
        """Initialize S3 client and bucket configuration."""
        self.bucket_name = bucket_name
        self.s3_client = boto3.client(
            's3',
            access_key=access_key,
            secret=secret,
            region=region
        )
    
    def get_partitions(self, prefix: str = ''):
        """Find all partition files in S3 bucket and extract metadata."""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            if 'Contents' not in response:
                logger.warning(f"No files found with prefix: {prefix}")
                return []
            
            partition_files = []
            for obj in response['Contents']:
                key = obj['Key']
                if key.endswith('.parquet'):
                    file_info = self._extract_file_metadata(key, obj)
                    if file_info:
                        partition_files.append(file_info)
            
            logger.info(f"Found {len(partition_files)} partition files")
            return sorted(partition_files, key=lambda x: (x.get('group', 0), x.get('part', 0)))
            
        except ClientError as e:
            logger.error(f"Error listing S3 objects: {e}")
            raise
    
    def _extract_file_metadata(self, key: str, obj_info: Dict):
        """Extract metadata from partition filename."""
        filename = Path(key).name
        
        # Expected format: part_X_X.parquet 
        if filename.startswith('part_') and filename.endswith('.parquet'):
            try:
                parts = filename.replace('.parquet', '').split('_')
                if len(parts) == 3:
                    part_num = int(parts[1])
                    group_num = int(parts[2])
                    
                    return {
                        'key': key,
                        'filename': filename,
                        'part': part_num,
                        'group': group_num,
                        'size_bytes': obj_info['Size'],
                        'last_modified': obj_info['LastModified']
                    }
            except (ValueError, IndexError):
                raise ValueError(f"Could not parse filename: {filename}")

    def read_partition(self, key: str, columns: Optional[List[str]] = None):
        """Read a single partition file from S3."""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            parquet_data = response['Body'].read()
            
            df = pd.read_parquet(io.BytesIO(parquet_data), columns=columns)
            logger.debug(f"Loaded partition {key}: {len(df)} rows")
            return df
            
        except ClientError as e:
            logger.error(f"Error reading partition {key}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error parsing parquet file {key}: {e}")
            raise
    
    def stream(self, partition_files: List[Dict[str, Any]], 
                         columns: Optional[List[str]] = None):
        """Stream partitions one at a time to minimize memory usage."""
        for file_info in partition_files:
            key = file_info['key']
            logger.info(f"Streaming partition: {key}")
            
            try:
                df = self.read_partition(key, columns)
                yield df
            except Exception as e:
                logger.error(f"Failed to stream partition {key}: {e}")
                continue
    
    def get_partition_info(self, partition_files: List[Dict[str, Any]]):
        """Get summary information about partitions."""
        if not partition_files:
            return {}
        
        total_size = sum(f['size_bytes'] for f in partition_files)
        size_mb = total_size / (1024 * 1024)
        
        return {
            'total_partitions': len(partition_files),
            'total_size_mb': round(size_mb, 2),
            'avg_size_mb': round(size_mb / len(partition_files), 2),
            'date_range': {
                'earliest': min(f['last_modified'] for f in partition_files),
                'latest': max(f['last_modified'] for f in partition_files)
            }
        }
    
    def verify(self) -> bool:
        """Verify that bucket exists and is accessible."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Successfully connected to bucket: {self.bucket_name}")
            return True
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                logger.error(f"Bucket {self.bucket_name} not found")
            elif error_code == '403':
                logger.error(f"Access denied to bucket {self.bucket_name}")
            else:
                logger.error(f"Error accessing bucket {self.bucket_name}: {e}")
            return False
