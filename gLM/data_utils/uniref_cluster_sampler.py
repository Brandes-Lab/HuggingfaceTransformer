import numpy as np
import pyarrow.parquet as pq 
from datasets import Dataset
import random


class RandomClusterSampler:
    def __init__(self, parquet_path):
        self.path = parquet_path
        
        # Read metadata once
        self.meta = pq.read_metadata(parquet_path)
        self.num_row_groups = self.meta.num_row_groups 
        
        # Open ParquetFile once 
        self.pf = pq.ParquetFile(parquet_path)

    def sample_clusters(self):
        # randomly select a row group (multiple clusters per row group)
        rg = np.random.randint(0, self.num_row_groups)

        # read only that row group 
        table = self.pf.read_row_group(rg)
        n = table.num_rows

        # Pick random row (cluster) from the row group
        cluster_idx = np.random.randint(0, n)
        row = table.slice(cluster_idx, 1)

        cluster_id = row["cluster_id"][0].as_py()
        rep_id      = row["representative_id"][0].as_py()
        members     = row["members"][0].as_py()

        return {
            "cluster_id": cluster_id,
            "representative_id": rep_id,
            "members": members
        }



class CachedRowGroupClusterSampler:
    def __init__(self, parquet_path):
        self.pf = pq.ParquetFile(parquet_path)
        self.num_row_groups = self.pf.num_row_groups
        self.row_group_cache = []
        self.current_rg_index = None

    def sample_cluster(self):
        # If cache is empty, load a new row group
        if not self.row_group_cache:
            # Choose a random row group
            rg_idx = random.randint(0, self.num_row_groups - 1)
            table = self.pf.read_row_group(rg_idx)
            rows = table.to_pylist()  # 1840 rows typically
            random.shuffle(rows)  # Shuffle to maintain randomness
            self.row_group_cache.extend(rows)
            self.current_rg_index = rg_idx

        # Pop one sample from cache
        row = self.row_group_cache.pop()
        return {
            "cluster_id": row["cluster_id"],
            "representative_id": row["representative_id"],
            "members": row["members"],
        }
 