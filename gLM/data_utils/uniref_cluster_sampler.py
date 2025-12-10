import numpy as np
import pyarrow.parquet as pq 
from datasets import Dataset

# class RandomClusterSampler:
#     def __init__(self, parquet_path):
#         self.path = parquet_path
#         self.meta = pq.read_metadata(parquet_path)
#         self.num_row_groups = self.meta.num_row_groups 

#     def sample_clusters(self):
#         # randomly select a row group (multiple clusters per row group)
#         rg = np.random.randint(0, self.num_row_groups)

#         # read only that row group 
#         # table = pq.read_table(self.path, row_groups=[rg])
#         pf = pq.ParquetFile(self.path)
#         table = pf.read_row_group(rg)
#         n = table.num_rows # number of clusters in that row group

#         # Pick random row (cluster) from the row group
#         cluster_idx = np.random.randint(0, n)
        
#         row = table.slice(cluster_idx, 1)
#         cluster_id = row["cluster_id"][0].as_py()
#         rep_id = row["representative_id"][0].as_py()
#         members = row["members"][0].as_py()

#         return cluster_id, members


import numpy as np
import pyarrow.parquet as pq 

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
