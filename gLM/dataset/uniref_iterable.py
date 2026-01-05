import torch
from torch.utils.data import IterableDataset
import random
from gLM.data_utils.uniref_cluster_sampler import RandomClusterSampler
from gLM.data_utils.uniref_cluster_sampler import InMemoryClusterSampler
from gLM.data_utils.uniref_cluster_sampler import CachedRowGroupClusterSampler
from gLM.data_utils.seq_pair_sampler import pick_pairs
from gLM.sequences.seq_fetcher import SequenceFetcher
from gLM.sequences.pairwise_align import align_pair, percent_identity
from gLM.tokenizers.phylo_tokenizer import PhyloTokenizerLoader
from torch.utils.data import get_worker_info
import os


class UniRefClusterIterableDataset(IterableDataset):
    def __init__(self, parquet_path, index_db_path, fasta_path, tokenizer, max_seq_len, training_type, batch_size):
        super().__init__()
        # self.sampler = RandomClusterSampler(parquet_path)
        # self.sampler = InMemoryClusterSampler(parquet_path)
        # self.sampler = CachedRowGroupClusterSampler(parquet_path)
        self.parquet_path = parquet_path
        self.fasta_path = fasta_path
        self.index_db_path = index_db_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.training_type = training_type
        self.batch_size = batch_size
        self.fetcher = None # Will be initialized per worker
    
    def init_worker(self):
        self.fetcher = SequenceFetcher(self.fasta_path, self.index_db_path)

        worker_info = get_worker_info()
        seed = worker_info.id if worker_info else 0
        random.seed(seed)

        # self.sampler = InMemoryClusterSampler(self.parquet_path)
        self.sampler = CachedRowGroupClusterSampler(self.parquet_path)

    def __iter__(self):
        # worker_info = get_worker_info()
        # if worker_info is not None:
        #     print(f"[Worker {worker_info.id}/{worker_info.num_workers} | PID {os.getpid()}] Starting iterator")
        if self.fetcher is None:
            self.init_worker()

        while True:
            # MLM 
            if self.training_type == "MLM":
                raw_seqs = []

                while len(raw_seqs) < self.batch_size:
                    # cluster = self.sampler.sample_clusters()
                    cluster = self.sampler.sample_cluster()
                    rep = cluster["representative_id"]
                    members = cluster["members"]

                    # skip empty clusters
                    if not members:
                        continue

                    seq_id = random.choice(members + [rep])
                    
                    try:
                        seq = self.fetcher(seq_id)
                    except KeyError:
                        continue

                    seq = seq[:self.max_seq_len]
                    raw_seqs.append(seq)
                
                # tokenize batch
                tokenized_batch = self.tokenizer(
                    raw_seqs,
                    padding='longest',
                    truncation=True,
                    max_length=self.max_seq_len,
                    return_tensors='pt'
                )

                for i in range(self.batch_size):
                    item = {k: v[i] for k, v in tokenized_batch.items()}
                    yield item

            # Phylo 
            elif self.training_type == "phylo":
                aligned_pairs = []
                percent_ids = []

                while len(aligned_pairs) < self.batch_size:
                    # cluster = self.sampler.sample_clusters()
                    cluster = self.sampler.sample_cluster()
                    rep = cluster["representative_id"]
                    members = cluster["members"]
                    # skip empty clusters
                    if not members:
                        continue
                    pair = pick_pairs(rep, members)
                    if pair is None:
                        continue
                    s1_id, s2_id = pair
                    try:
                        s1 = self.fetcher(s1_id)
                        s2 = self.fetcher(s2_id)
                    except KeyError:
                        continue
                    a1, a2 = align_pair(s1, s2)
                    if len(a1) != len(a2):
                        continue
                    aligned_pairs.append((a1, a2))
                    pid = percent_identity(a1, a2)
                    percent_ids.append(pid)

                # Tokenize batch
                encoded_batch = self.tokenizer.batch_encode_aligned(
                aligned_pairs,
                max_length=self.max_seq_len,
                )

                for enc, pid in zip(encoded_batch, percent_ids):
                    enc["percent_identity"] = pid
                    yield enc
                    
