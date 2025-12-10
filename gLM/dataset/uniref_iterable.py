import torch
from torch.utils.data import IterableDataset
from gLM.data_utils.uniref_cluster_sampler import RandomClusterSampler
from gLM.data_utils.seq_pair_sampler import pick_pairs
from gLM.sequences.seq_fetcher import SequenceFetcher
from gLM.sequences.pairwise_align import align_pair, percent_identity
from gLM.tokenizers.phylo_tokenizer import PhyloTokenizerLoader


class UniRefClusterIterableDataset(IterableDataset):
    def __init__(self, parquet_path, index_db_path, fasta_path, tokenizer, max_seq_len):
        super().__init__()
        self.sampler = RandomClusterSampler(parquet_path)
        self.fasta_path = fasta_path
        self.index_db_path = index_db_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
    
    def __iter__(self):
        # Per-worker instance of FASTA fetcher
        fetcher = SequenceFetcher(self.fasta_path, self.index_db_path)

        # infinite generator
        while True:
            cluster = self.sampler.sample_clusters()
            rep = cluster["representative_id"]
            members = cluster["members"]

            # print("REP:", rep, "MEMBERS:", members, type(members))

            # skip empty clusters
            if not members:
                continue

            # pick a pair
            
            pair = pick_pairs(rep, members)
            if pair is None:
                continue
            seq1_id, seq2_id = pair
            # print("PAIR:", seq1_id, seq2_id)

            # fetch AA sequences
            try:
                s1 = fetcher(seq1_id)
                s2 = fetcher(seq2_id)
            except KeyError:
                # IDs missing in FASTA index → skip
                continue
            
            # print("SEQ1:", s1)
            # print(f"Len SEQ1: {len(s1)}")
            # print("SEQ2:", s2)
            # print(f"Len SEQ2: {len(s2)}")

            # align sequences
            a1, a2 = align_pair(s1, s2)
            # print("ALN1:", a1)
            # print(f"Len ALN1: {len(a1)}")
            # print("ALN2:", a2)
            # print(f"Len ALN2: {len(a2)}")

            # truncate
            if len(a1) > self.max_seq_len:
                a1 = a1[:self.max_seq_len]
            if len(a2) > self.max_seq_len:
                a2 = a2[:self.max_seq_len]
            
            # print("TRUNC1:", a1)
            # print(f"Len TRUNC1: {len(a1)}")
            # print("TRUNC2:", a2)
            # print(f"Len TRUNC2: {len(a2)}")

            # compute percent identity
            pid = percent_identity(a1, a2)
            # print("PID:", pid)

            # tokenize
            item = self.tokenizer.encode_aligned(a1, a2)
            # print("ENCODED ITEM TYPE:", type(item))
            # print("ENCODED ITEM:", item)
            item["percent_identity"] = pid
            
            yield item
