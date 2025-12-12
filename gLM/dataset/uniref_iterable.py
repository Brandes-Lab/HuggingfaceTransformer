import torch
from torch.utils.data import IterableDataset
import random
from gLM.data_utils.uniref_cluster_sampler import RandomClusterSampler
from gLM.data_utils.seq_pair_sampler import pick_pairs
from gLM.sequences.seq_fetcher import SequenceFetcher
from gLM.sequences.pairwise_align import align_pair, percent_identity
from gLM.tokenizers.phylo_tokenizer import PhyloTokenizerLoader


class UniRefClusterIterableDataset(IterableDataset):
    def __init__(self, parquet_path, index_db_path, fasta_path, tokenizer, max_seq_len, training_type):
        super().__init__()
        self.sampler = RandomClusterSampler(parquet_path)
        self.fasta_path = fasta_path
        self.index_db_path = index_db_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.training_type = training_type
    
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
            
            # Phylo modeling
            if self.training_type == "phylo":
                print("In the Uniref Iterable - Phylo branch")
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
                

                # align sequences
                a1, a2 = align_pair(s1, s2)
                # Replace all '-' with '[GAP]'
                a1 = a1.replace("-", "[GAP]")
                a2 = a2.replace("-", "[GAP]")

                tokens1 = self.tokenizer.tokenize(a1)
                tokens2 = self.tokenizer.tokenize(a2)

                assert len(tokens1) == len(tokens2), "Alignment mismatch after tokenization"

                trunc_len = min(self.max_seq_len, len(tokens1))
                tokens1 = tokens1[:trunc_len]
                tokens2 = tokens2[:trunc_len]

                a1 = ''.join(tokens1)
                a2 = ''.join(tokens2)

                # compute percent identity
                pid = percent_identity(a1, a2)
                # print("PID:", pid)

                # tokenize
                item = self.tokenizer.encode_aligned(a1, a2)

                item["percent_identity"] = pid


            elif self.training_type == "MLM":
                # print("In the Uniref Iterable - MLM branch")
                seq_id = random.choice(members + [rep])
                
                try:
                    seq = fetcher(seq_id)
                    # print(f'Fetched sequence for ID {seq_id}:, length: {len(seq)}')
                except KeyError:
                    continue

                seq = seq[:self.max_seq_len]
                # print(f'Truncated sequence for ID {seq_id}:, length: {len(seq)}')

                # tokenize 
                encoded = self.tokenizer(
                    seq, 
                    padding='longest',
                    truncation=True, 
                    max_length=self.max_seq_len,
                    return_tensors='pt'
                )
                # Squeeze out the extra batch dimension
                item = {k: v.squeeze(0) for k, v in encoded.items()}
            
            yield item
