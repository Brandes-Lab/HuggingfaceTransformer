from torch.utils.data import Dataset
from datasets import load_from_disk
from gLM.sequences.pairwise_align import align_pair, percent_identity
from gLM.sequences.seq_fetcher import SequenceFetcher
import random


class Uniref90ArrowDatasetForFASTA(Dataset):
    def __init__(self, dataset_path: str, training_type: str, fasta_path: str, idx_db_path: str, max_tries: int = 30):
        super().__init__()
        self.training_type = training_type
        self.dataset = load_from_disk(dataset_path)
        self.fasta_path = fasta_path
        self.idx_db_path = idx_db_path
        self.max_tries = max_tries
        self._fetcher = None  # lazy per worker

    def _get_fetcher(self):
        if self._fetcher is None:
            self._fetcher = SequenceFetcher(self.fasta_path, self.idx_db_path)
        return self._fetcher

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        fetch = self._get_fetcher()

        for _ in range(self.max_tries):
            ridx = random.randrange(len(self.dataset))
            ex = self.dataset[ridx]
            # # One pass through the dataset in random order, but still sample two random members from each cluster
            # ex = self.dataset[idx]
            member_ids = ex["member_ids"]

            if not member_ids or len(member_ids) < 2:
                continue

            m1, m2 = random.sample(member_ids, 2)

            try:
                s1 = fetch(m1)
                s2 = fetch(m2)
            except KeyError:
                continue

            if self.training_type == "MLM":
                return s1 if random.random() < 0.5 else s2

            elif self.training_type == "phylo_encoder_only":
                a1, a2 = align_pair(s1, s2)
                if len(a1) != len(a2):
                    continue
                pid = percent_identity(a1, a2)
                return (a1, a2, pid)

            elif self.training_type == "phylo_encoder_decoder":
                return (s1, s2)

            else:
                raise ValueError(f"Unknown training_type: {self.training_type}")

        raise RuntimeError("Failed to sample a valid pair after max_tries (ID/index mismatch?).")
    

# -------------------------
# Deterministic eval dataset
# -------------------------
class Uniref90ArrowEvalDatasetForFASTA(Dataset):
    """
    Evaluation dataset for held-out UniRef90 clusters.

    Uses the provided index deterministically (one cluster per dataset index),
    unlike the training dataset which samples a random cluster each time.

    By default, it picks the first two members in each cluster for reproducibility.
    You can switch to random member sampling by setting deterministic_pair=False.
    """
    def __init__(
        self,
        dataset_path: str,
        training_type: str,
        fasta_path: str,
        idx_db_path: str,
        max_tries: int = 30,
        deterministic_pair: bool = True,
    ):
        super().__init__()
        self.training_type = training_type
        self.dataset = load_from_disk(dataset_path)
        self.fasta_path = fasta_path
        self.idx_db_path = idx_db_path
        self.max_tries = max_tries
        self.deterministic_pair = deterministic_pair
        self._fetcher = None

    def _get_fetcher(self):
        if self._fetcher is None:
            self._fetcher = SequenceFetcher(self.fasta_path, self.idx_db_path)
        return self._fetcher

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        fetch = self._get_fetcher()
        ex = self.dataset[idx]
        member_ids = ex["member_ids"]

        target_pair = {"UniRef100_UPI00164F0495", "UniRef100_UPI0022467D2D"} 
        

        if not member_ids or len(member_ids) < 2:
            raise RuntimeError(f"Cluster at idx={idx} has fewer than 2 members")

        if self.deterministic_pair:
            m1, m2 = member_ids[0], member_ids[1]
            # print(f"Deterministic pair for idx={idx}: {m1}, {m2}")
            
            if {m1, m2} == target_pair:
                print(f"Matched target pair at idx={idx}: {m1}, {m2}")

            try:
                s1 = fetch(m1)
                s2 = fetch(m2)
                if {m1, m2} == target_pair:
                    print(f"Successfully fetched target pair at idx={idx}")
            
            except KeyError as e:
                raise RuntimeError(
                    f"Deterministic pair failed for idx={idx}: ({m1}, {m2})"
                ) from e

            if self.training_type == "MLM":
                return s1

            elif self.training_type == "phylo_encoder_only":
                a1, a2 = align_pair(s1, s2)
                if len(a1) != len(a2):
                    raise RuntimeError(
                        f"Aligned lengths differ for deterministic pair at idx={idx}: ({m1}, {m2})"
                    )
                pid = percent_identity(a1, a2)
                return (a1, a2, pid)

            elif self.training_type == "phylo_encoder_decoder":
                if {m1, m2} == target_pair:
                    print(f"Returning target pair at idx={idx}")
                    print(f"s1: {s1[:50]}... (length {len(s1)})")
                    print(f"s2: {s2[:50]}... (length {len(s2)})")
                return (s1, s2)

            else:
                raise ValueError(f"Unknown training_type: {self.training_type}")

        else:
            for _ in range(self.max_tries):
                m1, m2 = random.sample(member_ids, 2)

                try:
                    s1 = fetch(m1)
                    s2 = fetch(m2)
                except KeyError:
                    continue

                if self.training_type == "MLM":
                    return s1

                elif self.training_type == "phylo_encoder_only":
                    a1, a2 = align_pair(s1, s2)
                    if len(a1) != len(a2):
                        continue
                    pid = percent_identity(a1, a2)
                    return (a1, a2, pid)

                elif self.training_type == "phylo_encoder_decoder":
                    return (s1, s2)

                else:
                    raise ValueError(f"Unknown training_type: {self.training_type}")

            raise RuntimeError(f"Failed to sample a valid pair for idx={idx}")
