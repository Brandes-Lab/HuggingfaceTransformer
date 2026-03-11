from torch.utils.data import Dataset
from datasets import load_from_disk
from gLM.sequences.pairwise_align import align_pair, percent_identity
import random
import lmdb


class Uniref90ArrowDatasetForLMDB(Dataset):
    """
    Map-style dataset backed by a HuggingFace dataset saved via save_to_disk().
    Each row is a UniRef90 cluster with:
      - "cluster_id": str
      - "member_ids": List[str]  (already prefixed with 'Uniref100_')
    Sequences are fetched on-demand from an LMDB (key=member_id, value=sequence bytes).
    """
    
    def __init__(self, dataset_path: str, training_type: str, lmdb_path: str):
        super().__init__()
        self.training_type = training_type
        self.dataset = load_from_disk(dataset_path)
        self.lmdb_path = lmdb_path
        self._env = None 
    
    def _get_env(self):
        if self._env is None:
            self._env = lmdb.open(
                self.lmdb_path, 
                readonly=True, 
                lock=False, 
                subdir=True)
        return self._env

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def _fetch_seq(txn, member_id:str) -> str:
        v = txn.get(member_id.encode('utf-8'))
        if v is None:
            raise KeyError(f"Sequence ID {member_id} not found in LMDB")
        return v.decode("ascii", errors="strict")
    
    def __getitem__(self, idx):
        """
        Sample a random cluster globally 
        Sample two distinct member_ids from that cluster
        Fetch sequences from LMDB
        Returns data in the format expected by PhyloCollator:
        - MLM: single string
        - phylo_encoder_only: (a1, a2, pid) tuple
        - phylo_encoder_decoder: (s1, s2) tuple
        """
        # Ignore idx and sample a random cluster globally
        ridx = random.randrange(len(self.dataset))
        ex = self.dataset[ridx]
        member_ids = ex["member_ids"]
        if not member_ids or len(member_ids) < 2:
            # Skip this example by trying the next one
            return self.__getitem__(idx)
        
        m1, m2 = random.sample(member_ids, 2)

        env = self._get_env()
        with env.begin(write=False) as txn:
            try:
                s1 = self._fetch_seq(txn, m1)
                s2 = self._fetch_seq(txn, m2)
            except KeyError:
                # Skip this example by trying the next one
                return self.__getitem__(idx)
        
        if self.training_type == "MLM":
            return s1 if random.random() < 0.5 else s2
            
        elif self.training_type == "phylo_encoder_only":
            a1, a2 = align_pair(s1, s2)
            if len(a1) != len(a2):
                # Skip this example by trying the next one
                return self.__getitem__(idx)
            pid = percent_identity(a1, a2)
            return (a1, a2, pid)
            
        elif self.training_type == "phylo_encoder_decoder":
            return (s1, s2)
        

# -------------------------
# Deterministic eval dataset
# -------------------------
from torch.utils.data import Dataset
from datasets import load_from_disk
import random
import lmdb

class Uniref90ArrowEvalDatasetForLMDB(Dataset):
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
        lmdb_path: str,
        max_tries: int = 30,
        deterministic_pair: bool = True,
    ):
        super().__init__()
        self.training_type = training_type
        self.dataset = load_from_disk(dataset_path)
        self.lmdb_path = lmdb_path
        self._env = None
        self.max_tries = max_tries
        self.deterministic_pair = deterministic_pair

    def _get_env(self):
        if self._env is None:
            self._env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                subdir=True,
                readahead=False,
            )
        return self._env

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def _fetch_seq(txn, member_id: str) -> str:
        v = txn.get(member_id.encode("utf-8"))
        if v is None:
            raise KeyError(f"Sequence ID {member_id} not found in LMDB")
        return v.decode("ascii", errors="strict")

    def __getitem__(self, idx):
        ex = self.dataset[idx]
        member_ids = ex["member_ids"]

        if not member_ids or len(member_ids) < 2:
            raise RuntimeError(f"Cluster at idx={idx} has fewer than 2 members")

        env = self._get_env()

        if self.deterministic_pair:
            m1, m2 = member_ids[0], member_ids[1]

            with env.begin(write=False) as txn:
                try:
                    s1 = self._fetch_seq(txn, m1)
                    s2 = self._fetch_seq(txn, m2)
                except KeyError as e:
                    raise RuntimeError(
                        f"Deterministic pair ({m1}, {m2}) missing from LMDB for idx={idx}"
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
                return (s1, s2)

            else:
                raise ValueError(f"Unknown training_type: {self.training_type}")

        else:
            for _ in range(self.max_tries):
                m1, m2 = random.sample(member_ids, 2)

                with env.begin(write=False) as txn:
                    try:
                        s1 = self._fetch_seq(txn, m1)
                        s2 = self._fetch_seq(txn, m2)
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

            raise RuntimeError(
                f"Failed to get a valid evaluation sample for idx={idx} after {self.max_tries} tries"
            )