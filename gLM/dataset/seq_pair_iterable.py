import json
import random
from typing import Iterator, Tuple, Optional, List

import torch
from torch.utils.data import IterableDataset, get_worker_info

from gLM.sequences.pairwise_align import align_pair, percent_identity


class SeqPairIterableDataset(IterableDataset):
    """
    Reads jsonl where each line includes at least:
      - "seq1": str
      - "seq2": str

    Behavior preserved:
      - training_type == "MLM":
          return  single raw sequence string (randomly s1 or s2)
      - training_type == "phylo_encoder_only":
          aligns each pair
          returns single (a1, a2, pid) tuple
      - else (e.g. "phylo_encoder_decoder"):
        returns single input-target pair (s1, s2)
\    """

    def __init__(self, dataset_path: str, tokenizer, training_type: str, shuffle_buffer: int = 0):
        super().__init__()
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.training_type = training_type
        self.shuffle_buffer = shuffle_buffer  # 0 disables; >0 enables light streaming shuffle per worker

    def _line_iterator(self) -> Iterator[str]:
        """
        Worker-sharded streaming: each worker reads the same file but only consumes
        lines where (line_idx % num_workers == worker_id). This avoids duplicate work.
        """
        wi = get_worker_info()
        worker_id = wi.id if wi else 0
        num_workers = wi.num_workers if wi else 1

        with open(self.dataset_path, "r") as f:
            for i, line in enumerate(f):
                if (i % num_workers) != worker_id:
                    continue
                line = line.strip()
                if line:
                    yield line

    def _stream_examples(self) -> Iterator[Tuple[str, str]]:
        """
        Yields (s1, s2) strings only. Optionally does a small buffer shuffle to avoid
        strictly sequential ordering without loading the full dataset.
        """
        it = self._line_iterator()

        if self.shuffle_buffer and self.shuffle_buffer > 1:
            buf: List[Tuple[str, str]] = []
            # fill buffer
            for _ in range(self.shuffle_buffer):
                try:
                    line = next(it)
                except StopIteration:
                    break
                ex = json.loads(line)
                s1 = ex.get("seq1")
                s2 = ex.get("seq2")
                if isinstance(s1, str) and isinstance(s2, str) and s1 and s2:
                    buf.append((s1, s2))

            # stream with replacement
            while buf:
                j = random.randrange(len(buf))
                yield buf[j]
                try:
                    line = next(it)
                except StopIteration:
                    buf.pop(j)
                    continue
                ex = json.loads(line)
                s1 = ex.get("seq1")
                s2 = ex.get("seq2")
                if isinstance(s1, str) and isinstance(s2, str) and s1 and s2:
                    buf[j] = (s1, s2)
                else:
                    buf.pop(j)
        else:
            for line in it:
                ex = json.loads(line)
                s1 = ex.get("seq1")
                s2 = ex.get("seq2")
                if isinstance(s1, str) and isinstance(s2, str) and s1 and s2:
                    yield s1, s2

    def __iter__(self):
        while True:
            stream = self._stream_examples()

            for s1, s2 in stream:
                if self.training_type == "MLM": # yield single raw sequence
                    # pick s1 or s2 randomly
                    # s1, s2 = next(stream)
                    if random.random() < 0.5:
                        yield s1 
                    else:
                        yield s2
                elif self.training_type == "phylo_encoder_only":
                    # align and yield
                    # s1, s2 = next(stream)
                    a1, a2 = align_pair(s1, s2)
                    if len(a1) != len(a2):
                        continue
                    pid = percent_identity(a1, a2)
                    yield (a1, a2, pid) # yield aligned pair and pid

                elif self.training_type == "phylo_encoder_decoder":
                    # s1, s2 = next(stream)
                    # print(len(s1), len(s2))
                    yield (s1, s2) # yield input-target pair

