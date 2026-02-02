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
          uses ONLY s1 (seq1), returns one item per sequence
      - training_type == "phylo_encoder_only":
          aligns s1/s2, returns tokenizer.batch_encode_aligned(...) items + percent_identity
      - else (e.g. "phylo_encoder_decoder"):
          returns {"input_ids","attention_mask","labels"} using s1 as input and s2 as target
    """

    def __init__(self, dataset_path: str, tokenizer, max_seq_len: int, training_type: str, batch_size: int,
                 shuffle_buffer: int = 0):
        super().__init__()
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.training_type = training_type
        self.batch_size = batch_size
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

        while True:  # infinite dataset

            stream = self._stream_examples()

            try:

                while True:

                    if self.training_type == "MLM":

                        raw_seqs = []

                        while len(raw_seqs) < self.batch_size:
                            s1, _ = next(stream)
                            raw_seqs.append(s1[: self.max_seq_len])

                        tokenized = self.tokenizer(
                            raw_seqs,
                            padding="longest",
                            truncation=True,
                            max_length=self.max_seq_len,
                            return_tensors="pt",
                        )

                        for i in range(self.batch_size):
                            yield {k: v[i] for k, v in tokenized.items()}

                    elif self.training_type == "phylo_encoder_only":

                        aligned_pairs = []
                        pids = []

                        while len(aligned_pairs) < self.batch_size:
                            s1, s2 = next(stream)

                            a1, a2 = align_pair(s1, s2)
                            if len(a1) != len(a2):
                                continue

                            aligned_pairs.append((a1, a2))
                            pids.append(percent_identity(a1, a2))

                        encoded_batch = self.tokenizer.batch_encode_aligned(
                            aligned_pairs,
                            max_length=self.max_seq_len,
                        )

                        for enc, pid in zip(encoded_batch, pids):
                            enc["percent_identity"] = pid
                            yield enc

                    else:

                        inputs = []
                        targets = []

                        while len(inputs) < self.batch_size:
                            s1, s2 = next(stream)
                            inputs.append(s1[: self.max_seq_len])
                            targets.append(s2[: self.max_seq_len])

                        enc = self.tokenizer(
                            inputs,
                            padding="longest",
                            truncation=True,
                            max_length=self.max_seq_len,
                            return_tensors="pt",
                        )

                        dec = self.tokenizer(
                            targets,
                            padding="longest",
                            truncation=True,
                            max_length=self.max_seq_len,
                            return_tensors="pt",
                        )

                        for i in range(self.batch_size):
                            yield {
                                "input_ids": enc["input_ids"][i],
                                "attention_mask": enc["attention_mask"][i],
                                "labels": dec["input_ids"][i],
                            }

            except StopIteration:
                # file ended → restart automatically
                continue

    
    
    # def __iter__(self):
    #     while True: # infinite dataset
    #         stream = self._stream_examples()

    #     while True:
    #         if self.training_type == "MLM":
    #             # Only s1 used
    #             raw_seqs: List[str] = []
    #             while len(raw_seqs) < self.batch_size:
    #                 s1, _ = next(stream)  # may raise StopIteration; handled below
    #                 raw_seqs.append(s1[: self.max_seq_len])

    #             tokenized = self.tokenizer(
    #                 raw_seqs,
    #                 padding="longest",
    #                 truncation=True,
    #                 max_length=self.max_seq_len,
    #                 return_tensors="pt",
    #             )

    #             # yield one item at a time (same as your current logic)
    #             # (views into the batch tensors; cheap)
    #             for i in range(self.batch_size):
    #                 yield {k: v[i] for k, v in tokenized.items()}

    #         elif self.training_type == "phylo_encoder_only":
    #             aligned_pairs: List[Tuple[str, str]] = []
    #             pids: List[float] = []

    #             while len(aligned_pairs) < self.batch_size:
    #                 s1, s2 = next(stream)
    #                 a1, a2 = align_pair(s1, s2)
    #                 if len(a1) != len(a2):
    #                     continue
    #                 aligned_pairs.append((a1, a2))
    #                 pids.append(percent_identity(a1, a2))

    #             encoded_batch = self.tokenizer.batch_encode_aligned(
    #                 aligned_pairs,
    #                 max_length=self.max_seq_len,
    #             )

    #             for enc, pid in zip(encoded_batch, pids):
    #                 enc["percent_identity"] = pid
    #                 yield enc

    #         else:
    #             # Encoder-decoder style: s1 input, s2 target
    #             inputs: List[str] = []
    #             targets: List[str] = []

    #             while len(inputs) < self.batch_size:
    #                 s1, s2 = next(stream)
    #                 inputs.append(s1[: self.max_seq_len])
    #                 targets.append(s2[: self.max_seq_len])

    #             enc = self.tokenizer(
    #                 inputs,
    #                 padding="longest",
    #                 truncation=True,
    #                 max_length=self.max_seq_len,
    #                 return_tensors="pt",
    #             )
    #             dec = self.tokenizer(
    #                 targets,
    #                 padding="longest",
    #                 truncation=True,
    #                 max_length=self.max_seq_len,
    #                 return_tensors="pt",
    #             )

    #             for i in range(self.batch_size):
    #                 yield {
    #                     "input_ids": enc["input_ids"][i],
    #                     "attention_mask": enc["attention_mask"][i],
    #                     "labels": dec["input_ids"][i],
    #                 }
