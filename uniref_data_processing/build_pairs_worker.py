#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import os
import sys
import random
from pathlib import Path
from typing import List

from Bio import SeqIO

csv.field_size_limit(sys.maxsize)

# TSV column names
COL_CLUSTER_ID = "cluster_id"
COL_MEMBER_CNT = "member_count"
COL_MEMBER_IDS = "member_ids"

# Split fractions
TRAIN_FRAC = 0.90
VAL_FRAC   = 0.05
TEST_FRAC  = 0.05

def parse_member_ids(member_ids_str: str) -> List[str]:
    return [x.strip() for x in member_ids_str.split(",") if x.strip()]

def candidate_ids_from_row(row: dict) -> List[str]:
    cands: List[str] = []
    cands.extend(parse_member_ids(row[COL_MEMBER_IDS]))

    # de-dup preserve order
    return list(dict.fromkeys([c for c in cands if c]))

def try_get_record(idx, raw_id: str):
    raw_id = raw_id.strip()
    if raw_id.startswith("UniRef"):
        candidates = (raw_id,)
    else:
        candidates = (raw_id, f"UniRef100_{raw_id}", f"UniRef90_{raw_id}")
    for cid in candidates:
        rec = idx.get(cid)
        if rec is not None:
            return rec
    return None

def split_from_cluster(cluster_id: str) -> str:
    h = hashlib.md5(cluster_id.encode()).hexdigest()
    x = (int(h, 16) % 1_000_000) / 1_000_000.0
    if x < TRAIN_FRAC:
        return "train"
    elif x < TRAIN_FRAC + VAL_FRAC:
        return "val"
    else:
        return "test"

def pair_key_tuple(a: str, b: str):
    return (a, b) if a <= b else (b, a)

def pair_key_hash64(a: str, b: str) -> int:
    x, y = pair_key_tuple(a, b)
    h = hashlib.blake2b(f"{x}|{y}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_tsv", required=True)
    ap.add_argument("--index_db", required=True)
    ap.add_argument("--out_dir", required=True, help="Base out dir for shard outputs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--p_select", type=float, default=0.02)
    ap.add_argument("--use_hashed_pair_keys", action="store_true", default=True)
    ap.add_argument("--max_rows", type=int, default=0, help="0 = no limit (debug)")

    args = ap.parse_args()

    shard_name = Path(args.shard_tsv).stem  # shard_00012
    shard_out = Path(args.out_dir) / shard_name
    shard_out.mkdir(parents=True, exist_ok=True)

    train_out = shard_out / "train.jsonl"
    val_out   = shard_out / "validation.jsonl"
    test_out  = shard_out / "test.jsonl"

    rng = random.Random(args.seed ^ (hash(shard_name) & 0xFFFFFFFF))

    idx = SeqIO.index_db(args.index_db)

    # shard-local dedupe to reduce pointless writes
    seen_pairs_local = set()

    skipped = {
        "coin": 0,
        "bad_row": 0,
        "too_few_candidates": 0,
        "missing_seq": 0,
        "dup_pair_local": 0,
    }
    wrote = {"train": 0, "val": 0, "test": 0}

    with open(args.shard_tsv, newline="") as fin, \
         open(train_out, "w") as f_train, open(val_out, "w") as f_val, open(test_out, "w") as f_test:

        reader = csv.DictReader(fin, delimiter="\t")
        for col in (COL_CLUSTER_ID, COL_MEMBER_IDS):
            if col not in (reader.fieldnames or []):
                raise ValueError(f"Missing required column '{col}'. Found: {reader.fieldnames}")

        rows_seen = 0
        for row in reader:
            rows_seen += 1
            if args.max_rows and rows_seen > args.max_rows:
                break

            if rng.random() > args.p_select:
                skipped["coin"] += 1
                continue

            try:
                cands = candidate_ids_from_row(row)
            except Exception:
                skipped["bad_row"] += 1
                continue

            if len(cands) < 2:
                skipped["too_few_candidates"] += 1
                continue

            raw1, raw2 = rng.sample(cands, 2)

            rec1 = try_get_record(idx, raw1)
            rec2 = try_get_record(idx, raw2)
            if rec1 is None or rec2 is None:
                skipped["missing_seq"] += 1
                continue

            id1, id2 = rec1.id, rec2.id

            pk_int = pair_key_hash64(id1, id2) if args.use_hashed_pair_keys else None
            pk = str(pk_int) if pk_int is not None else json.dumps(pair_key_tuple(id1, id2))

            if pk in seen_pairs_local:
                skipped["dup_pair_local"] += 1
                continue
            seen_pairs_local.add(pk)

            example = {
                "pair_key": pk,
                "cluster_id": row.get(COL_CLUSTER_ID),
                "member_count": (
                    int(row[COL_MEMBER_CNT])
                    if (COL_MEMBER_CNT in row and row[COL_MEMBER_CNT] not in (None, ""))
                    else None
                ),
                "raw_id1": raw1,
                "raw_id2": raw2,
                "id1": id1,
                "id2": id2,
                "seq1": str(rec1.seq),
                "seq2": str(rec2.seq),
            }

            split = split_from_cluster(row[COL_CLUSTER_ID])
            line = json.dumps(example) + "\n"
            if split == "train":
                f_train.write(line); wrote["train"] += 1
            elif split == "val":
                f_val.write(line); wrote["val"] += 1
            else:
                f_test.write(line); wrote["test"] += 1

    print(f"[{shard_name}] wrote train={wrote['train']:,} val={wrote['val']:,} test={wrote['test']:,}")
    print(f"[{shard_name}] skipped: {skipped}")
    print(f"[{shard_name}] local unique pairs: {len(seen_pairs_local):,}")

if __name__ == "__main__":
    main()
