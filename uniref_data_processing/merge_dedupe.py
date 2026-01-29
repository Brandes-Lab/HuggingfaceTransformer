#!/usr/bin/env python3
import argparse
import glob
import json
import os
from pathlib import Path

def merge_one_split(files, out_path: str, cap: int | None):
    seen = set()
    wrote = 0

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fout:
        for fp in files:
            with open(fp, "r") as fin:
                for line in fin:
                    if cap is not None and wrote >= cap:
                        return wrote, len(seen)

                    try:
                        ex = json.loads(line)
                    except Exception:
                        continue

                    pk = ex.get("pair_key")
                    if pk is None:
                        # if somehow missing, fallback to id1/id2
                        id1 = ex.get("id1"); id2 = ex.get("id2")
                        if id1 is None or id2 is None:
                            continue
                        pk = f"{min(id1,id2)}|{max(id1,id2)}"

                    if pk in seen:
                        continue
                    seen.add(pk)

                    fout.write(json.dumps(ex) + "\n")
                    wrote += 1

    return wrote, len(seen)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sharded_out_base", required=True, help="OUT_BASE used by workers")
    ap.add_argument("--final_out_dir", required=True)
    ap.add_argument("--max_train", type=int, default=1_000_000)
    ap.add_argument("--max_val", type=int, default=50_000)
    ap.add_argument("--max_test", type=int, default=50_000)
    args = ap.parse_args()

    base = args.sharded_out_base.rstrip("/")

    train_files = sorted(glob.glob(os.path.join(base, "shard_*", "train.jsonl")))
    val_files   = sorted(glob.glob(os.path.join(base, "shard_*", "validation.jsonl")))
    test_files  = sorted(glob.glob(os.path.join(base, "shard_*", "test.jsonl")))

    print(f"Found train shards: {len(train_files)}")
    print(f"Found val shards:   {len(val_files)}")
    print(f"Found test shards:  {len(test_files)}")

    final_train = str(Path(args.final_out_dir) / "train.jsonl")
    final_val   = str(Path(args.final_out_dir) / "validation.jsonl")
    final_test  = str(Path(args.final_out_dir) / "test.jsonl")

    train_w, train_u = merge_one_split(train_files, final_train, args.max_train)
    val_w, val_u     = merge_one_split(val_files, final_val, args.max_val)
    test_w, test_u   = merge_one_split(test_files, final_test, args.max_test)

    print("\nDONE MERGE")
    print(f"Train: wrote={train_w:,} unique_seen={train_u:,} -> {final_train}")
    print(f"Val:   wrote={val_w:,} unique_seen={val_u:,} -> {final_val}")
    print(f"Test:  wrote={test_w:,} unique_seen={test_u:,} -> {final_test}")

if __name__ == "__main__":
    main()
