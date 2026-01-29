#!/usr/bin/env python3
import argparse
import os

def split_tsv(tsv_path: str, out_dir: str, num_shards: int, lines_per_shard: int | None):
    os.makedirs(out_dir, exist_ok=True)

    with open(tsv_path, "r", encoding="utf-8", errors="replace") as fin:
        header = fin.readline()
        if not header:
            raise ValueError("Empty TSV (no header)")

        shard_idx = 0
        line_in_shard = 0
        fout = None

        def open_new_shard(i: int):
            path = os.path.join(out_dir, f"shard_{i:05d}.tsv")
            f = open(path, "w", encoding="utf-8")
            f.write(header)
            return f, path

        fout, shard_path = open_new_shard(shard_idx)
        print(f"Writing {shard_path}")

        for line in fin:
            if lines_per_shard is None:
                # Round-robin distribution (good for load balance)
                if line_in_shard > 0 and (line_in_shard % 1 == 0):
                    pass
                # Instead of round-robin within one file, we'll do simple modulo by global line count:
                # We'll rewrite as: close/open based on modulo would be messy.
                # So for round-robin we prefer fixed num_shards and write to many open files.
                raise ValueError("lines_per_shard must be set for this splitter. Use --lines_per_shard.")
            else:
                # Fixed chunk size splitting
                if line_in_shard >= lines_per_shard:
                    fout.close()
                    shard_idx += 1
                    if shard_idx >= num_shards:
                        # if user gave both, stop at num_shards
                        break
                    fout, shard_path = open_new_shard(shard_idx)
                    print(f"Writing {shard_path}")
                    line_in_shard = 0

                fout.write(line)
                line_in_shard += 1

        if fout:
            fout.close()

    print("Done splitting.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_shards", type=int, required=True, help="How many shard files to produce (max).")
    ap.add_argument("--lines_per_shard", type=int, required=True, help="Lines per shard (excluding header).")
    args = ap.parse_args()

    split_tsv(args.tsv_path, args.out_dir, args.num_shards, args.lines_per_shard)

if __name__ == "__main__":
    main()
