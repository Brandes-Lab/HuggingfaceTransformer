import sys, os, csv, time, argparse
import xxhash
from datasets import Dataset, Features, Value, Sequence

max_int = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        max_int = int(max_int / 10)

TSV_PATH = "/gpfs/data/brandeslab/Data/uniref/uniref90_cluster_members_ge2_bk_test.tsv"
OUT_BASE = "/gpfs/data/brandeslab/Data/uniref/uniref90_clusters_arrow"
PREFIX   = "UniRef100_"

# 0.19% test (0.01% resolution via mod 10000 buckets) -> ~101k clusters
TEST_BUCKETS = 19
TOTAL_ROWS   = 53_506_765  # clusters (lines minus header), only used for % logs
LOG_EVERY    = 100_000

def is_test(cluster_id: str) -> bool:
    return (xxhash.xxh64(cluster_id).intdigest() % 10_000) < TEST_BUCKETS

features = Features({
    "cluster_id": Value("string"),
    "member_ids": Sequence(Value("string")),
})

def make_generator(which: str):
    assert which in ("train", "test")
    t0 = time.time()
    seen = kept = 0

    with open(TSV_PATH, "rt", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            seen += 1
            if seen % LOG_EVERY == 0:
                dt = time.time() - t0
                rate = seen / dt if dt > 0 else 0.0
                print(
                    f"[{which}] seen={seen:,} ({100*seen/TOTAL_ROWS:.2f}%) kept={kept:,} rate={rate:,.0f} rows/s",
                    flush=True
                )

            cid = row["cluster_id"]
            in_test = is_test(cid)

            # Split filter:
            if which == "test" and not in_test:
                continue
            if which == "train" and in_test:
                continue

            mids = row["member_ids"].split(",")
            kept += 1
            yield {"cluster_id": cid, "member_ids": [PREFIX + m for m in mids]}

    print(f"[{which}] DONE seen={seen:,} kept={kept:,}", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--out_dir", default=OUT_BASE, help="Base output directory")
    args = ap.parse_args()

    out = os.path.join(args.out_dir, args.split)
    os.makedirs(out, exist_ok=True)

    ds = Dataset.from_generator(
        make_generator,
        gen_kwargs={"which": args.split},
        features=features,
        writer_batch_size=10_000
    )
    ds.save_to_disk(out)
    print("Saved", args.split, "to", out, flush=True)

if __name__ == "__main__":
    main()