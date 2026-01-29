#!/usr/bin/env python3
import argparse
import csv
import sys
import xml.etree.ElementTree as ET
from typing import Optional, List


def strip_prefixes(s: str) -> str:
    # Remove either prefix if present
    if s.startswith("UniRef90_"):
        return s[len("UniRef90_") :]
    if s.startswith("UniRef100_"):
        return s[len("UniRef100_") :]
    return s


def localname(tag: str) -> str:
    # Handle XML namespaces like "{ns}entry"
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def get_member_count(entry_elem: ET.Element) -> Optional[int]:
    # <property type="member count" value="2"/>
    for child in entry_elem:
        if localname(child.tag) != "property":
            continue
        if child.attrib.get("type") == "member count":
            val = child.attrib.get("value")
            if val is None:
                return None
            try:
                return int(val)
            except ValueError:
                return None
    return None


def get_uniref100_id_from_dbref(dbref_elem: ET.Element) -> Optional[str]:
    # dbReference contains <property type="UniRef100 ID" value="UniRef100_..."/>
    for prop in dbref_elem:
        if localname(prop.tag) != "property":
            continue
        if prop.attrib.get("type") == "UniRef100 ID":
            return prop.attrib.get("value")
    return None


def collect_member_uniref100_ids(entry_elem: ET.Element) -> List[str]:
    """
    Collects a UniRef100 ID for:
      - representativeMember/dbReference
      - each member/dbReference
    Returns list length expected to match "member count" if XML is consistent.
    """
    ids: List[str] = []

    # representativeMember
    for child in entry_elem:
        if localname(child.tag) == "representativeMember":
            # It typically has a dbReference child
            for rep_child in child:
                if localname(rep_child.tag) == "dbReference":
                    uid = get_uniref100_id_from_dbref(rep_child)
                    if uid:
                        ids.append(uid)
                    break
            break

    # member elements
    for child in entry_elem:
        if localname(child.tag) != "member":
            continue
        for mem_child in child:
            if localname(mem_child.tag) == "dbReference":
                uid = get_uniref100_id_from_dbref(mem_child)
                if uid:
                    ids.append(uid)
                break

    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True, help="Path to uniref90.xml")
    ap.add_argument("--out", required=True, help="Path to output TSV")
    ap.add_argument(
        "--on-mismatch",
        choices=["skip", "keep"],
        default="skip",
        help="If len(member_ids)!=member_count before dedup: skip entry or keep anyway",
    )
    args = ap.parse_args()

    # Streaming parse
    context = ET.iterparse(args.xml, events=("end",))

    with open(args.out, "w", newline="") as f_out:
        writer = csv.writer(f_out, delimiter="\t")
        writer.writerow(["cluster_id", "member_count", "member_ids"])

        for event, elem in context:
            if localname(elem.tag) != "entry":
                continue

            entry_id = elem.attrib.get("id")
            if not entry_id:
                elem.clear()
                continue

            member_count = get_member_count(elem)
            if member_count is None:
                print(f"[WARN] No member count for entry {entry_id}", file=sys.stderr)
                elem.clear()
                continue

            # Only consider entries with member_count >= 2
            if member_count < 2:
                elem.clear()
                continue

            # Collect per-member UniRef100 IDs (including representative)
            raw_ids = collect_member_uniref100_ids(elem)

            # Validate: list length matches member_count (pre-dedup)
            if len(raw_ids) != member_count:
                msg = (
                    f"[WARN] member_count mismatch for {entry_id}: "
                    f"member_count={member_count}, collected={len(raw_ids)}"
                )
                print(msg, file=sys.stderr)
                if args.on_mismatch == "skip":
                    elem.clear()
                    continue

            # Augment member IDs by prefixing with cluster core (stripped)
            # cluster_core = strip_prefixes(entry_id)
            augmented = []
            for uid in raw_ids:
                member_core = strip_prefixes(uid)
                augmented.append(member_core)

            # Dedup and update member_count
            # (Stable order while deduping)
            seen = set()
            dedup_augmented = []
            for x in augmented:
                if x in seen:
                    continue
                seen.add(x)
                dedup_augmented.append(x)

            new_count = len(dedup_augmented)

            # If after dedup count < 2, drop the row
            if new_count < 2:
                elem.clear()
                continue

            writer.writerow([entry_id, new_count, ",".join(dedup_augmented)])

            # Free memory
            elem.clear()

    print(f"Done. Wrote: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
