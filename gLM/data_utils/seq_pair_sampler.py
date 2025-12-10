import random

def pick_pairs(rep, members):
    # members list does NOT include rep
    ids = [rep] + members

    # Case 1: cannot make a pair
    if len(ids) < 2:
        return None

    # Case 2: exactly two sequences → deterministic
    if len(ids) == 2:
        return ids[0], ids[1]

    # Case 3: randomly pick two out of many
    return tuple(random.sample(ids, 2))
