import parasail 

_matrix = parasail.blosum62

def align_pair(seq1, seq2): 
    result = parasail.nw_trace_scan_16(seq1, seq2, 10, 1, _matrix)
    # trace = parasail.traceback(result, seq1, seq2)
    tb = result.traceback

    # Get aligned sequences
    a1 = tb.query   # contains '-' for gaps
    a2 = tb.ref     # contains '-' for gaps

    return a1, a2

def percent_identity(a1, a2):
    """
    Compute percent identity between two aligned sequences (strings already containing -).
    """
    matches = 0
    aligned = 0
    for x, y in zip(a1, a2):
        if x == "<PAD>" or y == "<PAD>":
            continue
        aligned += 1
        if x == y:
            matches += 1
    if aligned == 0:
        return 0.0
    return 100.0 * matches / aligned
    