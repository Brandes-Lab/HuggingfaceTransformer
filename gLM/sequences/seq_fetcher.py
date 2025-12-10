from Bio import SeqIO

class SequenceFetcher:
    def __init__(self, fasta_path, index_db_path):
        # index_db: reuses existing .idx file, avoids rebuilding
        self.index = SeqIO.index_db(index_db_path, [fasta_path], "fasta")

    def __call__(self, seq_id):
        # UniRef100 format requires prefix
        prefixed_seq_id = f"UniRef100_{seq_id}"
        record = self.index[prefixed_seq_id]
        return str(record.seq)
