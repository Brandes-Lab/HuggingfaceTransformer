# from tokenizers import Tokenizer
# from tokenizers.models import WordLevel
# from tokenizers.pre_tokenizers import Split
# from transformers import PreTrainedTokenizerFast

# # Define vocabulary: amino acids + special tokens
# amino_acids = list("ACDEFGHIKLMNPQRSTVWYBXZJUO-")
# special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

# # Create vocabulary dict
# vocab = {token: i for i, token in enumerate(special_tokens + amino_acids)}

# # Create the tokenizer
# tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
# tokenizer.pre_tokenizer = Split(pattern="", behavior="isolated")  # character-level

# # Wrap as a HuggingFace-compatible tokenizer
# hf_tokenizer = PreTrainedTokenizerFast(
#     tokenizer_object=tokenizer,
#     unk_token="[UNK]",
#     pad_token="[PAD]",
#     cls_token="[CLS]",
#     sep_token="[SEP]",
#     mask_token="[MASK]"
# )

# # Save to disk
# hf_tokenizer.save_pretrained("char_tokenizer")



from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from transformers import PreTrainedTokenizerFast

# Define vocabulary: ONLY standard 20 amino acids
amino_acids = list("ACDEFGHIKLMNPQRSTVWY")  # Standard 20 only
special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "-"]

# Build vocabulary
vocab = {token: i for i, token in enumerate(special_tokens + amino_acids)}

print(f"Vocab: {vocab}")
print(f"'-' ID: {vocab['-']}")

# Create tokenizer
tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
tokenizer.pre_tokenizer = Split(pattern="", behavior="isolated")

# Wrap it
hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]"
)

# Test
print("\nTests:")
print("'-' converts to:", hf_tokenizer.convert_tokens_to_ids("-"))
print("Tokenization of 'ACD-GH':", hf_tokenizer("ACD-GH"))
print("Decoding:", hf_tokenizer.decode([6, 7, 8, 5, 11, 12]))
print("Vocab size:", len(hf_tokenizer))

hf_tokenizer.save_pretrained("phylo_char_tokenizer")