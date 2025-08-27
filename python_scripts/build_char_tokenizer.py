from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from transformers import PreTrainedTokenizerFast

# Define vocabulary: amino acids + special tokens
amino_acids = list("ACDEFGHIKLMNPQRSTVWYBXZJUO-")
special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

# Create vocabulary dict
vocab = {token: i for i, token in enumerate(special_tokens + amino_acids)}

# Create the tokenizer
tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
tokenizer.pre_tokenizer = Split(pattern="", behavior="isolated")  # character-level

# Wrap as a HuggingFace-compatible tokenizer
hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]"
)

# Save to disk
hf_tokenizer.save_pretrained("char_tokenizer")
