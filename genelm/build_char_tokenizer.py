import argparse

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from transformers import PreTrainedTokenizerFast


def build_char_tokenizer(output_dir="char_tokenizer"):
    """Build a character-level tokenizer for biological sequences."""
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
        mask_token="[MASK]",
    )

    # Save to disk
    hf_tokenizer.save_pretrained(output_dir)
    print(f"Tokenizer saved to {output_dir}")
    print(f"Vocabulary size: {len(vocab)}")
    return hf_tokenizer


def main():
    """CLI entry point for building character tokenizer."""
    parser = argparse.ArgumentParser(
        description="Build character-level tokenizer for biological sequences"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="char_tokenizer",
        help="Directory to save the tokenizer (default: char_tokenizer)",
    )

    args = parser.parse_args()
    build_char_tokenizer(args.output_dir)


if __name__ == "__main__":
    main()
