#!/usr/bin/env python

# nplm/build_word_vocab.py
from __future__ import annotations
import argparse
import json
import os
from typing import Optional
from nplm.tokenization.word_tokenizer import WordTokenizer, TokenizerConfig

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a word-level vocabulary from a JSONL corpus (one document per line)."
    )
    p.add_argument(
        "--jsonl_dir",
        type=str,
        required=True,
        help="Directory containing .jsonl or .jsonl.gz files (recursively).",
    )
    p.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the induced vocabulary JSON (e.g., artifacts/wikitext2_word_vocab.json).",
    )
    p.add_argument(
        "--text_field",
        type=str,
        default="text",
        help='JSON field name containing document text (default: "text").',
    )
    p.add_argument(
        "--min_freq",
        type=int,
        default=2,
        help="Minimum frequency threshold to include a token (default: 2).",
    )
    p.add_argument(
        "--max_vocab",
        type=int,
        default=20000,
        help="Cap vocabulary size (excluding specials). Set <=0 for unlimited (default: 20000).",
    )
    p.add_argument(
        "--lowercase",
        action="store_true",
        help="Lowercase text before tokenization.",
    )
    p.add_argument(
        "--tokenizer",
        type=str,
        choices=["simple", "whitespace"],
        default="simple",
        help='Tokenizer type: "simple" (regex) or "whitespace" (split). Default: simple.',
    )
    p.add_argument(
        "--strip_punct",
        action="store_true",
        help='If using tokenizer="simple", drop single-character punctuation tokens.',
    )
    p.add_argument(
        "--no_bos",
        action="store_true",
        help="Do not include <bos> token in the vocabulary.",
    )
    p.add_argument(
        "--no_eos",
        action="store_true",
        help="Do not include <eos> token in the vocabulary.",
    )
    p.add_argument(
        "--report_unk_rate_dir",
        type=str,
        default=None,
        help="Optional: path to JSONL dir (e.g., validation set) to report estimated UNK rate.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    max_vocab = None if args.max_vocab <= 0 else args.max_vocab
    cfg = TokenizerConfig(
        min_freq=args.min_freq,
        max_vocab=max_vocab,
        lowercase=args.lowercase,
        tokenizer=args.tokenizer,
        strip_punct=args.strip_punct,
        include_bos=not args.no_bos,
        include_eos=not args.no_eos,
    )

    vocab = WordTokenizer.build_from_corpus(
        jsonl_dir=args.jsonl_dir,
        config=cfg,
        text_field=args.text_field,
        progress=True,
    )

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    vocab.save(args.output_path)

    print(f"[nplm] Saved vocab of size {len(vocab.id_to_token)} -> {args.output_path}")

    if args.report_unk_rate_dir:
        rate = vocab.unk_rate_on_dir(args.report_unk_rate_dir, text_field=args.text_field)
        print(f"[nplm] Estimated UNK rate on '{args.report_unk_rate_dir}': {rate:.4%}")


if __name__ == "__main__":
    main()
