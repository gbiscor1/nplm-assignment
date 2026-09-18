# nplm/tokenization/word_tokenizer.py
"""
Implements a simple word-level tokenizer and vocabulary builder following the approach used in Bengio et al. (2003), including vocabulary construction, encoding, decoding, persistence, and unknown-token reporting.
"""

from __future__ import annotations

import gzip
import json
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from nplm.tokenization.tokenization_interface import Tokenizer

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x


# ---- Special tokens ----
PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"

DEFAULT_SPECIALS = [PAD, UNK, BOS, EOS]


# ---- Basic tokenization ----
_TOKEN_PATTERNS = {
    "whitespace": None,
    "simple": r"[A-Za-z0-9_'-]+|[^\sA-Za-z0-9_]",
}


def _iter_jsonl_paths(jsonl_dir: str) -> Iterator[str]:
    for root, _dirs, files in os.walk(jsonl_dir):
        for fname in files:
            if fname.endswith(".jsonl") or fname.endswith(".jsonl.gz"):
                yield os.path.join(root, fname)


def _open_textmaybe_gzip(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")

    return open(path, "r", encoding="utf-8", errors="ignore")


def iter_text_from_jsonl_dir(jsonl_dir: str, text_field: str = "text") -> Iterator[str]:
    import json as _json

    for path in _iter_jsonl_paths(jsonl_dir):
        with _open_textmaybe_gzip(path) as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                try:
                    obj = _json.loads(line)
                except Exception:
                    continue

                if text_field in obj and isinstance(obj[text_field], str):
                    yield obj[text_field]


def basic_tokenize(text: str, *, lowercase: bool = False, tokenizer: str = "simple", strip_punct: bool = False) -> List[str]:
    if lowercase:
        text = text.lower()

    pattern = _TOKEN_PATTERNS.get(tokenizer)

    if pattern is None:
        return text.split()

    tokens = re.findall(pattern, text)

    if strip_punct:
        tokens = [token for token in tokens if re.match(r"[A-Za-z0-9_'-]+$", token)]

    return tokens


@dataclass
class TokenizerConfig:
    min_freq: int = 2
    max_vocab: Optional[int] = 20000
    lowercase: bool = False
    tokenizer: str = "simple"
    strip_punct: bool = False
    include_bos: bool = True
    include_eos: bool = True
    specials: Optional[List[str]] = None


class WordTokenizer(Tokenizer):
    """
    Class WordTokenizer: word-level tokenizer implementation with vocabulary construction, encoding, decoding, persistence, and unknown-token reporting.

    Internal parameters:
        token_to_id (Dict[str, int]): mapping from tokens to integer IDs.
        id_to_token (List[str]): ordered mapping from integer IDs to tokens.
        config (TokenizerConfig): configuration controlling vocabulary and tokenization behavior.
        freqs (Optional[Dict[str, int]]): optional token frequency counts from the training corpus.
    """

    def __init__(self, token_to_id: Dict[str, int], id_to_token: List[str], config: TokenizerConfig, freqs: Optional[Dict[str, int]] = None):
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token
        self.config = config
        self.freqs = freqs or {}

        self.pad_id = self.token_to_id.get(PAD)
        self.unk_id = self.token_to_id.get(UNK)
        self.bos_id = self.token_to_id.get(BOS) if config.include_bos else None
        self.eos_id = self.token_to_id.get(EOS) if config.include_eos else None

    @classmethod
    def build_from_corpus(cls, jsonl_dir: str, config: Optional[TokenizerConfig] = None, text_field: str = "text", progress: bool = True) -> "WordTokenizer":
        config = config or TokenizerConfig()
        specials = config.specials or DEFAULT_SPECIALS

        counter: Counter = Counter()
        texts = iter_text_from_jsonl_dir(jsonl_dir, text_field=text_field)
        iterator = tqdm(texts, desc="Scanning text", unit="doc") if progress else texts

        for document in iterator:
            tokens = basic_tokenize(document, lowercase=config.lowercase, tokenizer=config.tokenizer, strip_punct=config.strip_punct)
            counter.update(tokens)

        items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        base_vocab: List[str] = [token for token, count in items if count >= config.min_freq]

        if config.max_vocab is not None and config.max_vocab > 0:
            base_vocab = base_vocab[:max(0, config.max_vocab)]

        final_tokens: List[str] = []

        for special in specials:
            if special not in final_tokens:
                final_tokens.append(special)

        for token in base_vocab:
            if token not in final_tokens:
                final_tokens.append(token)

        token_to_id = {token: index for index, token in enumerate(final_tokens)}
        id_to_token = final_tokens

        return cls(token_to_id=token_to_id, id_to_token=id_to_token, config=config, freqs=dict(counter))

    def save(self, output_path: Path) -> Path:
        """
        save: saves the tokenizer vocabulary, configuration, and token frequencies to disk.

        Params:
            output_path (Path): destination path for the tokenizer artifact.

        Returns:
            Path (Path): path to the saved tokenizer artifact.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "token_to_id": self.token_to_id,
            "id_to_token": self.id_to_token,
            "config": asdict(self.config),
            "freqs": self.freqs,
        }

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

        return output_path

    @classmethod
    def load(cls, input_path: Path) -> "WordTokenizer":
        """
        load: loads a saved word tokenizer artifact from disk.

        Params:
            input_path (Path): path to the saved tokenizer artifact.

        Returns:
            WordTokenizer (WordTokenizer): loaded word tokenizer instance.
        """
        with input_path.open("r", encoding="utf-8") as file:
            obj = json.load(file)

        config = TokenizerConfig(**obj["config"])

        return cls(
            token_to_id=obj["token_to_id"],
            id_to_token=obj["id_to_token"],
            config=config,
            freqs=obj.get("freqs", {}),
        )

    def encode(self, text: str) -> list[int]:
        """
        encode: converts text into token IDs using the configured word-level tokenizer.

        Params:
            text (str): text to tokenize and encode.

        Returns:
            list[int] (list[int]): encoded token IDs.
        """
        return self.encode_text(text)

    def decode(self, token_ids: list[int]) -> str:
        """
        decode: converts token IDs back into a space-separated text representation.

        Params:
            token_ids (list[int]): token IDs to decode.

        Returns:
            str (str): decoded text.
        """
        return " ".join(self.decode_ids(token_ids))

    def encode_tokens(self, tokens: List[str]) -> List[int]:
        return [self.token_to_id.get(token, self.unk_id) for token in tokens]

    def decode_ids(self, ids: List[int]) -> List[str]:
        return [self.id_to_token[token_id] if 0 <= token_id < len(self.id_to_token) else UNK for token_id in ids]

    def tokenize(self, text: str) -> List[str]:
        return basic_tokenize(
            text,
            lowercase=self.config.lowercase,
            tokenizer=self.config.tokenizer,
            strip_punct=self.config.strip_punct,
        )

    def encode_text(self, text: str, with_bos_eos: bool = False) -> List[int]:
        tokens = self.tokenize(text)

        if with_bos_eos:
            tokens = ([BOS] if self.config.include_bos else []) + tokens + ([EOS] if self.config.include_eos else [])

        return self.encode_tokens(tokens)

    def unk_rate_on_dir(self, jsonl_dir: str, text_field: str = "text") -> float:
        total = 0
        unknown = 0

        for document in iter_text_from_jsonl_dir(jsonl_dir, text_field=text_field):
            tokens = self.tokenize(document)
            total += len(tokens)
            unknown += sum(1 for token in tokens if token not in self.token_to_id)

        return (unknown / total) if total > 0 else 0.0