"""
A small BPE tokenizer that we train ourselves.

Byte-Pair Encoding starts with individual characters as tokens. It then
repeatedly finds the pair of neighbouring tokens that appears most often
and merges that pair into a new token. After a few thousand merges, common
words (like "the" or "once") become a single token, while rare words still
break into smaller pieces.

This file does not download or load any pretrained tokenizer. The vocabulary
and merge rules are learned only from our training text.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


# Special tokens live at the start of the vocabulary.
# <|endoftext|> marks the end of a story or a chat exchange so the model
# can learn when to stop talking.
END_OF_TEXT = "<|endoftext|>"
UNK = "<|unk|>"
SPECIAL_TOKENS = [END_OF_TEXT, UNK]

# Split raw text into "words" (and punctuation) before learning merges.
# Merges never cross these boundaries, which keeps training fast and
# matches the usual GPT-2 style of BPE.
PRETOKEN_RE = re.compile(
    r"<\|endoftext\|>|"
    r"'(?:[sSdDmM]|ll|LL|re|RE|ve|VE)|"
    r" ?\w+|"
    r" ?[^\s\w]+|"
    r"\s+"
)


def _split_pretokens(text: str) -> list[str]:
    return PRETOKEN_RE.findall(text)


def _merge_pair_in_word(word: tuple[str, ...], pair: tuple[str, str], new_token: str) -> tuple[str, ...]:
    """Replace every adjacent occurrence of `pair` inside one word."""
    out: list[str] = []
    i = 0
    a, b = pair
    while i < len(word):
        if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
            out.append(new_token)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


class BPETokenizer:
    """Character-base BPE with a JSON save format we fully control."""

    def __init__(self) -> None:
        self.vocab: list[str] = []
        self.stoi: dict[str, int] = {}
        self.merges: list[tuple[str, str]] = []  # in the order they were learned
        # For fast encoding: a merged token -> its left and right parts is not
        # needed; we apply merges in order on each pretoken and cache results.
        self._encode_cache: dict[str, list[int]] = {}
        self._ranks: dict[tuple[str, str], int] = {}

    # ----- train -----

    def train(self, texts: Iterable[str], vocab_size: int = 4096, verbose: bool = True) -> None:
        """Learn merges from raw text until we reach `vocab_size` tokens."""
        if vocab_size < len(SPECIAL_TOKENS) + 50:
            raise ValueError("vocab_size is too small")

        # Count how often each pretoken (word, space+word, punctuation) appears.
        pretoken_counts: Counter[str] = Counter()
        charset: set[str] = set()
        for text in texts:
            for piece in _split_pretokens(text):
                if piece == END_OF_TEXT:
                    continue
                pretoken_counts[piece] += 1
                charset.update(piece)

        # Always keep basic printable ASCII so user text with new punctuation
        # is less likely to fall back to <|unk|>.
        for code in range(32, 127):
            charset.add(chr(code))
        charset.add("\n")
        charset.add("\t")

        self.vocab = list(SPECIAL_TOKENS) + sorted(charset)
        self.stoi = {tok: i for i, tok in enumerate(self.vocab)}
        self.merges = []

        # Represent each unique word as a tuple of characters.
        words: dict[tuple[str, ...], int] = {}
        for piece, count in pretoken_counts.items():
            words[tuple(piece)] = words.get(tuple(piece), 0) + count

        n_merges = vocab_size - len(self.vocab)
        if verbose:
            print(
                f"BPE: {len(pretoken_counts):,} unique pretokens, "
                f"{len(charset)} base characters, learning {n_merges:,} merges..."
            )

        for merge_i in range(n_merges):
            stats: Counter[tuple[str, str]] = Counter()
            for word, freq in words.items():
                if len(word) < 2:
                    continue
                for i in range(len(word) - 1):
                    stats[(word[i], word[i + 1])] += freq
            if not stats:
                break
            pair, pair_count = stats.most_common(1)[0]
            if pair_count < 2:
                break
            new_token = pair[0] + pair[1]
            # If that string is already a token (can happen with overlaps),
            # still record the merge; the vocab entry is reused.
            if new_token not in self.stoi:
                self.stoi[new_token] = len(self.vocab)
                self.vocab.append(new_token)
            self.merges.append(pair)

            new_words: dict[tuple[str, ...], int] = {}
            for word, freq in words.items():
                merged = _merge_pair_in_word(word, pair, new_token)
                new_words[merged] = new_words.get(merged, 0) + freq
            words = new_words

            if verbose and (merge_i + 1) % 500 == 0:
                print(f"  merge {merge_i + 1}/{n_merges}  most recent {pair!r} -> {new_token!r} ({pair_count:,} times)")

        self._finalize()
        if verbose:
            print(f"BPE done. Vocabulary size: {len(self.vocab):,}")

    def _finalize(self) -> None:
        self._ranks = {pair: i for i, pair in enumerate(self.merges)}
        self._encode_cache = {END_OF_TEXT: [self.stoi[END_OF_TEXT]]}

    # ----- encode / decode -----

    def encode_one_pretoken(self, piece: str) -> list[int]:
        if piece in self._encode_cache:
            return self._encode_cache[piece]
        if piece == END_OF_TEXT:
            ids = [self.stoi[END_OF_TEXT]]
            self._encode_cache[piece] = ids
            return ids

        symbols = tuple(piece)
        # Apply merges in the same order they were learned (GPT-2 / original BPE).
        # Using ranks also lets us always merge the best-ranked pair first,
        # which matches how we trained (sequential merges).
        while True:
            if len(symbols) < 2:
                break
            candidate = None
            best_rank = None
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                rank = self._ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    candidate = pair
            if candidate is None:
                break
            symbols = _merge_pair_in_word(symbols, candidate, candidate[0] + candidate[1])

        unk_id = self.stoi[UNK]
        ids = [self.stoi.get(s, unk_id) for s in symbols]
        # Bound the cache so encoding a huge unique-word corpus cannot grow forever.
        if len(self._encode_cache) < 200_000:
            self._encode_cache[piece] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece in _split_pretokens(text):
            ids.extend(self.encode_one_pretoken(piece))
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        unk = UNK
        parts: list[str] = []
        vocab = self.vocab
        n = len(vocab)
        for i in ids:
            if 0 <= int(i) < n:
                tok = vocab[int(i)]
                if tok != UNK:
                    parts.append(tok)
                else:
                    parts.append(unk)
            else:
                parts.append(unk)
        return "".join(parts)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ----- save / load -----

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "type": "bpe_from_scratch",
            "special_tokens": SPECIAL_TOKENS,
            "vocab": self.vocab,
            "merges": [list(p) for p in self.merges],
        }
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        tok = cls()
        tok.vocab = list(data["vocab"])
        tok.stoi = {s: i for i, s in enumerate(tok.vocab)}
        tok.merges = [tuple(p) for p in data["merges"]]
        tok._finalize()
        return tok
