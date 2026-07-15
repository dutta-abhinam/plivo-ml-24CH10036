"""Byte-level BPE tokenizer, trained ONLY on train_corpus.txt.

Why replace the vocab-256 byte tokenizer: this corpus is ~33% Devanagari *by
bytes* (every Hindi code point is 3 UTF-8 bytes). A byte tokenizer therefore
spends 3 prediction slots per Hindi character, so 2000 steps of training see
far less *text* than they could. BPE merges frequent byte sequences (English
sub-words and whole Devanagari syllables) into single tokens, compressing the
stream ~3-4x. With a fixed step budget the model then sees several times more
effective context per step, which is what actually moves bits-per-byte down.

Guarantees kept (the grader enforces them):
  * lossless: decode(encode(text)) == text for arbitrary UTF-8, because every
    token id expands to a fixed byte string and encoding only *groups* adjacent
    bytes -- it never drops or reorders them. Every one of the 256 byte values
    is a base token, so there is always a byte fallback.
  * load() takes no required args and resolves its data file relative to
    __file__, so grading (cwd = submission folder, no internet) works.

Pure stdlib (re, json) -- no numpy, no external tokenizer library.
"""
import json
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT = os.path.join(_HERE, "bpe.json")

# Pre-tokenization: partition text into letter / digit / other / whitespace
# runs so merges never cross a word boundary. Devanagari (U+0900-U+097F) counts
# as letters. A single leading space attaches to a word (the "_the" trick);
# longer whitespace runs fall through to the final class. Every character
# matches exactly one branch, so the pieces always re-concatenate to the input.
_LETTER = r"A-Za-zऀ-ॿ"
_PAT = re.compile(
    rf"[ ]?[{_LETTER}]+|[ ]?[0-9]+|[ ]?[^ \t\n\r\f\v{_LETTER}0-9]+|\s+"
)


class BPETokenizer:
    def __init__(self, merges, pattern=None):
        # merges: list of [a, b] id pairs, in learned order (rank = position).
        self.pat = re.compile(pattern) if pattern else _PAT
        self.merges_list = merges
        self.merge_rank = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for rank, (a, b) in enumerate(merges):
            new_id = 256 + rank
            self.merge_rank[(a, b)] = new_id
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
        self.vocab_size = 256 + len(merges)

    # -- encoding -----------------------------------------------------------
    def _encode_piece(self, ids):
        # greedy: repeatedly merge the adjacent pair with the lowest rank
        while len(ids) >= 2:
            best_rank, best_i = None, None
            for i in range(len(ids) - 1):
                nid = self.merge_rank.get((ids[i], ids[i + 1]))
                if nid is not None and (best_rank is None or nid < best_rank):
                    best_rank, best_i = nid, i
            if best_i is None:
                break
            ids[best_i:best_i + 2] = [best_rank]
        return ids

    def encode(self, text):
        out = []
        for piece in self.pat.findall(text):
            out.extend(self._encode_piece(list(piece.encode("utf-8"))))
        return out

    def decode(self, ids):
        b = b"".join(self.vocab[i] for i in ids)
        return b.decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "bpe", "merges": self.merges_list,
                       "pattern": self.pat.pattern}, f)


class ByteTokenizer:
    """Fallback so the interface never breaks if bpe.json is missing."""
    vocab_size = 256

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(ids).decode("utf-8", errors="replace")


def load(path=None):
    """Return the tokenizer used by train.py and evaluate.py."""
    # TOKMODE=byte forces the raw byte tokenizer (used only for my own
    # ablation runs; grading never sets it, so bpe.json is the default).
    if os.environ.get("TOKMODE") == "byte":
        return ByteTokenizer()
    # BPE_JSON lets my ablation runs point at a different vocab file; grading
    # leaves it unset and gets the default bpe.json.
    path = path or os.environ.get("BPE_JSON") or _DEFAULT
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("type") == "bpe":
            return BPETokenizer(d["merges"], d.get("pattern"))
    return ByteTokenizer()


# ---------------------------------------------------------------------------
# training (run once; not used at grading time)
# ---------------------------------------------------------------------------
def train_bpe(corpus_path, vocab_size=2048, out_path=_DEFAULT):
    """Word-frequency byte-level BPE with incremental pair-count updates."""
    from collections import Counter
    text = open(corpus_path, encoding="utf-8").read()
    pieces = _PAT.findall(text)
    assert "".join(pieces) == text, "pre-tokenizer is not a lossless partition"

    freq = Counter(pieces)
    words = [[list(w.encode("utf-8")), f] for w, f in freq.items()]

    pair_counts = Counter()
    pair_where = {}                    # pair -> set of word indices
    for wi, (ids, f) in enumerate(words):
        for a, b in zip(ids, ids[1:]):
            pair_counts[(a, b)] += f
            pair_where.setdefault((a, b), set()).add(wi)

    n_merges = vocab_size - 256
    merges = []
    for _ in range(n_merges):
        if not pair_counts:
            break
        pair = max(pair_counts, key=pair_counts.get)
        if pair_counts[pair] <= 1:
            break
        new_id = 256 + len(merges)
        merges.append([pair[0], pair[1]])
        a, b = pair
        for wi in list(pair_where.get(pair, ())):
            ids, f = words[wi]
            for p, q in zip(ids, ids[1:]):           # drop old contributions
                pair_counts[(p, q)] -= f
                if pair_counts[(p, q)] <= 0:
                    del pair_counts[(p, q)]
            merged, i = [], 0                        # merge (a,b) in this word
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(ids[i])
                    i += 1
            words[wi][0] = merged
            for p, q in zip(merged, merged[1:]):     # add new contributions
                pair_counts[(p, q)] += f
                pair_where.setdefault((p, q), set()).add(wi)
        pair_where.pop(pair, None)

    tok = BPETokenizer(merges)
    tok.save(out_path)
    print(f"trained BPE: {len(merges)} merges -> vocab {tok.vocab_size}, "
          f"saved {out_path}")
    return tok


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="../data/train_corpus.txt")
    ap.add_argument("--vocab", type=int, default=2048)
    ap.add_argument("--out", default=_DEFAULT)
    a = ap.parse_args()
    train_bpe(a.corpus, a.vocab, a.out)
