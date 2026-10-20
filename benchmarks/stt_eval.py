"""STT scoring: number-aware WER, canonical number matching and car-term recall.

Reference texts have numbers spelled as words. STT engines differ (sherpa: UPPERCASE
words, Whisper: digits), so both sides are normalised before comparison.
"""
from __future__ import annotations

import re

import jiwer

ONES = {"zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
        "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
DIGIT_WORDS = {k for k, v in ONES.items() if v < 10}
NUM_WORDS = set(ONES) | set(TENS) | {"hundred", "thousand", "million", "and"}
_WORD_OF = {v: k for k, v in ONES.items() if k != "oh"}


def _two_digits_to_words(n: int) -> str:
    if n < 20:
        return _WORD_OF[n]
    t, o = divmod(n, 10)
    tw = {v: k for k, v in TENS.items()}[t * 10]
    return tw + (f" {_WORD_OF[o]}" if o else "")


def int_to_words(n: int) -> str:
    if n < 100:
        return _two_digits_to_words(n)
    if n < 1000:
        h, r = divmod(n, 100)
        return f"{_WORD_OF[h]} hundred" + (f" and {int_to_words(r)}" if r else "")
    if n < 1_000_000:
        th, r = divmod(n, 1000)
        return f"{int_to_words(th)} thousand" + (f" {int_to_words(r)}" if r else "")
    m, r = divmod(n, 1_000_000)
    return f"{int_to_words(m)} million" + (f" {int_to_words(r)}" if r else "")


def normalise(text: str) -> str:
    """lowercase, strip punctuation, spell out digits so Whisper-style output compares with words."""
    t = text.lower().replace("+", " plus ")
    t = re.sub(r"(?<=\d),(?=\d{3})", "", t)  # 55,000 -> 55000
    t = re.sub(r"[%$]", " ", t)

    def join_groups(m: re.Match) -> str:  # "50 444 3322" -> "504443322" (phone-style grouping)
        d = re.sub(r"\D", "", m.group(0))
        return d if len(d) >= 7 else m.group(0)

    t = re.sub(r"\d{1,4}(?:[\s-]\d{1,4}){2,}", join_groups, t)

    def digits(m: re.Match) -> str:
        s = m.group(0)
        if len(s) >= 5 and s.startswith("0") or len(s) >= 7:  # phone-like: speak digit by digit
            return " " + " ".join(_WORD_OF[int(c)] for c in s) + " "
        return " " + int_to_words(int(s)) + " "

    t = re.sub(r"\d+", digits, t)
    t = re.sub(r"[^a-z\s'-]", " ", t).replace("-", " ")
    t = t.replace("mercedes benz", "mercedes benz")
    return re.sub(r"\s+", " ", t).strip()


def number_spans(toks: list[str]) -> list[tuple[str, int, int]]:
    """Find numbers in normalised tokens: [(canonical_value, start_idx, end_idx_exclusive), ...].

    Digit-word runs (>=3) become a digit string; magnitude phrases become integers; the
    'twenty twenty one' year pattern becomes 2021.
    """
    out: list[tuple[str, int, int]] = []
    i = 0
    while i < len(toks):
        w = toks[i]
        if w in DIGIT_WORDS:
            j = i
            while j < len(toks) and toks[j] in DIGIT_WORDS:
                j += 1
            if j - i >= 3:
                out.append(("".join(str(ONES[x]) for x in toks[i:j]), i, j))
                i = j
                continue
        if w in ONES or w in TENS or w in ("hundred", "thousand", "million"):
            j = i
            while j < len(toks) and (toks[j] in NUM_WORDS):
                j += 1
            chunk = toks[i:j]
            # year pattern: "twenty twenty one", "twenty nineteen"
            if chunk[0] == "twenty" and len(chunk) >= 2 and not ({"hundred", "thousand", "million", "and"} & set(chunk)):
                second = chunk[1]
                if second in TENS or ONES.get(second, 0) >= 10:
                    rest = _small(chunk[1:])
                    if rest is not None and 10 <= rest <= 99:
                        out.append((str(2000 + rest), i, j))
                        i = j
                        continue
            out.append((str(_value(chunk)), i, j))
            i = j
            continue
        i += 1
    return out


def canon_numbers(text: str) -> list[str]:
    """Numbers a listener would hear, as canonical strings (wording-independent)."""
    return [v for v, _, _ in number_spans(normalise(text).split())]


def _small(chunk: list[str]) -> int | None:
    total = 0
    for w in chunk:
        if w in TENS:
            total += TENS[w]
        elif w in ONES:
            total += ONES[w]
        else:
            return None
    return total


def _value(chunk: list[str]) -> int:
    total, cur = 0, 0
    for w in chunk:
        if w == "and":
            continue
        if w in ONES:
            cur += ONES[w]
        elif w in TENS:
            cur += TENS[w]
        elif w == "hundred":
            cur = max(cur, 1) * 100
        elif w in ("thousand", "million"):
            total += max(cur, 1) * (1000 if w == "thousand" else 1_000_000)
            cur = 0
    return total + cur


def wer(ref: str, hyp: str) -> float:
    r, h = normalise(ref), normalise(hyp)
    if not r:
        return 0.0
    return jiwer.wer(r, h)


def term_recall(ref: str, hyp: str, terms: list[str]) -> tuple[int, int]:
    """(found, expected) for known car/place/name terms that appear in the reference."""
    r, h = " " + normalise(ref) + " ", " " + normalise(hyp) + " "
    exp = found = 0
    for t in terms:
        nt = " " + normalise(t) + " "
        if nt in r:
            exp += 1
            found += int(nt in h)
    return found, exp
