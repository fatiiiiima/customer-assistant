"""Common first names, for the name step of a booking (owner, 29 Oct: the agent did not get "Ahmed"; callers will have Arab, white and Indian names).

The recogniser writes what it hears, and a name it does not know comes out spelled oddly or as a near-miss ("Achmed", "Amed"). Training the recogniser on names
would need recordings of many people saying them and a GPU; this instead corrects the written name against config/common_names.txt, carefully:

* a first name ON the list is never changed (so every real spelling a caller may use belongs on the list: "Ahmad" and "Ahmed" are both there);
* a first name NOT on the list becomes a listed one only when exactly one listed name has the same letters in another order ("Jhon", "Hussien"), or one listed
  name (or its spellings) sounds the same by a rough consonant key ("Annel" -> Anil, "Kyrin" -> Kiran), or exactly one is a close match (fuzzy ratio >= 85, same
  first letter); several different names, or nothing close, leaves it as heard;
* surnames and longer names are left alone (only the first word is looked at).

The read-back ("... for Ahmed, on ... Shall I book it?") stays the caller's chance to correct it. Measured offline in docs/ARCH_REVIEW.md section 28.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

LIST = Path(__file__).resolve().parents[3] / "config" / "common_names.txt"
_SOUND = [("ph", "f"), ("kh", "k"), ("gh", "g"), ("sh", "s"), ("ch", "k"), ("ck", "k"), ("q", "k"), ("c", "k"), ("z", "s"), ("w", "v")]
MIN_RATIO = 85.0
SOUND_MIN_RATIO = 65.0           # a sound-alike must still share this much spelling (29 Oct: without it, "Kiara" became Cherry and "Seren" Shirin)
TIE_GAP = 3.0                    # a second candidate this close to the best one makes the match ambiguous: nothing is changed


def sound(word: str) -> str:
    """A rough sound key: the first letter and the consonants after it, with letters that sound alike merged ("Anil" and "Annel" -> "anl", "Kiran", "Kyrin" and "Curin"
    -> "krn", "Ramesh" and "Ramish" -> "rms"). Vowels, h and y after the first letter carry little of a name across accents and spellings, so they are dropped."""
    w = word.lower()
    for a, b in _SOUND:
        w = w.replace(a, b)
    out = w[:1]
    for ch in w[1:]:
        if ch in "aeiouyh" or (out and ch == out[-1]):
            continue
        out += ch
    return out


@lru_cache(maxsize=1)
def by_sound() -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for low, spelled in known().items():
        groups.setdefault(sound(low), []).append(spelled)
    return groups


@lru_cache(maxsize=1)
def known() -> dict[str, str]:
    """lower case -> the spelling on the list."""
    out: dict[str, str] = {}
    if LIST.exists():
        for line in LIST.read_text(encoding="utf-8").splitlines():
            w = line.strip()
            if w and not w.startswith("#"):
                out.setdefault(w.lower(), w)
    return out


def _one_swap(a: str, b: str) -> bool:
    """b is a with two neighbouring letters swapped."""
    if len(a) != len(b) or a == b:
        return False
    diff = [i for i in range(len(a)) if a[i] != b[i]]
    return len(diff) == 2 and diff[0] > 0 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]]     # never the first letter ("Ira" is not Ria)


def correct_first(word: str) -> str:
    """One first name, corrected to a listed one when it is a close, unambiguous sound-alike that is not itself listed."""
    from rapidfuzz import fuzz

    names = known()
    low = word.lower()
    if len(low) < 3 or low in names or not low.isalpha():
        return word
    swapped = [spelled for n, spelled in names.items() if _one_swap(low, n)]
    if len(swapped) == 1:                                               # two neighbouring letters swapped: "Jhon", "Hussien" (not "Myra" -> Mary: a name of its own)
        return swapped[0]
    alike = [x for x in by_sound().get(sound(low), []) if x[0].lower() == low[0] and abs(len(x) - len(low)) <= 1 and fuzz.ratio(low, x.lower()) >= SOUND_MIN_RATIO]
    if len(sound(low)) >= 3 and alike and all(fuzz.ratio(alike[0].lower(), x.lower()) >= 75 for x in alike):
        return max(alike, key=lambda x: fuzz.ratio(low, x.lower()))    # one name (or its spellings: Sara / Sarah) sounds the same: "Annel" -> Anil
    scored = sorted(((fuzz.ratio(low, n), spelled) for n, spelled in names.items() if n[0] == low[0]), reverse=True)
    if not scored or scored[0][0] < MIN_RATIO:
        return word
    if len(scored) > 1 and scored[1][0] >= scored[0][0] - TIE_GAP and scored[1][1].lower() != scored[0][1].lower():
        return word                                                     # two listed names are about as close: do not guess
    return scored[0][1]


def correct(name: str | None) -> str | None:
    """A whole name as read from the caller's words: its first word corrected (see correct_first), the rest as heard."""
    if not name:
        return name
    words = name.split()
    return " ".join([correct_first(words[0]), *words[1:]])
