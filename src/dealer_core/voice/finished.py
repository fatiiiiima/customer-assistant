"""Does what the caller has said so far READ as a finished sentence? (voice.smart_text_check, ARCH_REVIEW section 92)

Smart Turn v3 judges only the sound of the end of a turn. In the 2 Nov 17:35 call it held "I'm looking for Hyundai." for 0.85 s, though the words are complete. The recogniser
writes the end of a sentence with a full stop or a question mark, and a sentence that is not over ends on a word that needs another ("for", "the", "and", "is"). This looks
at the words the endpointer's peek recognised and says "finished" only when:
  * the text ends with a full stop, a question mark or an exclamation mark (a comma, a dash or "..." is a pause, not an end);
  * the last word is not one that needs another after it (an article, a preposition, a conjunction, an auxiliary verb, a filler);
  * it has enough words: 3 before a full stop, 2 before a question mark ("What colour?"). One-word answers after a question are `answers.is_complete_answer`'s job, which knows the question.
A rule, not a model: nothing to download, nothing to load, a few microseconds. It can still be wrong the way any short-pause rule can: a caller who finishes a clause, stops for the
detector's short silence and carries on ("I'm looking for a Toyota. ... Under 80,000.") has the turn ended on the first half. That is why the setting is off until a call shows the cost.
"""
from __future__ import annotations

import re

_END = re.compile(r"[.?!]['\")\]]*\s*$")
_TRAILING_PAUSE = re.compile(r"(?:\.\.\.|…|[,;:\-–—])\s*$")
_WORD = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)?")

# A sentence is unlikely to end on these; where one can ("I don't know where it is.") it is held as before, which is the safe side of the mistake. Pronouns are NOT here: "tell me about it." and "is it?" are whole.
NEEDS_MORE = frozenset("""
a an the some any another each every
and or but because if while although though as than plus
of to for with without in on at by from into onto about over under between through against per via
is are was were am be been being do does did has have had can could will would should shall may might must
i'm i've i'll i'd you're we're they're it's that's there's what's who's
my your his her its our their
um uh er erm hmm
not very really just quite
what which who whom whose how when where why whether
""".split())

# Only a question mark can excuse these: "what colour is it?" ends on a pronoun and is whole, "I'm looking for..." does not.
FILLERS = frozenset("um uh er erm hmm mm mhm huh".split())


def reads_finished(text: str) -> bool:
    """True when `text` ends like a complete sentence (see the module note)."""
    t = (text or "").strip()
    if not t or _TRAILING_PAUSE.search(t) or not _END.search(t):
        return False
    words = [w.lower().replace("’", "'") for w in _WORD.findall(t)]
    if not words or all(w in FILLERS for w in words):
        return False
    question = "?" in t[-3:]
    if len(words) < (2 if question else 3):
        return False
    last = words[-1]
    if last in FILLERS:
        return False
    if question:                                              # "What colour is it?", "Is it available?": the question mark is the end, whatever the last word
        return last not in {"a", "an", "the", "and", "or", "but", "of", "to", "for", "with", "in", "on", "at", "by", "from", "about"}
    return last not in NEEDS_MORE
