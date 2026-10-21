"""The model AUTHORS the spoken reply, from the conversation and the facts. Templates are only the emergency fallback.

    understand   <- the model picks the intent (grammar-constrained; it also sees what the agent just said)
    decide / act <- code: search, car facts, pricing engine, booking rules, state machine. Produces FACTS and a STANDARD WORDING.
    say it       <- the model, from: the conversation so far + what the customer just said + the FACTS (+ the standard wording as a
                    reference it must NOT copy). It writes the reply in its own words.            <- this module
    safe to say? <- `verify_reply` checks hard safety: no invented number, name or claim; the decision, the amounts and the key phrases are
                    kept; the question the system is waiting on is still asked. Only if that fails (or the model is down) the standard
                    wording is spoken instead.

Risk tiers set how strict the check is:
  LOW   (searching, car questions): free wording, any true fact from the FACTS; no invented numbers, names or claims.
  HIGH  (price decisions, booking, read-backs, confirmations, hand-offs, policy): every figure, date, name and key phrase of the system's
        decision must be present ("subject to confirmation", "colleague", ...), none added. Free wording around them.
  CHAT  (small talk): friendly, no facts, no claims about the dealership, comes back to the cars.

The model never sees the floor price or any pricing limit: only the decision the pricing engine already made.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher

from . import voice_rules
from .draft import _CHARS, _NEG, _QUOTED, capitals, clean_draft, numbers
from .leadin import _PROMISE, LeadInBackend, LeadInOutcome
from .phrasebook import FORBIDDEN

_NUMBER_WORDS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
                 "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]
MAX_SENTENCES = 4
MAX_WORDS = 75
WRITE_TOKENS = 80
RETRIES = 1                 # one more try, told why the first was rejected
TEMPERATURE = 0.6           # sampling, so the wording varies from call to call (the facts do not)

HIGH_INTENTS = {"make_offer", "ask_lowest_price", "book_test_drive", "give_slot", "give_contact", "reschedule", "cancel_booking",
                "send_details", "claim_authority", "ask_policy", "angry", "request_human"}
HIGH_STATES = {"negotiating", "slot", "contact", "confirm"}
HIGH_TOOLS = ("propose_price", "book", "escalate", "cancel", "move", "send", "reschedule")
KEY_WORDS = ("colleague", "manager", "approval", "whatsapp", "cancelled", "booked", "subject to confirmation")

# Claims about a car or the dealership that the model must not make unless the FACTS (or the customer) already said them.
CLAIMS = re.compile(r"\b(?:\w+\s)?condition\b|pristine|immaculate|spotless|flawless|\bmint\b|brand[- ]new|like new|well[- ]maintained|accident[- ]free|"
                    r"one[- ]owner|single owner|reliable|powerful|economical|efficient|spacious|luxurious|low[- ]mileage|high[- ]mileage|fully loaded|"
                    r"sunroof|leather|navigation|bluetooth|camera|turbo|\bv6\b|\bv8\b|\bawd\b|\b4wd\b|warranty|guarantee|financ\w*|\bloans?\b|insurance|"
                    r"delivery|trade[- ]?in|discount\w*|\bfree\b|last one|only one left|in demand|opening hours|open until|open till|sold out|"
                    r"inside and out|runs (?:great|perfectly|like)|never been|no accidents?|good (?:morning|afternoon|evening|night)|"
                    r"(?:we(?:'ve| have| do have)(?: got)?|i(?:'ve| have)(?: got)?) (?:a few|some|several|plenty|lots|many|loads)|that match|all (?:of them )?(?:are )?(?:red|white|black|blue|grey|gray|silver|brown|green)|"
                    r"\bsolid\b|no issues|no problems|trouble[- ]free|hassle[- ]free|"
                    r"top[- ]notch|great shape|good shape|excellent shape|runs well|drives (?:great|well|beautifully)|smooth ride|fuel[- ]efficient|"
                    r"great (?:car|deal|value|price)|good (?:deal|value|price)|bargain|steal", re.I)

INTENT_WORDS = {
    "search_cars": "they are looking for cars (a search)", "ask_car_info": "they asked about a car", "make_offer": "they made a price offer",
    "ask_lowest_price": "they asked for the best price", "book_test_drive": "they want to book a test drive", "give_slot": "they gave a day or time",
    "give_contact": "they gave their name and number", "confirm_yes": "they said yes", "confirm_no": "they said no", "greet": "they greeted",
    "goodbye": "they said goodbye", "out_of_scope": "they said something unrelated to cars", "ask_policy": "they asked about policy (warranty, financing...)",
    "angry": "they are upset", "request_human": "they asked for a person", "reschedule": "they want to change a booking", "cancel_booking": "they want to cancel",
    "send_details": "they asked for the details to be sent", "claim_authority": "they claimed special authority",
}
PENDING_WORDS = {
    "handoff_offer": "whether they want to be connected to a colleague", "amount_check": "confirmation of the amount they said",
    "offer_booking": "whether they want to book a test drive", "move_existing": "whether to move their existing booking",
    "more_or_book": "whether they want to hear more about the car or book a test drive",
    "search_other": "whether they want the system to search for the other cars it just proposed",
}


def risk_of(intent: str, state_before: str, state_after: str, decision: str, tools: list[str], handed_off: bool) -> str:
    if intent in HIGH_INTENTS or state_before in HIGH_STATES or state_after in HIGH_STATES or decision or handed_off:
        return "high"
    if any(t.startswith(HIGH_TOOLS) for t in tools):
        return "high"
    return "low"


def list_items(core: str) -> list[str]:
    """Items of any enumeration in the standard wording ("A, B, C and D"): short phrases a rewrite must not silently drop."""
    items: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", core):
        if sentence.count(",") >= 3:
            parts = [re.sub(r"^.*\b(?:only on|only|i have|we have|we've got|come in|are)\s+(?:the\s+)?", "", x.strip().rstrip(".?!").lower())
                     for x in re.split(r",\s*|\s+and\s+|\s+or\s+", sentence)]
            short = [x for x in parts if 1 <= len(x.split()) <= 3 and not re.search(r"\d", x) and not x.startswith(("among", "but ", "so ", "and "))]
            if len(parts) >= 4 and len(short) >= 0.75 * len(parts):        # a real enumeration, not a description with commas
                items.extend(short)
    return items


GENERIC_CLOSER = re.compile(r"^(?:is there |do you have )?(?:anything|any other|what) (?:else|questions)|anything else|what else", re.I)


def is_generic_closer(sentence: str) -> bool:
    return bool(GENERIC_CLOSER.search(sentence.strip()))


def question_keys(core: str) -> set[str]:
    """The content words of the question the system is waiting on ("book a test drive?"): the author may reword it but must still ask it."""
    sentences = voice_rules.sentences(core)
    if not sentences or not sentences[-1].rstrip().endswith("?") or is_generic_closer(sentences[-1]):
        return set()
    stop = {"would", "you", "like", "to", "the", "a", "an", "or", "and", "do", "does", "is", "it", "that", "this", "of", "for", "your", "me", "i",
            "can", "could", "what", "which", "one", "have", "any", "in", "on", "at", "about", "be", "so", "if", "there", "with", "my", "we", "us"}
    return {w.rstrip("s") for w in re.findall(r"[a-z]+", sentences[-1].lower()) if w not in stop and len(w) > 2}


# What small talk must never say: claims about the dealership, its services or its stock beyond what the facts give.
_DEALERSHIP_CLAIM = re.compile(r"\b(?:we|our)\s+(?:also\s+)?(?:have|has|offer|offers|provide|provides|sell|sells|do|does|are open|open|close|closed|"
                               r"accept|serve|serves|give|gives)\b|\bopen(?:ing)? (?:hours|until|till|from)\b|\b(?:coffee|snacks?|parking|wifi|delivery|"
                               r"insurance|warranty|loan|finance|financing|trade-?in)\b", re.I)


def verify_chat(draft: str, user_text: str, core: str, facts: str, history: str) -> list[str]:
    """Small talk ("its so hot today"): any friendly wording, but no facts, no numbers, no claims about the dealership, and it must come
    back to the cars. Used only when the standard wording is just "I can only help with cars"."""
    problems: list[str] = []
    if not _CHARS.match(draft):
        problems.append("characters other than plain letters, digits and basic punctuation")
    if len(voice_rules.sentences(draft)) > 3 or len(draft.split()) > 40:
        problems.append("too long for small talk")
    if numbers(draft) - (numbers(facts) | numbers(history) | numbers(user_text)):
        problems.append("new number(s) in small talk")
    if FORBIDDEN.search(draft) or (_PROMISE.search(draft) and not _PROMISE.search(core)):
        problems.append("commitment or commercial wording")
    if _DEALERSHIP_CLAIM.search(draft):
        problems.append("a claim about the dealership")
    known = {w.lower() for w in re.findall(r"[A-Za-z0-9']+", user_text + " " + core + " " + facts + " " + history)}
    known |= {"agmc"}
    new_caps = {w for w in capitals(draft) if w not in known and w.rstrip("s") not in {k.rstrip("s") for k in known}}
    if new_caps:
        problems.append(f"new name(s) {sorted(new_caps)[:3]}")
    if not re.search(r"\b(?:cars?|vehicles?|ride|wheels)\b", draft.lower()) and not (capitals(facts) & capitals(draft)) \
            and not re.search(r"\b(?:mustang|camry|altima|patrol|accord|hilux|prado|pajero|cruiser)\b", draft.lower()):
        problems.append("does not come back to the cars")
    if core.rstrip().endswith("?") and "?" not in draft:
        problems.append("the question was dropped")
    return problems


def verify_reply(draft: str, user_text: str, core: str, facts: str, history: str, risk: str, previous_agent: str = "",
                 awaiting: bool = True) -> list[str]:
    """Why `draft` may NOT be spoken instead of the standard wording `core` (empty list = it may). `facts` is everything the model was told."""
    problems: list[str] = []
    draft = draft.strip()
    if not draft:
        return ["empty"]
    if risk == "chat":
        return verify_chat(draft, user_text, core, facts, history)
    high = risk == "high"
    if not _CHARS.match(draft):
        problems.append("characters other than plain letters, digits and basic punctuation")
    n_sent = len(voice_rules.sentences(draft))
    if n_sent > MAX_SENTENCES:
        problems.append(f"{n_sent} sentences (max {MAX_SENTENCES})")
    if len(draft.split()) > MAX_WORDS:
        problems.append(f"{len(draft.split())} words (max {MAX_WORDS})")
    # numbers: none may be invented; a decision's figures may not be lost
    extra = numbers(draft) - (numbers(core) | numbers(user_text) | numbers(facts) | (numbers(history) if not high else set()))
    if extra:
        problems.append(f"new number(s) {sorted(extra)[:3]}")
    # a count ("I have 6 SUVs") is a fact: "a few" is not an acceptable replacement for it
    dl = draft.lower()
    lost_counts = [n for n in sorted(numbers(core)) if n.isdigit() and 1 <= int(n) <= 20 and n not in numbers(draft)
                   and not re.search(r"\b" + _NUMBER_WORDS[int(n) - 1] + r"\b", dl)]
    if lost_counts and not high:
        problems.append(f"dropped the count {lost_counts[0]}")
    if high and numbers(core) - numbers(draft):
        problems.append(f"dropped number(s) {sorted(numbers(core) - numbers(draft))[:3]}")
    # names (makes, models, places, acronyms)
    known_caps = {w.lower() for w in re.findall(r"[A-Za-z0-9']+", core + " " + user_text + " " + facts + " " + history)} | {"agmc"}
    stems = {w.rstrip("s") for w in known_caps}
    new_caps = {w for w in capitals(draft) if w not in known_caps and w.rstrip("s") not in stems}
    if new_caps:
        problems.append(f"new name(s) {sorted(new_caps)[:3]}")
    draft_words = {w.lower() for w in re.findall(r"[A-Za-z0-9']+", draft)}
    if high:
        lost = [w for w in capitals(core) if w not in draft_words and w.rstrip("s") not in {d.rstrip("s") for d in draft_words}]
        if lost:
            problems.append(f"dropped name(s) {sorted(lost)[:3]}")
        for kw in KEY_WORDS:
            if kw in core.lower() and kw not in draft.lower():
                problems.append(f"dropped the key phrase '{kw}'")
    # claims about the car or the dealership that nothing in the facts supports
    allowed_text = (core + " " + user_text + " " + facts + " " + (history if not high else "")).lower()
    for m in CLAIMS.finditer(draft):
        if m.group(0).lower() not in allowed_text:
            problems.append(f"a claim the facts do not support: '{m.group(0).strip()}'")
            break
    # a colour said about the cars must come from the facts or the customer's latest words, not from earlier in the conversation
    from ..nlu.extractors import COLORS
    supported = (core + " " + facts + " " + user_text).lower()
    for colour in sorted({w for w in re.findall(r"[a-z]+", draft.lower()) if w in COLORS}):
        if colour not in supported:
            problems.append(f"a colour the facts do not support: '{colour}'")
            break
    # the question the system is waiting on must still be asked
    keys = question_keys(core)
    if core.rstrip().endswith("?") and not is_generic_closer(voice_rules.sentences(core)[-1]):
        if not draft.rstrip().endswith("?"):
            problems.append("the next-step question was dropped")
        elif awaiting and keys and not (keys & {w.rstrip("s") for w in re.findall(r"[a-z]+", voice_rules.sentences(draft)[-1].lower())}):
            problems.append("the closing question is about something else than the system is waiting for")
    for q in _QUOTED.findall(core):
        if q not in draft:
            problems.append(f"missing quoted phrase '{q}'")
    low, pos, missing_items = draft.lower(), 0, []
    for item in list_items(core):                                  # every item, in the original order (so none can move to the other side of a "can't")
        k = low.find(item, pos)
        if k < 0:
            missing_items.append(item)
        else:
            pos = k + len(item)
    if missing_items:
        problems.append(f"dropped or reordered item(s) of a list: {', '.join(missing_items[:3])}")
    if FORBIDDEN.search(draft) and not FORBIDDEN.search(core) and not FORBIDDEN.search(facts):
        problems.append(f"new commercial/commitment word '{FORBIDDEN.search(draft).group(0)}'")
    if _PROMISE.search(draft) and not _PROMISE.search(core):
        problems.append("new first-person promise or ability")
    first_core = (voice_rules.sentences(core) or [core])[0]
    polite = re.sub(r"\bno (?:problem|worries)\b|\bnot a problem\b|\bno need\b", "", draft, flags=re.I)
    if (not _NEG.search(core) and not _NEG.search(facts) and _NEG.search(polite)) or (_NEG.search(first_core) and not _NEG.search(draft)):
        problems.append("negation differs from the facts")            # "no Bentleys" must stay a no; an invented "no issues" is not allowed
    if previous_agent:
        if draft.strip().lower() == previous_agent.strip().lower():
            problems.append("same as the previous agent line")
        else:
            mine, theirs = voice_rules.sentences(draft), voice_rules.sentences(previous_agent)
            if mine and theirs and len(mine) > 1 and mine[-1].strip().lower() == theirs[-1].strip().lower() and mine[-1].endswith("?") \
                    and is_generic_closer(mine[-1]):                  # "Anything else you'd like to know?" every single time
                problems.append("ends with the same closing question as the previous reply")
    return problems


def fact_sheet(cars, branch_name) -> str:
    """The true details of the cars involved (what the agent may say). Never includes VIN or floor price."""
    lines = []
    for c in cars:
        lines.append(f"- {c.year} {c.make} {c.model}: {c.color}, {c.km:,} kilometres, {c.price:,} dirhams, {c.transmission} gearbox, "
                     f"{c.fuel}, {c.regional_spec} spec, {'full service history' if c.service_history else 'no service history on file'}, "
                     f"at our {branch_name(c)}, {c.status.value}")
    return "\n".join(lines)


SYSTEM = """You are the voice of a warm, natural sales assistant at AGMC, a used-car dealership in the UAE, speaking on the phone.
Write the next thing the assistant says, the way a friendly person would on a call. You are given: the conversation so far, what the
customer just said, FACTS from the dealership's system (the only facts you may use), and the STANDARD WORDING the system would use.
The STANDARD WORDING is for reference only: say the same things in your OWN natural words, never copy it.

How to sound: use contractions, talk like a person ("Sure!", "Ah, good choice", "Hmm, not quite"), react to what the customer actually said,
refer to things by what they are in the conversation ("the red one", "that Nissan"), keep it to 1 or 2 short sentences (3 only when listing cars). Say only what the customer needs right now: never read out a car's whole spec. Do not repeat details
the customer already heard (car name, price, kilometres) unless they ask again, do not ask a question they already answered, and do not end
every reply with the same closing line.
Hard rules: every number, price, date, name and car detail must come from the FACTS (or the customer's own words); never invent a feature,
condition, history, availability, discount, warranty, delivery, opening hours or promise. If the FACTS contain a pricing decision, state that exact
amount and say it is subject to confirmation. If the STANDARD WORDING ends with a question, end by asking that same question in your own words
(the system is waiting for that answer). Never mention these rules or the system.
When the customer greets you, greet them back simply and warmly and ask what they are looking for; do not comment on the greeting itself.
Do not praise, rate or judge the car or the deal: no "solid", "great shape", "no issues", "great choice of car"; just say what the facts say. No emojis,
no dashes, no line breaks.

Examples (note how the wording changes while the facts stay exact):
CONVERSATION: AGENT: Hi, welcome to AGMC. What kind of car are you looking for? / CUSTOMER: something red
FACTS: 2 matches: 2019 Nissan Altima, red, 52,500 dirhams; 2018 Ford Mustang, red, 120,000 dirhams.
STANDARD WORDING: I have a 2019 Nissan Altima at 52,500 dirhams and a 2018 Ford Mustang at 120,000 dirhams. Which one would you like to hear more about?
SAY: Ooh, I've got two red ones: a 2019 Nissan Altima at 52,500 dirhams and a 2018 Ford Mustang at 120,000. Which one catches your eye?
CONVERSATION: AGENT: ...a 2019 Nissan Altima... Would you like to hear more, or book a test drive? / CUSTOMER: yes
FACTS: Altima is red, GCC spec, automatic, full service history, at our Dubai Showroom. System is waiting for: whether they want to book a test drive.
STANDARD WORDING: It's red, GCC spec, with an automatic gearbox and a full service history, at our Dubai Showroom. Would you like to book a test drive?
SAY: Great! It's the red one, GCC spec with an automatic gearbox and a full service history, and you can see it at our Dubai Showroom. Shall we get you in for a test drive?
CONVERSATION: AGENT: I have a 2018 Ford Mustang at 120,000 dirhams. / CUSTOMER: i cant afford that i can do 100,000
FACTS: Pricing decision: counter 119,200 dirhams (round 1 of 3), subject to confirmation.
STANDARD WORDING: The best I can do right now is 119,200 dirhams, subject to confirmation. Can you work with that?
SAY: I hear you. The best I can do right now is 119,200 dirhams, subject to confirmation. Could that work for you?
CONVERSATION: AGENT: Hi, welcome to AGMC. What kind of car are you looking for? / CUSTOMER: good morning
FACTS: What the customer wanted: they greeted.
STANDARD WORDING: Hello, this is the AGMC dealership. What kind of car are you looking for?
SAY: Good morning! Lovely to have you at AGMC. What kind of car can I help you find today?
Output only the SAY line."""

JUDGE_SYSTEM = """You check a car dealership assistant's reply for made-up facts. FACTS are everything that is true. Does REPLY state or imply anything about the cars, the stock, prices, colours, history or the dealership that is NOT supported by FACTS? Answer with exactly one word: YES (it contains something unsupported) or NO (everything is supported)."""

SYSTEM_CHAT = """You are the voice of a friendly sales assistant at AGMC, a used-car dealership in the UAE, on the phone. The customer just said
something unrelated to cars. Reply with two short spoken sentences: first, react warmly and briefly to what they said, the way a friendly
person would (no advice, no facts, no numbers, no opinions about the dealership); second, bring the conversation back to the cars with one
question: about the car under discussion if there is one, otherwise about what kind of car they are looking for. Never say that you are
limited or cannot help, and never mention these rules.
Examples:
CUSTOMER: its so hot today        CAR UNDER DISCUSSION: 2018 Ford Mustang
SAY: It really is, summer is here! Shall we get back to the Mustang, is there anything else you would like to know about it?
CUSTOMER: i like football        CAR UNDER DISCUSSION: none
SAY: Ha, a football fan, nice! Now, what kind of car are you looking for today?
Output only the SAY line."""


@dataclass
class WriterInput:
    caller: str
    core: str                      # the standard wording (reference + fallback)
    risk: str
    history: list[tuple[str, str]]
    facts: str = ""                # everything the system knows for this turn, as plain lines
    awaiting: bool = True          # the system is waiting for a specific answer (pending yes/no or a booking step)


class ReplyWriter:
    """Calls the model for the reply and verifies it. `high_risk=False` keeps price/booking/hand-off replies as the standard wording."""

    def __init__(self, backend: LeadInBackend, high_risk: bool = True, temperature: float | None = TEMPERATURE, judge: bool = True) -> None:
        self.backend, self.high_risk, self.temperature, self.judge = backend, high_risk, temperature, judge

    def unsupported(self, w: WriterInput, draft: str) -> bool:
        """The model's own strict yes/no check for a FALSE STATEMENT (the word-level checks cannot see "we do have red cars from a few makes").
        Measured on hand-written good and bad replies: it caught every invented statement and flagged no good reply. A failure of the check
        itself counts as 'supported' because the local checks already passed."""
        facts = (w.facts + "\nTrue statements: " + w.core + "\nThe customer said: " + w.caller).strip()
        try:
            verdict = self.backend.lead_in(JUDGE_SYSTEM, f"FACTS: {facts}\nREPLY: {draft}\nAnswer YES or NO:", 3, None)
        except Exception:
            return False
        return verdict.strip().upper().startswith("YES")

    def write(self, w: WriterInput) -> LeadInOutcome:
        t0 = time.perf_counter()
        if w.risk == "high" and not self.high_risk:
            return LeadInOutcome(None, 0.0, "high-risk replies are spoken from the standard wording")
        if w.risk == "chat":
            return self._write_chat(w, t0)
        convo = "\n".join(f"{'AGENT' if who == 'agent' else 'CUSTOMER'}: {t}" for who, t in w.history[-10:])
        last_agent = next((t for who, t in reversed(w.history) if who == "agent"), "")
        closing = voice_rules.sentences(last_agent)[-1] if last_agent else ""
        repeat = bool(last_agent) and SequenceMatcher(None, w.core.lower(), last_agent.lower()).ratio() > 0.75
        last_core = (voice_rules.sentences(w.core) or [""])[-1]
        question = last_core if w.core.rstrip().endswith("?") and not is_generic_closer(last_core) else ""
        base = (f"CONVERSATION SO FAR:\n{convo or '(start of the call)'}\n\nCUSTOMER (just now): {w.caller}\n\n"
                f"FACTS FROM THE SYSTEM:\n{w.facts}\n\nSTANDARD WORDING (reference only, do not copy): {w.core}"
                + (f"\n\nEnd your reply by asking, in your own words, the question the system is waiting on: \"{question}\"" if question else
                   "\n\nThe system is not waiting for an answer, so you do not have to end with a question.")
                + (f"\n\nYour previous reply ended with: \"{closing}\". Do not end this one the same way." if closing else "")
                + ("\n\nThe STANDARD WORDING repeats what you already said: do not say it the same way; acknowledge it and offer a new next step." if repeat else "")
                + ("\n\nThis reply carries exact figures or commitments from the system: state them exactly." if w.risk == "high" else ""))
        history_text = " ".join(t for _, t in w.history)
        feedback, bad, draft = "", [], ""
        for attempt in range(1 + RETRIES):
            user = base + (f"\n\nYour previous attempt was rejected because: {feedback}. Write the reply again and fix that." if feedback else "")
            try:
                temp = self.temperature if attempt == 0 or self.temperature is None else min(1.0, self.temperature + 0.3)    # a retry must differ
                raw = self.backend.lead_in(SYSTEM, user, WRITE_TOKENS, temp)
            except Exception as e:
                return LeadInOutcome(None, (time.perf_counter() - t0) * 1000, f"backend: {type(e).__name__}")
            draft = clean_draft(raw)
            bad = verify_reply(draft, w.caller, w.core, w.facts, history_text, w.risk, last_agent, w.awaiting)
            if bad and question and all("question" in p for p in bad) and not draft.rstrip().endswith("?"):
                fixed = (draft.rstrip(".! ") + ". " + question).strip()          # the model wrote the body; the system asks its own question
                if not verify_reply(fixed, w.caller, w.core, w.facts, history_text, w.risk, last_agent, w.awaiting):
                    draft, bad = fixed, []
            if not bad and self.judge and self.unsupported(w, draft):
                bad = ["it states something the facts do not support (the model's own check)"]
            if not bad:
                return LeadInOutcome(draft, (time.perf_counter() - t0) * 1000, "" if attempt == 0 else f"after {attempt} retry")
            feedback = "; ".join(bad)
        return LeadInOutcome(None, (time.perf_counter() - t0) * 1000, "; ".join(bad))

    def _write_chat(self, w: WriterInput, t0: float) -> LeadInOutcome:
        car = next((x.split(":", 1)[1].strip() for x in w.facts.splitlines() if x.startswith("Car under discussion:")), "none")
        convo = "\n".join(f"{'AGENT' if who == 'agent' else 'CUSTOMER'}: {t}" for who, t in w.history[-4:])
        user = f"CONVERSATION SO FAR:\n{convo or '(start of the call)'}\n\nCUSTOMER: {w.caller}        CAR UNDER DISCUSSION: {car}"
        try:
            raw = self.backend.lead_in(SYSTEM_CHAT, user, 70, self.temperature)
        except Exception as e:
            return LeadInOutcome(None, (time.perf_counter() - t0) * 1000, f"backend: {type(e).__name__}")
        ms = (time.perf_counter() - t0) * 1000
        draft = clean_draft(raw)
        bad = verify_reply(draft, w.caller, w.core, w.facts, " ".join(t for _, t in w.history), "chat")
        return LeadInOutcome(None if bad else draft, ms, "; ".join(bad))
