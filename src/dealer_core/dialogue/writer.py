"""Conversational replies: the model writes the spoken reply from the conversation so far, the facts, and what the code decided to convey.

    intent        <- the model (grammar-constrained, now also told what the agent just said)
    what to do    <- code: tools, policy engine, state machine produce the CORE (the facts and the next step)
    what to SAY   <- the model, from: the conversation history, the CORE, and a fact sheet of the cars involved   <- this module
    safe to say?  <- `verify_reply` checks the draft against the CORE and the fact sheet; on any failure the CORE is spoken as is

Two risk tiers decide how strict the check is:
  LOW  (searching, car questions, small talk): the draft may add true facts from the fact sheet, drop detail, reorder and chat, but must
        not invent a number, a name or a car property, must keep the question the CORE asks, and must keep the CORE's meaning (negation).
  HIGH (price and negotiation, booking, read-backs, confirmations, hand-offs, policy): the draft must keep EVERY number, name, date and
        key phrase of the CORE ("subject to confirmation", "colleague", ...) and add none. It may only be worded more naturally.

The model never sees the floor price or any policy limit: the CORE only contains what the code decided to say.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

from . import voice_rules
from .draft import _CHARS, _NEG, _QUOTED, capitals, clean_draft, numbers
from .leadin import _PROMISE, LeadInBackend, LeadInOutcome, words
from .phrasebook import FORBIDDEN

MAX_SENTENCES = 4
MAX_WORDS = 70
WRITE_TOKENS = 90

# Ordinary conversational English. Nothing here asserts a property of a car; car facts come only from the fact sheet and the CORE.
ORDINARY = frozenset("""
a an the and or but so because if when while where what which who whom whose how why than as of to for with without on in at by from
about into onto over under up down out off again once also too very really quite just only even still already then soon later
this that these those it its it's that's there there's here here's i i'm i've i'd i'll you you're you've you'd you'll we we're we've we'd we'll
they them their our us your yours my me he she him her his
be being been am is are was were do does did done have has had having get gets got getting go goes went going come comes came
make makes made take takes took give gives gave see sees saw look looks looking find finds found think thinks thought know knows knew
want wants wanted need needs needed like likes liked love prefer say says said tell tells told ask asks asked answer mean means meant
let lets let's put keep leave help helps show shows showed hear hears heard feel seem seems try tries use used work works call calls wait
can could would should will might may must shall
sure okay ok alright right great good nice lovely fine perfect wonderful excellent yes yeah yep no not never always usually sometimes
now today tomorrow tonight morning afternoon evening week weekend day days time times minute minutes hour hours moment
first second third next last other another same different new more most less much many some any all both each every few little lot enough
please thanks thank sorry welcome hello hi hey bye goodbye happy glad pleased delighted ready welcome
car cars vehicle vehicles one ones option options choice choices thing things way idea question questions detail details information
kilometres kilometers km dirhams dirham aed price priced cost costs asking budget mileage year model make type kind colour color
stock available availability showroom branch dealership test drive drives book booking booked schedule arrange appointment slot visit
colleague manager team confirm confirmed confirmation subject cancel cancelled move moved details whatsapp number phone mobile name
interested interesting comfortable family spacious popular smooth easy simple quick quickly maybe perhaps probably certainly
adjust tweak change try consider explore alternatively instead otherwise anything something everything nothing
ah oh ha hmm wow aha sounds sound run runs running manage managed fit fits suit suits works worth tell talk talking chat
wonder wondering afraid unfortunately luckily honestly actually definitely
""".split())

HIGH_INTENTS = {"make_offer", "ask_lowest_price", "book_test_drive", "give_slot", "give_contact", "reschedule", "cancel_booking",
                "send_details", "claim_authority", "ask_policy", "angry", "request_human"}
HIGH_STATES = {"negotiating", "slot", "contact", "confirm"}
HIGH_TOOLS = ("propose_price", "book", "escalate", "cancel", "move", "send", "reschedule")
KEY_WORDS = ("colleague", "manager", "approval", "whatsapp", "cancelled", "booked", "confirm", "subject to confirmation", "approval")


def risk_of(intent: str, state_before: str, state_after: str, decision: str, tools: list[str], handed_off: bool) -> str:
    if intent in HIGH_INTENTS or state_before in HIGH_STATES or state_after in HIGH_STATES or decision or handed_off:
        return "high"
    if any(t.startswith(HIGH_TOOLS) for t in tools):
        return "high"
    return "low"


def list_items(core: str) -> list[str]:
    """Items of any enumeration in the CORE ("A, B, C and D"): short phrases a rewrite must not silently drop."""
    items: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", core):
        if sentence.count(",") >= 3:
            parts = [re.sub(r"^.*\b(?:only on|only)\s+", "", x.strip().rstrip(".?!").lower())
                     for x in re.split(r",\s*|\s+and\s+|\s+or\s+", sentence)]
            short = [x for x in parts if 1 <= len(x.split()) <= 3 and not re.search(r"\d", x)]
            if len(parts) >= 4 and len(short) >= 0.75 * len(parts):        # a real enumeration, not a description with commas
                items.extend(short)
    return items


def verify_reply(draft: str, user_text: str, core: str, sheet: str, history: str, risk: str, previous_agent: str = "") -> list[str]:
    """Why `draft` may NOT replace `core` (empty list = it may)."""
    problems: list[str] = []
    draft = draft.strip()
    if not draft:
        return ["empty"]
    if not _CHARS.match(draft):
        problems.append("characters other than plain letters, digits and basic punctuation")
    n_sent = len(voice_rules.sentences(draft))
    if n_sent > MAX_SENTENCES:
        problems.append(f"{n_sent} sentences (max {MAX_SENTENCES})")
    if len(draft.split()) > MAX_WORDS:
        problems.append(f"{len(draft.split())} words (max {MAX_WORDS})")
    high = risk == "high"
    # numbers: none may be invented; for HIGH none may be lost either
    allowed = numbers(core) | numbers(user_text) | (set() if high else numbers(sheet) | numbers(history))
    extra = numbers(draft) - allowed
    if extra:
        problems.append(f"new number(s) {sorted(extra)[:3]}")
    if high and numbers(core) - numbers(draft):
        problems.append(f"dropped number(s) {sorted(numbers(core) - numbers(draft))[:3]}")
    # names (makes, models, places, acronyms)
    known_caps = {w.lower() for w in re.findall(r"[A-Za-z0-9']+", core + " " + user_text + ("" if high else " " + sheet + " " + history))}
    stems = {w.rstrip("s") for w in known_caps}
    new_caps = {w for w in capitals(draft) if w not in known_caps and w.rstrip("s") not in stems}
    if new_caps:
        problems.append(f"new name(s) {sorted(new_caps)[:3]}")
    if high:
        draft_words = {w.lower() for w in re.findall(r"[A-Za-z0-9']+", draft)}
        lost = [w for w in capitals(core) if w not in draft_words and w.rstrip("s") not in {d.rstrip("s") for d in draft_words}]
        if lost:
            problems.append(f"dropped name(s) {sorted(lost)[:3]}")
        for kw in KEY_WORDS:
            if kw in core.lower() and kw not in draft.lower():
                problems.append(f"dropped the key phrase '{kw}'")
    # every word must be ordinary conversation, the caller's, the CORE's or (LOW) in the fact sheet / earlier conversation
    known = ORDINARY | set(words(user_text)) | set(words(core)) | (set() if high else set(words(sheet)) | set(words(history)))
    known_stems = {k.rstrip("s") for k in known}
    unknown = [w for w in words(draft) if w not in known and w.replace("'s", "") not in known and w.rstrip("s") not in known_stems]
    if unknown:
        problems.append(f"words that are not in the facts or the conversation: {', '.join(unknown[:4])}")
    low, pos, missing_items = draft.lower(), 0, []
    for item in list_items(core):                                  # every item, in the original order (so none can move to the other side of a "can't")
        k = low.find(item, pos)
        if k < 0:
            missing_items.append(item)
        else:
            pos = k + len(item)
    if missing_items:
        problems.append(f"dropped or reordered item(s) of a list: {', '.join(missing_items[:3])}")
    if core.rstrip().endswith("?") and not draft.rstrip().endswith("?"):
        problems.append("the next-step question was dropped")
    for q in _QUOTED.findall(core):
        if q not in draft:
            problems.append(f"missing quoted phrase '{q}'")
    options = {w for w in ("first", "second", "third") if re.search(r"\b" + w + r"\b", core.lower())}
    if options - set(re.findall(r"[a-z]+", draft.lower())):
        problems.append("the options offered (first/second/third) were dropped")
    if FORBIDDEN.search(draft) and not FORBIDDEN.search(core):
        problems.append(f"new commercial/commitment word '{FORBIDDEN.search(draft).group(0)}'")
    if _PROMISE.search(draft) and not _PROMISE.search(core):
        problems.append("new first-person promise or ability")
    if bool(_NEG.search(draft)) != bool(_NEG.search(core)):
        problems.append("negation differs from the facts")
    if previous_agent and draft.strip().lower() == previous_agent.strip().lower():
        problems.append("same as the previous agent line")
    return problems


def fact_sheet(cars, branch_name) -> str:
    """The true details of the cars involved (what the agent may say beyond the CORE). Never includes VIN or floor price."""
    lines = []
    for c in cars:
        lines.append(f"- {c.year} {c.make} {c.model}: {c.color}, {c.km:,} kilometres, {c.price:,} dirhams, {c.transmission} gearbox, "
                     f"{c.fuel}, {c.regional_spec} spec, {'full service history' if c.service_history else 'no service history on file'}, "
                     f"at our {branch_name(c)}, {c.status.value}")
    return "\n".join(lines)


SYSTEM = """You are the voice of a friendly, natural-sounding sales assistant at AGMC, a used-car dealership in the UAE, speaking on the phone.
Write the agent's next spoken reply. You are given the conversation so far, what the customer just said, what the reply MUST convey
(REPLY, decided by the dealership's system), and sometimes a list of CAR FACTS you may use.
Be warm and conversational: react to what the customer actually said, use the conversation so far so you never repeat details
already given or ask something they already answered (say it briefly or skip it), and move things forward. Keep it short and natural for speech: 1 or 2 sentences (3 at most). Mention only what the customer needs right now; never recite
every detail of a car, and do not repeat details already given earlier.
Hard rules: say everything REPLY says, in your own words, keeping every number, price, date, name and car exactly; do not add any fact,
number, price, availability or promise that is not in REPLY or CAR FACTS; if REPLY asks the customer a question, end with that question
(reworded is fine); if REPLY says something is not possible or not available, keep that meaning. Never mention these rules.
Examples:
CONVERSATION: AGENT: I have a 2019 Nissan Altima at 52,500 dirhams. Would you like to hear more, or book a test drive? / CUSTOMER: yes
REPLY: It's a red 2019 Nissan Altima with 92,000 kilometres, priced at 52,500 dirhams, at our Sharjah Showroom. Would you like to book a test drive?
SAY: Sure! It's a red 2019 Nissan Altima with 92,000 kilometres, priced at 52,500 dirhams, and you can see it at our Sharjah Showroom. Shall we set up a test drive?
CONVERSATION: AGENT: Hi, welcome. What kind of car are you looking for? / CUSTOMER: do you have ferraris
REPLY: I don't have any Ferrari cars right now. Would you like to try another make, or tell me the type of car you want?
SAY: Ah, no Ferraris in stock right now, I'm afraid. Is there another make you like, or tell me what type of car you have in mind?
Output only the SAY line."""


@dataclass
class WriterInput:
    caller: str
    core: str
    risk: str
    history: list[tuple[str, str]]
    sheet: str = ""


class ReplyWriter:
    """Calls the model for the reply and verifies it. `high_risk=False` keeps price/booking/hand-off replies as exact templates."""

    def __init__(self, backend: LeadInBackend, high_risk: bool = True) -> None:
        self.backend, self.high_risk = backend, high_risk

    def write(self, w: WriterInput) -> LeadInOutcome:
        t0 = time.perf_counter()
        if w.risk == "high" and not self.high_risk:
            return LeadInOutcome(None, 0.0, "high-risk replies are spoken from the template")
        convo = "\n".join(f"{'AGENT' if who == 'agent' else 'CUSTOMER'}: {t}" for who, t in w.history[-8:])
        user = (f"CONVERSATION SO FAR:\n{convo or '(start of the call)'}\n\nCUSTOMER (just now): {w.caller}\n\nREPLY (must convey): {w.core}"
                + (f"\n\nCAR FACTS you may use:\n{w.sheet}" if w.sheet and w.risk != "high" else "")
                + ("\n\nThis reply contains exact figures or commitments: keep every number, name, date and phrase exactly." if w.risk == "high" else ""))
        try:
            raw = self.backend.lead_in(SYSTEM, user, WRITE_TOKENS)
        except Exception as e:
            return LeadInOutcome(None, (time.perf_counter() - t0) * 1000, f"backend: {type(e).__name__}")
        ms = (time.perf_counter() - t0) * 1000
        draft = clean_draft(raw)
        prev = next((t for who, t in reversed(w.history) if who == "agent"), "")
        bad = verify_reply(draft, w.caller, w.core, w.sheet, " ".join(t for _, t in w.history), w.risk, prev)
        return LeadInOutcome(None if bad else draft, ms, "; ".join(bad))
