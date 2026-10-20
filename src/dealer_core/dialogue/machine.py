"""Dialogue engine: a deterministic state machine. The LLM (inside the NLU pipeline) only chooses the intent; every reply is
templated, every number comes from the tools / policy engine, and every state change that matters is read back first.

    greeting -> interest -> answering -> negotiating
                                      -> slot -> contact -> confirm (read-back) -> booked
    any state -> handoff (angry, asked for a person, below-floor price, repeated off-topic, system fault)

`handle()` returns the reply plus a TurnDebug (state, intent, rules-or-LLM, tools, policy decision, timings) for `--debug`.
"""
from __future__ import annotations

import re
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..models import CarStatus, CarView, SearchFilters
from ..nlu import extractors as ex
from ..nlu.backend import BackendError
from ..nlu.datetime_parse import combine, parse_when, part_of_day
from ..nlu.pipeline import NluPipeline, Understanding
from ..nlu.rules import TOPICS
from ..policy.negotiation import DecisionKind
from ..tools.escalation import EscalationReason
from ..tools.phones import speak_phone
from ..tools.results import BookingKind
from ..tools.service import DealerService
from . import templates as T
from .leadin import LeadInWriter
from .phrasebook import Phrasebook
from . import voice_rules
from .session import Pending, Session
from .states import Intent, State


@dataclass
class TurnDebug:
    state_before: str = ""
    state_after: str = ""
    intent: str = ""
    via: str = ""                      # "rules:<rule>" | "llm:<backend>" | "language"
    tools: list[str] = field(default_factory=list)
    decision: str = ""                 # policy-engine decision, when a price was discussed
    nlu_ms: float = 0.0
    total_ms: float = 0.0
    violations: list[str] = field(default_factory=list)
    note: str = ""
    lead_in: str = ""                  # the model-written lead-in sentence, when a hybrid reply used one
    lead_ms: float = 0.0


@dataclass
class TurnResult:
    reply: str
    debug: TurnDebug
    ended: bool = False


class DialogueEngine:
    def __init__(self, svc: DealerService, nlu: NluPipeline, phrasebook: Phrasebook | None = None, variant_seed: int = 7,
                 lead_writer: LeadInWriter | None = None) -> None:
        self.svc, self.nlu, self.writer = svc, nlu, lead_writer
        self.phrases, self.variant_seed = phrasebook or Phrasebook(), variant_seed

    def _say(self, s: Session, sid: str, **values: object) -> str:
        """Base wording or an approved variant for a low-risk situation (see phrasebook.py)."""
        return self.phrases.say(s.rng, s.last_variant, sid, **values)

    # ------------------------------------------------------------------------------------------------ entry points
    def new_session(self) -> Session:
        return Session(rng=random.Random(self.variant_seed))

    def opening(self, s: Session) -> str:
        return self._say(s, "opening")

    def handle(self, s: Session, text: str) -> TurnResult:
        t0 = time.perf_counter()
        dbg = TurnDebug(state_before=s.state.value)
        if s.ended:
            dbg.note = "session already ended"
            return TurnResult(T.HANDOFF_WAIT if s.handoff else self._say(s, "goodbye"), dbg, True)
        try:
            u = self.nlu.understand(s.state, s.car, text)
        except (BackendError, RuntimeError) as e:        # every NLU backend failed: never leave the caller hanging
            dbg.note = f"NLU unavailable: {e}"
            reply = self._handoff(s, dbg, EscalationReason.SYSTEM_FAULT, "I'm having a technical problem. Let me connect you with a colleague.")
            return self._finish(s, dbg, reply, t0)
        dbg.intent, dbg.nlu_ms = u.intent.value, u.ms
        if u.repaired:
            dbg.note = f"intent repaired by a consistency check ({u.repaired})"
        dbg.via = {"rules": f"rules:{u.rule}", "llm": f"llm:{u.backend}"}.get(u.via, u.via)
        reply = self._dispatch(s, u, text, dbg)
        reply = self._with_lead_in(s, u.intent.value, text, dbg, reply)
        return self._finish(s, dbg, reply, t0)

    def _with_lead_in(self, s: Session, intent: str, text: str, dbg: TurnDebug, reply: str) -> str:
        """Hybrid mode: put a verified model-written sentence in front of a low-risk templated reply (see leadin.py)."""
        if self.writer is None or s.ended or intent not in self.writer.intents:
            return reply
        out = self.writer.write(text, reply, s.last_lead)
        dbg.lead_ms = out.ms
        if out.text:
            s.last_lead, dbg.lead_in = out.text, out.text
            return f"{out.text} {reply}"
        dbg.note = (dbg.note + "; " if dbg.note else "") + f"lead-in dropped ({out.why_not})"
        return reply

    def _finish(self, s: Session, dbg: TurnDebug, reply: str, t0: float) -> TurnResult:
        dbg.state_after = s.state.value
        dbg.violations = voice_rules.check(reply, max_sentences=voice_rules.MAX_SENTENCES + (1 if dbg.lead_in else 0))
        dbg.total_ms = (time.perf_counter() - t0) * 1000
        return TurnResult(reply, dbg, s.ended)

    # ------------------------------------------------------------------------------------------------ helpers
    def _now(self) -> datetime:
        return self.svc.clock()

    def _tool(self, dbg: TurnDebug, call: str, result: str) -> None:
        dbg.tools.append(f"{call} -> {result}")

    def _car(self, s: Session) -> CarView | None:
        return self.svc.get_car(s.car) if s.car else None

    def _branch_name(self, c: CarView) -> str:
        return self.svc.slots.branch(c.branch_id).name

    def _resolve_car(self, s: Session, text: str) -> CarView | None:
        """The car in focus, or the one the caller points at among those just shown ('the second one', 'the Patrol')."""
        pick = ex.selection(text, s.shown)
        if pick:
            s.car = pick
        return self._car(s)

    def _ask_which(self, s: Session) -> str:
        n = len(s.shown)
        if n == 2:
            return "Which one do you mean: the first or the second?"
        if n >= 3:
            return "Which one do you mean: the first, the second or the third?"
        return "Which car are you asking about?"

    def _handoff(self, s: Session, dbg: TurnDebug, reason: EscalationReason, say: str, offer: int | None = None) -> str:
        h = self.svc.escalate_to_human(reason, s.car, offer, note=dbg.note or reason.value)
        self._tool(dbg, f"escalate_to_human({reason.value})", h.handoff_id)
        s.handoff, s.ended, s.state, s.pending = h, True, State.GOODBYE, None
        return say

    # ------------------------------------------------------------------------------------------------ dispatch
    def _dispatch(self, s: Session, u: Understanding, text: str, dbg: TurnDebug) -> str:
        intent = u.intent
        if u.via == "language":
            s.pending = Pending("handoff_offer")
            return "I'm sorry, I can only help in English at the moment. Would you like me to connect you with a colleague?"
        if intent is not Intent.OUT_OF_SCOPE:
            s.oos_streak = 0
        if s.pending:
            r = self._pending(s, intent, text, dbg)
            if r is not None:
                return r
        # ---- intents that mean the same thing in every state
        if intent is Intent.ANGRY:
            return self._handoff(s, dbg, EscalationReason.ANGRY, "I'm sorry about that. Let me connect you with a colleague right away.")
        if intent is Intent.REQUEST_HUMAN:
            return self._handoff(s, dbg, EscalationReason.REQUESTED_HUMAN, "Of course. I'm connecting you with a colleague now.")
        if intent is Intent.GOODBYE:
            s.ended, s.state = True, State.GOODBYE
            return self._say(s, "goodbye")
        if intent is Intent.OUT_OF_SCOPE:
            if s.filters and ex.mentions_unfilterable(text):       # "make it automatic": a car wish we cannot search on, not off-topic
                dbg.note = "car requirement we cannot filter on: said so instead of treating it as off-topic"
                return self._say(s, "cannot_filter")
            return self._off_topic(s)
        if intent is Intent.CLAIM_AUTHORITY:
            dbg.note = "authority claim ignored: limits are unchanged"
            return "I understand, but I can't change a price based on who is calling. I'm happy to help with the car, though."
        if intent is Intent.ASK_POLICY:
            return self._policy(s, text)
        if intent is Intent.GREET:
            if s.state is State.GREETING:
                s.state = State.INTEREST
                return self._say(s, "opening")
            return self._say(s, "hello_again")
        # ---- state-specific
        if intent is Intent.SEARCH_CARS:
            return self._search(s, text, dbg)
        if intent is Intent.ASK_CAR_INFO:
            return self._car_info(s, text, dbg)
        if intent in (Intent.MAKE_OFFER, Intent.ASK_LOWEST_PRICE):
            return self._price(s, intent, text, dbg)
        if intent is Intent.BOOK_TEST_DRIVE:
            return self._start_booking(s, text, dbg)
        if intent is Intent.GIVE_SLOT:
            return self._give_slot(s, text, dbg)
        if intent is Intent.GIVE_CONTACT:
            return self._give_contact(s, text, dbg)
        if intent is Intent.SEND_DETAILS:
            return self._send_details(s, text, dbg)
        if intent is Intent.RESCHEDULE:
            return self._reschedule(s, text, dbg)
        if intent is Intent.CANCEL_BOOKING:
            return self._cancel(s, dbg)
        if intent is Intent.CONFIRM_YES:
            return self._yes(s, dbg)
        if intent is Intent.CONFIRM_NO:
            return self._no(s, text, dbg)
        return self._say(s, "didnt_catch")

    # ------------------------------------------------------------------------------------------------ pending yes/no
    def _pending(self, s: Session, intent: Intent, text: str, dbg: TurnDebug) -> str | None:
        p = s.pending
        assert p is not None
        if intent not in (Intent.CONFIRM_YES, Intent.CONFIRM_NO):
            s.pending = None                       # the caller moved on; handle the new intent normally
            return None
        s.pending = None
        yes = intent is Intent.CONFIRM_YES
        if p.kind == "handoff_offer":
            if yes:
                return self._handoff(s, dbg, EscalationReason.OUT_OF_SCOPE, "Of course. I'm connecting you with a colleague now.")
            return "No problem. What kind of car are you looking for?"
        if p.kind == "amount_check":
            if yes:
                return self._price_for(s, p.value, dbg)
            return "Sorry about that. What price did you have in mind?"
        if p.kind == "offer_booking":
            if yes:
                return self._start_booking(s, "", dbg)
            return "No problem. Is there anything else I can help with?"
        if p.kind == "move_existing":
            if yes:
                s.rescheduling, s.state = p.value, State.SLOT
                return "Of course. Which day and time would you like instead?"
            return "Okay, I'll keep your existing booking. Is there anything else?"
        return None

    # ------------------------------------------------------------------------------------------------ small talk, policy, off-topic
    def _off_topic(self, s: Session) -> str:
        s.oos_streak += 1
        if s.oos_streak >= 2:
            s.pending = Pending("handoff_offer")
            return "I can only help with our used cars. Would you like me to connect you with a colleague?"
        return self._say(s, "off_topic_ask_car" if s.car is None else "off_topic_this_one")

    def _policy(self, s: Session, text: str) -> str:
        topic = next((t for t, rx in TOPICS.items() if re.search(rx, " ".join(ex.tokens_of(text)))), None)
        label = T.TOPIC_LABEL.get(topic or "")
        s.pending = Pending("handoff_offer")
        what = f"our confirmed {label} details" if label else "confirmed details on that"
        return f"I don't have {what} with me. Would you like me to connect you with a colleague?"

    # ------------------------------------------------------------------------------------------------ search
    def _search(self, s: Session, text: str, dbg: TurnDebug) -> str:
        vocab = self.svc.inventory.vocabulary()
        new = ex.search_filters(text, vocab)
        others = ex.wants_others(text)
        if ex.wants_listing(text) and s.filters and s.narrow_asks > 0:
            s.narrow_asks = 2                       # the caller does not want more questions: read out the matches
            new = {}
        elif not new and not others and s.filters and ex.points_at_a_car(text):
            return self._ask_which(s)                   # "tell me about the second one" with nothing on screen
        elif not new and not others and s.filters:
            dbg.note = "search request with no filter I can apply: said so instead of repeating the last answer"
            return self._say(s, "cannot_filter")
        # A named model, or a make said out of the blue, starts a fresh search ("do you have a Land Cruiser?"). A make said as the
        # answer to a narrowing question ("Toyota please") refines the current one instead.
        if others and not new and (s.car or s.shown):       # "I want another car instead": drop the car they were on, ask what they want
            s.excluded |= ({s.car} if s.car else set()) | {c.stock_id for c in s.shown}
            s.filters.pop("make", None)
            s.filters.pop("model", None)
            if not s.filters:
                s.car, s.shown, s.state, s.narrow_asks = None, [], State.INTEREST, 0
                return self._say(s, "ask_instead")
        if new.get("model") or (new.get("make") and s.narrow_asks == 0):
            s.filters = dict(new)
            s.excluded.clear()
        else:
            s.filters.update(new)
        if not s.filters and not others:
            return self._say(s, "ask_what_car")
        if others:
            s.excluded |= {c.stock_id for c in s.shown}
        f = SearchFilters(max_price=s.filters.get("max_price"), make=s.filters.get("make"), model=s.filters.get("model"),
                          body_type=s.filters.get("body_type"), color=s.filters.get("color"), year_min=s.filters.get("year_min"), max_km=s.filters.get("max_km"),
                          exclude_stock_ids=tuple(sorted(s.excluded)))
        res = self.svc.search_cars(f, limit=3)
        self._tool(dbg, f"search_cars({', '.join(f'{k}={v}' for k, v in s.filters.items())}{', exclude=' + str(len(s.excluded)) if s.excluded else ''})",
                   f"{res.total} match(es), reading {len(res.cars)}")
        s.state = State.ANSWERING
        if res.total == 0:
            s.state, s.shown, s.car = State.INTEREST, [], None
            return self._say(s, "no_match", desc=T.describe_filters(s.filters))
        if res.needs_narrowing and s.narrow_asks < 2:
            s.narrow_asks += 1
            s.state, s.shown, s.car = State.INTEREST, [], None
            sid, vals = T.narrowing_situation(res.total, s.filters, s.narrow_asks - 1)
            return self._say(s, sid, **vals)
        s.narrow_asks = 0
        s.shown = list(res.cars)
        s.car = res.cars[0].stock_id if len(res.cars) == 1 else None
        reply = T.cars_listed(list(res.cars))
        if ex.asks_if_only_one(text):             # "do you only have one Camry?": answer the question first
            if res.total == 1:
                reply = (f"Yes, that's the only {T.car_name(res.cars[0])} I have right now. "
                         "Would you like to hear more, or book a test drive?")
            else:
                reply = reply.replace("I have", "No, I have", 1)
        return reply

    # ------------------------------------------------------------------------------------------------ car questions
    def _car_info(self, s: Session, text: str, dbg: TurnDebug) -> str:
        car = self._resolve_car(s, text)
        if car is None:
            return self._ask_which(s)
        s.state = State.ANSWERING
        keys = ex.attributes_asked(text)
        self._tool(dbg, f"get_car({car.stock_id})", f"{car.status.value}, {car.km:,} km, {car.color}")
        branch = self._branch_name(car)
        claims = ex.claims_in_question(text)
        if claims:
            return self._confirm_claims(car, claims)
        if not keys:
            return T.car_summary(car, branch)
        if "accident" in keys:
            others = [k for k in keys if k != "accident"]
            tail = f" {T.cap(T.attribute_clause(others[0], car, branch))}." if others else ""
            return "I don't have accident records with me, but I can have a colleague confirm." + tail
        clauses = [c for k in keys[:2] if (c := T.attribute_clause(k, car, branch))]
        return T.cap(clauses[0]) + (f" and {clauses[1]}" if len(clauses) > 1 else "") + ". Anything else you'd like to know?"

    @staticmethod
    def _confirm_claims(car, claims: dict[str, str]) -> str:
        """A yes/no question about the car ("is it red?"): answer yes or no first, then say what it actually is."""
        actual = {"color": car.color, "year": str(car.year), "transmission": car.transmission.lower(), "fuel": car.fuel.lower(),
                  "spec": car.regional_spec.upper()}
        words = {"color": "{}", "year": "a {} model", "transmission": "{}", "fuel": "{}", "spec": "{} spec"}
        wrong = [k for k, v in claims.items() if actual[k].lower() != v.lower()]
        if wrong:
            k = wrong[0]
            return f"No, it's {words[k].format(actual[k])}, not {words[k].format(claims[k])}. Anything else you'd like to know?"
        said = " and ".join(words[k].format(actual[k]) for k in list(claims)[:2])
        return f"Yes, it's {said}. Anything else you'd like to know?"

    # ------------------------------------------------------------------------------------------------ price
    def _price(self, s: Session, intent: Intent, text: str, dbg: TurnDebug) -> str:
        car = self._resolve_car(s, text)
        if car is None:
            return self._ask_which(s)
        s.state = State.NEGOTIATING
        if intent is Intent.ASK_LOWEST_PRICE:
            return self._price_for(s, None, dbg)
        amount = ex.offer_amount(text)
        if amount is None:
            return "What price did you have in mind?"
        return self._price_for(s, amount, dbg)

    def _price_for(self, s: Session, offer: int | None, dbg: TurnDebug) -> str:
        assert s.car is not None
        s.state = State.NEGOTIATING
        out = self.svc.propose_price(s.car, offer, s.negotiation.get(s.car))
        s.negotiation[s.car] = out.state
        d = out.decision
        dbg.decision = f"{d.kind.value} amount={d.amount} round={d.round_no} rounds_left={d.rounds_left}" + (" final" if d.final else "") + (f" reason={d.reason}" if d.reason else "")
        self._tool(dbg, f"propose_price({s.car}, {offer})", d.kind.value + (f" {d.amount}" if d.amount else ""))
        if d.kind is DecisionKind.CLARIFY_AMOUNT:
            if d.amount:
                s.pending = Pending("amount_check", d.amount)
                return f"Sorry, did you say {T.money(d.amount)}?"
            return "Sorry, I didn't catch the price. What did you have in mind?"
        if d.kind is DecisionKind.ACCEPT:
            s.pending = Pending("offer_booking")
            return f"I can do {T.money(d.amount)}, subject to confirmation. Would you like to book a test drive?"
        if d.kind is DecisionKind.COUNTER:
            tail = "Does that work for you?" if d.final else "Can you work with that?"
            return (f"My final offer is {T.money(d.amount)}, subject to confirmation. {tail}" if d.final
                    else f"The best I can do right now is {T.money(d.amount)}, subject to confirmation. {tail}")
        if out.handoff:
            s.handoff, s.ended, s.state, s.pending = out.handoff, True, State.GOODBYE, None
            self._tool(dbg, f"escalate_to_human({out.handoff.reason})", out.handoff.handoff_id)
        if d.kind is DecisionKind.NEEDS_MANAGER:
            return f"I'll need my manager's approval for {T.money(d.amount)}. I'm connecting you with a colleague to confirm."
        return "I'm not able to go that low myself. Let me connect you with a colleague who can discuss it."

    # ------------------------------------------------------------------------------------------------ booking flow
    def _start_booking(self, s: Session, text: str, dbg: TurnDebug) -> str:
        car = self._resolve_car(s, text)
        if car is None:
            return self._ask_which(s)
        status = self.svc.car_status(car.stock_id)
        self._tool(dbg, f"car_status({car.stock_id})", status.value if status else "unknown")
        if status is not CarStatus.AVAILABLE:
            return self._car_gone(s, car, status.value if status else "unavailable", dbg)
        offers: list[datetime] = []
        today = self._now().astimezone(self.svc.slots.tz(car.branch_id)).date()
        for k in range(0, self.svc.cfg.test_drive.horizon_days + 1):
            free = self.svc.check_test_drive_slots(car.stock_id, today + timedelta(days=k)).slots
            if free:
                offers.append(free[0])
            if len(offers) == 2:
                break
        self._tool(dbg, f"check_test_drive_slots({car.stock_id}, next days)", f"{len(offers)} slot(s) offered")
        if not offers:
            return self._handoff(s, dbg, EscalationReason.OUT_OF_SCOPE, "I don't have any free slots in the next two weeks. Let me connect you with a colleague.")
        s.offered_slots, s.state = offers, State.SLOT
        now = self._now()
        options = " or ".join(T.when_short(o, now) for o in offers)
        return f"I can do {options}. Which works better for you?"

    def _pick_offered(self, s: Session, text: str) -> datetime | None:
        """'the first one', 'Thursday', 'the later one' among the slots just offered."""
        if not s.offered_slots:
            return None
        t = " ".join(ex.tokens_of(text))
        if re.search(r"\b(first|1st|earlier|earliest)\b", t):
            return s.offered_slots[0]
        if re.search(r"\b(second|2nd|later|latest|last)\b", t) and len(s.offered_slots) > 1:
            return s.offered_slots[-1]
        w = parse_when(text, self._now(), s.offered_slots[0].tzinfo)
        matches = [o for o in s.offered_slots if (w.day is None or o.date() == w.day) and (w.time is None or o.timetz().replace(tzinfo=None) == w.time)]
        return matches[0] if w.found and len(matches) == 1 else None

    def _give_slot(self, s: Session, text: str, dbg: TurnDebug) -> str:
        car = self._car(s)
        if car is None:
            return self._ask_which(s)
        tz = self.svc.slots.tz(car.branch_id)
        now = self._now()
        chosen = self._pick_offered(s, text)
        w = parse_when(text, now, tz)
        if chosen is None and w.day and w.time:
            chosen = combine(w.day, w.time, tz)
        if chosen is None and w.day and not w.time:
            free = self.svc.check_test_drive_slots(car.stock_id, w.day).slots
            pod = part_of_day(text)
            if pod:
                free = tuple(f for f in free if {"morning": f.hour < 12, "afternoon": 12 <= f.hour < 17, "evening": f.hour >= 17}[pod])
            self._tool(dbg, f"check_test_drive_slots({car.stock_id}, {w.day})", f"{len(free)} free")
            if not free:
                alts = self.svc.slots.alternatives(car, combine(w.day, datetime.min.time(), tz))
                s.offered_slots = list(alts)
                return self._alternatives("We have no free slots then.", alts, now)
            pick = [free[0], free[len(free) // 2]] if len(free) > 1 else [free[0]]
            s.offered_slots = pick
            return f"On {w.day:%A} I can do {' or '.join(T.when_short(p, now).split(' at ')[-1] for p in pick)}. Which would you like?"
        if chosen is None and w.time and not w.day:
            same = [o for o in s.offered_slots if o.timetz().replace(tzinfo=None) == w.time]
            if len(same) == 1:
                chosen = same[0]
            else:
                return "Which day would that be?"
        if chosen is None:
            s.misheard += 1
            return "Sorry, I didn't catch the day and time. For example, Saturday at four thirty?"
        reason = self.svc.slots.invalid_reason(car.branch_id, chosen)
        free_now = self.svc.slots.free_slots(car, chosen.astimezone(tz).date()) if reason is None else []
        self._tool(dbg, f"check_test_drive_slots({car.stock_id}, {chosen.date()})", "free" if chosen in free_now else (reason or "taken"))
        if reason or chosen not in free_now:
            alts = self.svc.slots.alternatives(car, chosen)
            s.offered_slots = list(alts)
            why = {"branch_closed": "We're closed then.", "too_soon": "That's a bit too soon.", "too_far_ahead": "That's too far ahead.",
                   "outside_hours_or_off_grid": "We book on the half hour, during opening hours."}.get(reason or "", "That slot isn't free.")
            return self._alternatives(why, alts, now)
        if s.rescheduling:
            return self._do_reschedule(s, chosen, dbg)
        s.draft_slot = chosen
        if s.draft_name and s.draft_phone:
            return self._read_back(s, dbg)
        s.state = State.CONTACT
        return f"Great, {T.when_short(chosen, now)}. Can I have your name and mobile number?"

    def _alternatives(self, why: str, alts: list[datetime], now: datetime) -> str:
        if not alts:
            return f"{why} Let me have a colleague find you a time."
        return f"{why} How about {' or '.join(T.when_short(a, now) for a in alts[:2])}?"

    def _give_contact(self, s: Session, text: str, dbg: TurnDebug) -> str:
        c = ex.contact(text)
        if c.heard_digits and not c.phone:
            s.misheard += 1
            return "I didn't catch a valid UAE mobile number. Could you repeat it digit by digit?"
        if c.name:
            s.draft_name = c.name
        if c.phone:
            s.draft_phone = c.phone
        if s.contact_purpose == "send":
            if not s.draft_phone:
                return "What's your mobile number, so I can send them on WhatsApp?"
            return self._deliver_details(s, dbg)
        if not s.draft_name and not s.draft_phone:
            s.misheard += 1
            return "Sorry, I didn't catch that. Could you tell me your name and mobile number?"
        if not s.draft_name:
            return "Thanks. And your name, please?"
        if not s.draft_phone:
            return f"Thanks, {s.draft_name.split()[0]}. And your mobile number?"
        return self._read_back(s, dbg)

    def _read_back(self, s: Session, dbg: TurnDebug) -> str:
        car = self._car(s)
        assert car and s.draft_slot and s.draft_name and s.draft_phone
        status = self.svc.car_status(car.stock_id)                      # the car can be sold DURING the call
        self._tool(dbg, f"car_status({car.stock_id})", status.value if status else "unknown")
        if status is not CarStatus.AVAILABLE:
            return self._car_gone(s, car, status.value if status else "unavailable", dbg)
        s.state = State.CONFIRM
        return (f"Let me confirm: a test drive of the {T.car_name(car)} on {T.when_full(s.draft_slot)} at our {self._branch_name(car)}, "
                f"for {s.draft_name}, phone {speak_phone(s.draft_phone)}. Is that right?")

    def _car_gone(self, s: Session, car: CarView, status: str, dbg: TurnDebug) -> str:
        similar = self.svc.similar_to(car.stock_id)                            # same body type, +/-20% price
        s.state, s.car, s.shown = State.ANSWERING, None, list(similar)
        s.draft_slot = None
        if similar:
            return f"I'm sorry, that car has just been {status}. I do have " + " and ".join(f"a {T.car_name(c)} at {T.money(c.price)}" for c in similar[:2]) + "."
        return f"I'm sorry, that car has just been {status}. Would you like to look at something else?"

    def _finalize(self, s: Session, dbg: TurnDebug) -> str:
        car = self._car(s)
        assert car and s.draft_slot and s.draft_name and s.draft_phone
        out = self.svc.book_test_drive(car.stock_id, s.draft_name, s.draft_phone, s.draft_slot)
        self._tool(dbg, f"book_test_drive({car.stock_id}, {s.draft_slot:%a %H:%M})", out.kind.value)
        now = self._now()
        if out.kind is BookingKind.BOOKED:
            s.booking, s.draft_slot = out.booking, None
            sent = self.svc.send_details("whatsapp", [car.stock_id], s.draft_phone)
            self._tool(dbg, f"send_details(whatsapp, [{car.stock_id}])", f"sent={sent.sent}" if sent.ok else sent.reason)
            s.state = State.CONFIRM
            bits = []
            if self.svc.cfg.test_drive.licence_required:          # TO CONFIRM WITH SALES: licence required, mentioned at booking
                bits.append("please bring a valid driving licence")
            if sent.ok:
                bits.append("I've sent the details to your WhatsApp")
            second = (" " + bits[0][0].upper() + bits[0][1:] + ("; " + bits[1] if len(bits) > 1 else "") + ".") if bits else ""
            return f"You're booked for {T.when_short(out.booking.start, now)}.{second}"
        if out.kind is BookingKind.CAR_UNAVAILABLE:
            return self._car_gone(s, car, out.reason, dbg)
        if out.kind in (BookingKind.SLOT_TAKEN, BookingKind.SLOT_INVALID):
            s.offered_slots, s.state = list(out.alternatives), State.SLOT
            return self._alternatives("That slot has just been taken.", list(out.alternatives), now)
        if out.kind is BookingKind.DUPLICATE and out.existing:
            s.booking, s.pending, s.state = out.existing, Pending("move_existing", out.existing.booking_id), State.CONFIRM
            return f"You already have a test drive for this car {T.when_short(out.existing.start, now)}. Would you like to move it?"
        s.state = State.CONTACT
        return "I couldn't use that phone number. Could you give me your mobile number again?"

    # ------------------------------------------------------------------------------------------------ yes / no / change / cancel
    def _yes(self, s: Session, dbg: TurnDebug) -> str:
        if s.state is State.CONFIRM and s.draft_slot:
            return self._finalize(s, dbg)
        if s.state is State.CONFIRM and s.booking:
            return "Is there anything else I can help with?"
        if s.car:
            s.pending = Pending("offer_booking")
            return "Great. Would you like to book a test drive for it?"
        return "Great. What kind of car are you looking for?"

    def _no(self, s: Session, text: str, dbg: TurnDebug) -> str:
        if s.state is State.CONFIRM and s.draft_slot:
            if re.search(r"\b(number|phone|mobile|name|spell)\b", " ".join(ex.tokens_of(text))):
                s.state = State.CONTACT
                s.draft_name = s.draft_phone = None
                return "Sorry about that. Could you give me your name and mobile number again?"
            s.state = State.SLOT
            return "No problem, let's fix it. Which day and time would you like instead?"
        if s.state is State.CONFIRM and s.booking:
            s.ended = True
            s.state = State.GOODBYE
            return self._say(s, "goodbye")
        if s.state is State.SLOT:
            return "No problem. Which day would suit you better?"
        if s.state is State.CONTACT:
            s.state, s.draft_slot = State.ANSWERING, None
            return "No problem, I won't take a booking. Is there anything else I can help with?"
        return "No problem. What would you like to look at instead?"

    def _reschedule(self, s: Session, text: str, dbg: TurnDebug) -> str:
        car = self._car(s)
        w = parse_when(text, self._now(), self.svc.slots.tz(car.branch_id)) if car else None
        if s.draft_slot is None and s.booking and not s.rescheduling:
            s.rescheduling = s.booking.booking_id
        if w and w.day and w.time and car:
            return self._give_slot(s, text, dbg)
        s.state = State.SLOT
        return "Of course. Which day and time would you like instead?"

    def _do_reschedule(self, s: Session, chosen: datetime, dbg: TurnDebug) -> str:
        out = self.svc.reschedule_booking(s.rescheduling, chosen)
        self._tool(dbg, f"reschedule_booking({s.rescheduling}, {chosen:%a %H:%M})", out.kind.value)
        now = self._now()
        if out.kind is BookingKind.RESCHEDULED:
            s.booking, s.rescheduling, s.state = out.booking, None, State.CONFIRM
            return f"Done, I've moved your test drive to {T.when_short(out.booking.start, now)}. Anything else?"
        if out.kind in (BookingKind.SLOT_TAKEN, BookingKind.SLOT_INVALID):
            return self._alternatives("That slot isn't free.", list(out.alternatives), now)
        s.rescheduling, s.state = None, State.ANSWERING
        return "I couldn't move that booking. Let me connect you with a colleague."

    def _cancel(self, s: Session, dbg: TurnDebug) -> str:
        if s.booking:
            out = self.svc.cancel_booking(s.booking.booking_id)
            self._tool(dbg, f"cancel_booking({s.booking.booking_id})", out.kind.value)
            s.booking, s.rescheduling, s.state = None, None, State.ANSWERING
            return "Your test drive is cancelled. Is there anything else I can help with?"
        s.draft_slot, s.offered_slots, s.state = None, [], State.ANSWERING
        return "No problem, I've dropped that request. Is there anything else I can help with?"

    # ------------------------------------------------------------------------------------------------ details by message
    def _send_details(self, s: Session, text: str, dbg: TurnDebug) -> str:
        car = self._resolve_car(s, text)
        if car is None and not s.shown:
            return self._ask_which(s)
        if s.draft_phone:
            return self._deliver_details(s, dbg)
        s.contact_purpose, s.state = "send", State.CONTACT
        return "Sure. What's your mobile number, so I can send them on WhatsApp?"

    def _deliver_details(self, s: Session, dbg: TurnDebug) -> str:
        ids = [s.car] if s.car else [c.stock_id for c in s.shown]
        res = self.svc.send_details("whatsapp", ids, s.draft_phone or "")
        self._tool(dbg, f"send_details(whatsapp, {ids})", f"sent={res.sent}" if res.ok else res.reason)
        s.contact_purpose, s.state = "booking", State.ANSWERING
        if not res.ok:
            return "I couldn't send that message. Let me have a colleague follow up."
        return "Done, I've sent the details to your WhatsApp. Is there anything else I can help with?"
