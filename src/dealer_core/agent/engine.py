"""Tool-calling agent engine (experimental, `dialogue.engine: agent`).

The state-machine engine decides in code what a caller meant and what happens next; the model only words the reply. Here the MODEL decides the next
action: each turn it picks ONE tool and its arguments (a short JSON object); CODE runs the tool against the real inventory / slots / bookings and
returns facts; the model then speaks the result, sentence by sentence, through the same reply writer and the same safety checks as before.

What stays in code, so the model cannot get it wrong:
  * every fact (cars, prices, slots) comes from the tool result; every spoken sentence is checked against those facts (writer.verify_sentence)
  * dates and times are parsed by code (`parse_when`) from the customer's own words
  * a booking is only made after a READ-BACK and a confirm_booking call; the car's availability and the slot are re-checked at that moment
  * handing over to a person and ending the call are tools the model can pick, executed by code

One extra short model call per turn (the decision), so a turn costs a little more time than in the state machine.
"""
from __future__ import annotations

import dataclasses
import json
import re
import time
from datetime import datetime
from difflib import SequenceMatcher

from ..dialogue import templates as T
from ..dialogue import voice_rules
from ..dialogue.machine import TurnDebug, TurnResult
from ..dialogue.session import Session
from ..dialogue.states import State
from ..dialogue.draft import clean_draft
from ..dialogue.writer import _DONE, split_point, ReplyWriter, WriterInput, fact_sheet, promise_in, verify_sentence
from ..models import BodyType, CarStatus, CarView, SearchFilters
from ..nlu import expected, extractors as ex
from ..nlu.datetime_parse import combine, parse_when, part_of_day
from ..tools.escalation import EscalationReason
from ..tools.phones import normalize_phone, speak_phone
from ..tools.results import BookingKind
from ..tools.service import DealerService

_PLAIN_YES = re.compile(r"(?:yes|yeah|yep|yup|sure|correct|right|perfect|ok|okay)(?: (?:yes|please|that is|that's|it is|is|correct|right|perfect|confirm(?: it)?|go ahead|thanks|thank you))*")
_CHANGE = re.compile(r"\b(?:no|not|change|different|another|other|cancel|wait|actually|instead|but)\b")
_VALUE = re.compile(r"\b(?:good price|fair price|good deal|worth it|worth the|too expensive|expensive|too much|overpriced|pricey)\b")
_ATTRIBUTE = {"km": re.compile(r"\b(?:kilomet\w*|km|mileage|miles|done|driven)\b"), "colour": re.compile(r"\b(?:colou?r)\b"),
              "price": re.compile(r"\b(?:price|cost|how much)\b"), "year": re.compile(r"\b(?:year|how old|model year)\b")}
_ABOUT_THIS_CAR = re.compile(r"\b(?:kilomet\w*|km|mileage|miles|colou?r|price|cost|how much|year|how old)\b")   # only what the code's line answers; engine, warranty, ... go to the model
_BOOKING_CUE = re.compile(r"\b(?:book|booking|test drive|drive it|see it|come|visit|appointment|today|tomorrow|morning|afternoon|evening|tonight)\b")
HAGGLE = re.compile(r"\b(?:best price|lowest|discount|cheaper|lower|negotiat\w*|bargain|reduce|knock|deal|last price|final price|do (?:it )?for|can you do)\b")

CONTINUE_SYSTEM = """You are Wazir, the friendly phone assistant of AGMC, a used-car dealership. Your reply has already begun (the start is given to you); you only add the closing question,
in one short natural sentence. Never repeat facts, never mention prices, numbers, discounts, warranties, policies or promises, and never say anything is booked or confirmed."""
CURRENCY = re.compile(r"\b(?:dirhams?|aed|dollars?|price|prices|cost|costs|pay|payment)\b", re.I)
POLICY = re.compile(r"\b(?:fixed|negotiat\w*|discount\w*|warrant\w*|guarant\w*|financ\w*|insur\w*|deliver\w*|return\w*|refund\w*|deposit|free|fees?|trade-?in|lowest|cheapest|best|special|promo\w*|offers?|deals?)\b", re.I)

EXACT_TOOLS = ("prepare_booking", "confirm_booking", "check_slots", "cancel_draft", "handoff", "end_call")
TOOLS = ("search_cars", "get_car", "check_slots", "prepare_booking", "confirm_booking", "cancel_draft", "handoff", "end_call", "none")

DECIDE_SYSTEM = """You are the brain of Wazir, the phone assistant of AGMC, a used-car dealership. Read the conversation and choose the ONE next action for the
customer's last message. Answer with ONLY the tool name, and for get_car, check_slots and prepare_booking the stock id of the car after it (like: get_car AG-101).

search_cars      the customer wants to find cars or narrows a search (make, model, type, colour, budget, year, kilometres). "any budget" / "doesn't matter" / "show me" too.
get_car          a question about one specific car: price, kilometres, colour, gearbox, fuel, anything.
check_slots      the customer asks which test-drive times are free.
prepare_booking  the customer wants a test drive, or gives a booking detail (day, time, name or phone number).
confirm_booking  the customer says yes to the read-back of the booking.
cancel_draft     the customer says no or wants to change the booking details.
handoff          the customer asks for a person, or is upset.
end_call         the customer says goodbye or is finished.
none             greeting, thanks, small talk, anything that needs no lookup.
                 A question or doubt about what is in stock ("do you have more?", "you don't have anymore?", "you said four") is search_cars, not none.

The customer's words come from speech recognition and can be misspelled. The stock id must come from CODE HINT, CAR IN FOCUS or LAST SHOWN ("it", "this one",
"the second one" mean a car from there). Prices are fixed: nothing to negotiate.
"""


def _clauses(text: str) -> list[str]:
    """The prefix as speakable pieces (a sentence, or a clause of 6+ words up to a comma), so speech starts on the first one."""
    out, rest = [], text
    while rest:
        cut = split_point(rest)
        if not cut:
            out.append(rest.strip())
            break
        out.append(rest[:cut[0]].strip())
        rest = rest[cut[1]:]
    return [x for x in out if x]


def _inventory_lines(svc: DealerService) -> str:
    from ..models import SearchFilters as _F

    cars = svc.inventory.search(_F(), 200)
    return "\n".join(f"{c.stock_id} {c.year} {c.make} {c.model} ({c.color} {c.body_type.value})" for c in cars)


def _json_from(text: str) -> dict:
    """The first {...} object in the model's text; {} if there is none or it does not parse."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        data = json.loads(text[start:end + 1])
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


class AgentEngine:
    """Same interface as DialogueEngine for the voice pipeline: new_session / opening / handle_stream."""

    is_agent = True

    def __init__(self, svc: DealerService, backend, reply_writer: ReplyWriter | None, agent_name: str | None = None) -> None:
        self.svc, self.backend, self.reply_writer = svc, backend, reply_writer
        self.model_only = False
        self.silent_fallback = True
        self.prefix_mode = False                         # experimental: code speaks the facts, the model only continues (see _prefix_reply)
        self.prefix_stats: dict = {}
        self.stream_high_risk = True
        self.nlu = None
        self.decide_tokens, self.decide_deadline_s = 16, 5.0
        self._last_text = ""

    # ------------------------------------------------------------------------------------------------ entry points
    def new_session(self) -> Session:
        return Session()

    def opening(self, s: Session) -> str:
        name = self.svc.cfg.app.agent_name
        c = self.svc.get_car(s.car) if s.car else None
        who = f"I'm {name}, and I'm here to help you find your perfect car." if name else "How can I help you find a car?"
        line = f"Hi, welcome to the AGMC dealership! {who} " + (f"I see you're looking at the {T.car_name(c)}. What would you like to know about it?"
                                                              if c is not None else "What are you looking for today?")
        s.remember("agent", line)
        return line

    def handle_stream(self, s: Session, text: str):
        """Yield the spoken sentences, then the TurnResult (same contract as DialogueEngine.handle_stream)."""
        t0 = time.perf_counter()
        dbg = TurnDebug(state_before=s.state.value)
        if s.ended:
            dbg.note = "session already ended"
            yield "Goodbye."
            yield TurnResult("Goodbye.", dbg, True)
            return
        decision = self._fast_decide(s, text) or self._decide(s, text)
        tool = decision.get("tool", "none")
        if tool not in TOOLS:
            tool = "none"
        args = self._args_for(s, tool, decision.get("car_id"), text) if tool in ("search_cars", "get_car", "check_slots", "prepare_booking") else {}
        if tool == "search_cars" and not args and s.shown and not (ex.wants_listing(text) or ex.wants_others(text) or expected.nothing_in_particular(text)):
            tool = "none"                                           # nothing searchable in it ("Bravo!", "throttle"): speech-recognition noise, not a request to list again
        picked = ex.selection(text, s.shown)
        if picked and tool in ("none", "check_slots", "search_cars"):
            tool, args = "get_car", {"car_id": picked}                # "tell me about the second one" is about that car, whatever the model called it
        elif tool == "none" and s.car and HAGGLE.search(" ".join(ex.tokens_of(text))):
            tool, args = "get_car", {"car_id": s.car}                 # "what's your best price" / "any discount": the answer is the fixed price
        dbg.intent, dbg.via = tool, "agent"
        dbg.nlu_ms = (time.perf_counter() - t0) * 1000
        result = self._run(s, tool, args if isinstance(args, dict) else {}, text, dbg)
        dbg.state_after = s.state.value
        spoken: list[str] = []
        prefix, question = self._split_core(result["core"])
        if self.prefix_mode and not s.ended and tool in EXACT_TOOLS:
            for part in _clauses(result["core"]):             # slot-filling questions, the read-back and the confirmation are spoken by code, word for word; the model only chose the tool
                spoken.append(part)
                yield part
            self.prefix_stats = {"continuation": "exact-wording"}
        elif self.prefix_mode and self.reply_writer is not None and not s.ended and prefix:
            dbg.risk = result["risk"]
            for sentence in self._prefix_reply(s, text, result, prefix, question, dbg):
                spoken.append(sentence)
                yield sentence
        elif self.reply_writer is not None and not s.ended:
            facts = "\n".join(result["facts"])
            w = WriterInput(caller=text, core=result["core"], risk=result["risk"], history=list(s.history), facts=facts, awaiting=result["awaiting"])
            dbg.risk = result["risk"]
            temperature = self.reply_writer.temperature
            for attempt in range(2):                                    # a reply the checks rejected outright: ask the model again, more carefully (still the model's words)
                if attempt:
                    self.reply_writer.temperature = 0.2
                    why = self.reply_writer.last_stream.get("stopped", "")
                    w = dataclasses.replace(w, facts=w.facts + f"\nYour previous attempt was rejected because: {why}. Write it again without that.")
                for sentence in self.reply_writer.stream(w):
                    spoken.append(sentence)
                    yield sentence
                info = self.reply_writer.last_stream
                dbg.lead_ms = info.get("first_ms", 0.0)
                if info.get("stopped"):
                    dbg.note = (dbg.note + "; " if dbg.note else "") + f"stream{' (retry)' if attempt else ''}: " + info["stopped"]
                if spoken:
                    break
            self.reply_writer.temperature = temperature
            if spoken:
                dbg.draft = " ".join(spoken)
                dbg.copy_ratio = round(SequenceMatcher(None, dbg.draft.lower(), result["core"].lower()).ratio(), 2)
        if not spoken and (not self.model_only or self.silent_fallback):
            spoken = [result["core"]]
            dbg.note = (dbg.note + "; " if dbg.note else "") + "FALLBACK: the model gave nothing usable, so the code-written wording was spoken"
            yield result["core"]
        elif not spoken:
            dbg.note = (dbg.note + "; " if dbg.note else "") + "model_only: the model gave no usable reply, so nothing was said"
        reply = " ".join(spoken)
        s.remember("customer", text)
        s.remember("agent", reply)
        dbg.about = s.focus
        dbg.violations = voice_rules.check(reply, max_sentences=voice_rules.MAX_SENTENCES + 2, max_words=90)
        dbg.total_ms = (time.perf_counter() - t0) * 1000
        yield TurnResult(reply, dbg, s.ended)

    def handle(self, s: Session, text: str) -> TurnResult:
        out = list(self.handle_stream(s, text))
        return out[-1]

    # ------------------------------------------------------------------------------------------------ prefix mode (experimental)
    @staticmethod
    def _split_core(core: str) -> tuple[str, str]:
        """The reference wording as (the sentences that state facts, the closing question the system is waiting on)."""
        sents = voice_rules.sentences(core)
        question = sents[-1] if sents and sents[-1].rstrip().endswith("?") else ""
        facts = [x for x in (sents[:-1] if question else sents) if not x.rstrip().endswith("?")]
        return " ".join(facts), question

    def _check_continuation(self, sentence: str, prefix: str, question: str, caller: str, facts: str, history: str) -> list[str]:
        """The follow-up may only ask the next question: no figures, no policy, no promise, nothing already said, nothing the code did not supply."""
        problems = list(verify_sentence(sentence, caller, prefix + " " + question, facts, history))
        if re.search(r"\d", sentence):
            problems.append("states a figure (prices, times and numbers come from the code)")
        if CURRENCY.search(sentence):
            problems.append("talks about money")
        if POLICY.search(sentence) and not POLICY.search(prefix + " " + question):
            problems.append("states a policy or commercial term the code did not supply")
        if promise_in(sentence) and not promise_in(prefix + " " + question):
            problems.append("makes a promise the code did not supply")
        if _DONE.search(sentence) and not _DONE.search(prefix):
            problems.append("says something is booked or confirmed")
        said = {w for w in re.findall(r"[a-z0-9']+", prefix.lower()) if len(w) > 3}
        mine = {w for w in re.findall(r"[a-z0-9']+", sentence.lower()) if len(w) > 3}
        if mine and len(mine & said) / len(mine) >= 0.6:
            problems.append("repeats what was just said")
        if len(sentence.split()) > 18:
            problems.append("too long for a follow-up")
        if question and not sentence.rstrip().endswith("?"):
            problems.append("does not ask the question the call is waiting on")
        return problems

    def _prefix_reply(self, s: Session, text: str, result: dict, prefix: str, question: str, dbg: TurnDebug):
        """Speak the code-written facts at once; then let the model continue the SAME assistant turn with one short, checked follow-up."""
        t0 = time.perf_counter()
        stats = {"prefix_chars": len(prefix), "continuation": "none-needed"}
        self.prefix_stats = stats
        for part in _clauses(prefix):
            yield part
        if not question:
            return
        stats["continuation"] = "rejected"
        history = " ".join(t for _, t in s.history)
        facts = "\n".join(result["facts"])
        convo = "\n".join(f"{'AGENT' if who == 'agent' else 'CUSTOMER'}: {t}" for who, t in s.history[-4:]) or "(start of the call)"
        user = (f"CONVERSATION SO FAR:\n{convo}\n\nCUSTOMER (just now): {text}\n\nYOUR REPLY HAS ALREADY STARTED, you have just said: \"{prefix}\"\n"
                f"The call now needs this answered: \"{question}\"\nContinue your reply with ONE short sentence (at most 14 words) that asks it in your own words. "
                "Do not repeat anything you already said, and mention no prices, numbers, policies or promises.")
        reasons = ""
        for attempt in range(2):
            u = user + (f"\nYour previous attempt was rejected because: {reasons}. Write it again without that." if reasons else "")
            buf = ""
            try:
                for piece in self.reply_writer.backend.lead_in_stream(CONTINUE_SYSTEM, u, 36, 0.2 if attempt else self.reply_writer.temperature, 4.0, prefill=prefix + " "):
                    buf += piece
                    m = re.search(r"[.!?](?:\s|$)", buf)
                    if m:
                        break
            except Exception as e:
                reasons = f"backend {type(e).__name__}"
                continue
            sentence = clean_draft(buf[:m.end()] if m else buf)
            sentence = re.sub(r"^(?:ok(?:ay)?|got it|sure|great|perfect|alright)[,!.]?\s+", "", sentence, flags=re.I).strip()
            if not sentence or not (m or sentence.endswith(("?", ".", "!"))):
                reasons = "no finished sentence"
                continue
            sentence = sentence[:1].upper() + sentence[1:]
            problems = self._check_continuation(sentence, prefix, question, text, facts, history)
            if not problems:
                stats["continuation"] = "accepted" if attempt == 0 else "accepted-on-retry"
                stats["continuation_ms"] = round((time.perf_counter() - t0) * 1000)
                yield sentence
                return
            reasons = "; ".join(problems)
        stats["reason"] = reasons
        dbg.note = (dbg.note + "; " if dbg.note else "") + f"prefix mode: follow-up rejected ({reasons})"

    # ------------------------------------------------------------------------------------------------ the decision
    def _context(self, s: Session) -> str:
        tz = self.svc.slots.tz(self.svc.cfg.branches[0].id)
        now = self.svc.clock().astimezone(tz)
        focus = self.svc.get_car(s.car) if s.car else None
        draft = (f"car {s.car or '-'}, name {s.draft_name or '-'}, phone {s.draft_phone or '-'}, "
                 f"when {T.when_full(s.draft_slot) if s.draft_slot else '-'}, read back to the customer: {'yes, waiting for their yes/no' if s.read_back else 'no'}")
        shown = "; ".join(f"{i + 1}) {c.stock_id} {c.year} {c.make} {c.model}" for i, c in enumerate(s.shown)) or "none"
        convo = "\n".join(f"{'AGENT' if who == 'agent' else 'CUSTOMER'}: {t}" for who, t in s.history[-4:]) or "(start of the call)"
        # The customer's words are matched to the stock by code (spelling-tolerant), so the model never has to know the whole inventory.
        found = ex.find_vehicle(ex.tokens_of(self._last_text), self.svc.inventory.vocabulary())
        hint = ""
        if found.get("make"):
            ids = [c.stock_id for c in self.svc.inventory.search(SearchFilters(make=found["make"], model=found.get("model")), 5)]
            hint = f"CODE HINT (spelling corrected): the customer's words match make {found['make']}" + (f", model {found['model']}" if found.get("model") else "") + f", ids {ids}\n"
        return (f"NOW: {now:%A %d %B %Y, %H:%M} (Dubai). Open every day.\n{hint}"
                f"CAR IN FOCUS: {focus.stock_id + ' ' + T.car_name(focus) if focus else 'none'}\nLAST SHOWN: {shown}\n"
                f"SEARCH SO FAR: {s.filters or 'nothing'}\nBOOKING DRAFT: {draft}\nCONVERSATION:\n{convo}\n")

    def _fast_decide(self, s: Session, text: str) -> dict | None:
        """The turns whose action the code can tell for certain, without asking the model (saves its 0.25-0.4 s decision call):
        a pick from the list just read out, the next booking detail while a booking is being filled in, and a plain yes to the read-back.
        Anything else, or anything mixed ("yes, but at five"), goes to the model as before."""
        picked = ex.selection(text, s.shown)
        if picked and s.state not in (State.SLOT, State.CONTACT, State.CONFIRM):
            return {"tool": "get_car", "car_id": picked, "via": "rules"}
        if not s.car or s.ended:
            return None
        words = ex.tokens_of(text)
        joined = " ".join(words)
        if (s.state not in (State.SLOT, State.CONTACT, State.CONFIRM) and (_ABOUT_THIS_CAR.search(joined) or HAGGLE.search(joined) or _VALUE.search(joined))
                and not _BOOKING_CUE.search(joined) and not ex.find_vehicle(words, self.svc.inventory.vocabulary()).get("make")):
            return {"tool": "get_car", "car_id": s.car, "via": "rules"}      # "how many kilometres has it done?" is about the car just discussed
        if s.state is State.CONFIRM and s.read_back and _PLAIN_YES.fullmatch(" ".join(words)):
            return {"tool": "confirm_booking", "via": "rules"}
        if s.state in (State.SLOT, State.CONTACT) and not _CHANGE.search(" ".join(words)):
            tz = self.svc.slots.tz(self.svc.cfg.branches[0].id)
            ct = ex.contact(text)
            if parse_when(text, self.svc.clock(), tz).found or ct.phone or (s.state is State.CONTACT and ct.name):
                return {"tool": "prepare_booking", "car_id": s.car, "via": "rules"}
        return None

    def _decide(self, s: Session, text: str) -> dict:
        """The model names the tool (and the car). Everything else - filters, name, phone, day and time - is taken from the customer's words by code."""
        self._last_text = text
        user = self._context(s) + f"CUSTOMER (now): {text}\nTOOL:"
        try:
            raw = self.backend.lead_in(DECIDE_SYSTEM, user, self.decide_tokens, None, self.decide_deadline_s)
        except Exception as e:                                          # the model did not answer in time: do nothing rather than guess
            return {"tool": "none", "args": {}, "error": type(e).__name__}
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
        data = _json_from(raw)
        tool = data.get("tool") if data else None
        if tool is None:
            m = re.search(r"\b(" + "|".join(TOOLS) + r")\b", raw)
            tool = m.group(1) if m else "none"
        car = re.search(r"\b[A-Z]{2}-\d{2,4}\b", raw)
        return {"tool": tool, "car_id": car.group(0) if car else (data.get("args", {}) or {}).get("car_id") if data else None}

    def _args_for(self, s: Session, tool: str, car_id: str | None, text: str) -> dict:
        """The tool's arguments, read from the customer's own words by code (the same extractors the state machine trusts)."""
        vocab = self.svc.inventory.vocabulary()
        if tool == "search_cars":
            return {k: v for k, v in ex.search_filters(text, vocab).items() if v not in (None, "")}
        args: dict = {"car_id": car_id or ex.selection(text, s.shown) or None}
        if not args["car_id"]:
            found = ex.find_vehicle(ex.tokens_of(text), vocab)
            if found.get("make"):
                hits = self.svc.inventory.search(SearchFilters(make=found["make"], model=found.get("model")), 2)
                if len(hits) == 1:
                    args["car_id"] = hits[0].stock_id
        if tool == "check_slots":
            args["when"] = text
        elif tool == "prepare_booking":
            ct = ex.contact(text)
            cue = re.search(r"\b(?:my name is|this is|i am|i'm|name's|it's)\b", text, re.I)
            when_found = parse_when(text, self.svc.clock(), self.svc.slots.tz(self.svc.cfg.branches[0].id)).found
            args["phone"] = ct.phone
            args["name"] = ct.name if (cue or not when_found) else None       # "tomorrow at four" must not become a name
            args["when"] = text
        return args

    # ------------------------------------------------------------------------------------------------ the tools (code)
    def _branch_name(self, c: CarView) -> str:
        return self.svc.slots.branch(c.branch_id).name

    def _car_arg(self, s: Session, args: dict) -> CarView | None:
        cid = str(args.get("car_id") or "").upper().replace(" ", "")
        car = self.svc.get_car(cid) if cid else None
        return car or (self.svc.get_car(s.car) if s.car else None)

    def _run(self, s: Session, tool: str, args: dict, text: str, dbg: TurnDebug) -> dict:
        fn = getattr(self, f"_t_{tool}")
        out = fn(s, args, text, dbg)
        out.setdefault("risk", "low")
        out.setdefault("awaiting", False)
        context = [f"- {T.car_name(c)}: {c.color}, {c.price:,} dirhams" for c in ([self.svc.get_car(s.car)] if s.car and self.svc.get_car(s.car) else []) + list(s.shown)][:5]
        out["facts"] = [f"What the customer wanted: {tool.replace('_', ' ')}.", *out.get("facts", []),
                        "Prices are fixed: there is no discount, best price or negotiation, and you must not say 'offer' or 'deal'.",
                        *(["Cars in this call (true facts you may use):", *context] if context else [])]
        return out

    def _t_none(self, s, args, text, dbg) -> dict:
        question = "What are you looking for today?" if not s.car and not s.shown else "Is there anything else I can help with?"
        context = []
        cars = ([self.svc.get_car(s.car)] if s.car and self.svc.get_car(s.car) else []) + list(s.shown)
        if cars:
            context = ["Cars the customer has been shown in this call (these ARE in stock):", *fact_sheet(cars[:4], self._branch_name).splitlines()]
        if s.filters:
            context.append(f"What they have asked for so far: {s.filters}.")
        return {"core": question, "facts": ["No lookup was needed. Answer from the conversation and these facts only; never say the dealership has no cars unless a search said so.",
                          "If the customer's words do not make sense here (speech recognition can mishear), say you did not catch that and ask them to say it again.", *context],
                "awaiting": False}

    def _t_search_cars(self, s, args, text, dbg) -> dict:
        vocab = self.svc.inventory.vocabulary()
        f = dict(s.filters)
        make = str(args.get("make") or "").strip().lower()
        model = str(args.get("model") or "").strip().lower()
        if model and not make:
            make = next((mk for mk, ms in vocab.items() if model in ms), "")
        if make and make != f.get("make"):
            f.pop("model", None)
        if make:
            f["make"] = make
        if model:
            f["model"] = model
        for key in ("color", "body_type"):
            if args.get(key):
                f[key] = str(args[key]).strip().lower()
        for key, cast in (("max_price", int), ("year_min", int), ("max_km", int)):
            try:
                if args.get(key) not in (None, "", "null"):
                    f[key] = cast(args[key])
            except (TypeError, ValueError):
                pass
        try:
            filters = SearchFilters(make=f.get("make"), model=f.get("model"), color=f.get("color"), max_price=f.get("max_price"),
                                    year_min=f.get("year_min"), max_km=f.get("max_km"),
                                    body_type=BodyType(f["body_type"]) if f.get("body_type") else None)
        except ValueError:
            f.pop("body_type", None)
            filters = SearchFilters(make=f.get("make"), model=f.get("model"), color=f.get("color"), max_price=f.get("max_price"))
        unchanged = f == s.filters                                   # nothing new said: "any budget" / "doesn't matter" / "show me"
        s.filters = f
        if ex.wants_others(text) and s.shown:                          # "any others?" / "you don't have anymore?": not the ones already read out
            filters = filters.model_copy(update={"exclude_stock_ids": tuple(c.stock_id for c in s.shown)})
        res = self.svc.search_cars(filters)
        dbg.tools.append(f"search_cars({f}) -> {res.total} match(es)")
        s.state = State.ANSWERING if res.cars else State.INTEREST
        s.shown = list(res.cars)
        s.car = res.cars[0].stock_id if res.total == 1 else None
        wanted = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in f.items()) or "anything"
        if not res.cars:
            return {"core": "I don't have anything that matches that. Would you like to try another make or type of car?",
                    "facts": [f"Search for {wanted}: no cars in stock match."], "awaiting": True}
        if len(res.cars) > 1 or res.needs_narrowing:                   # a list is names and prices only: no kilometres etc. for the model to mangle mid-sentence
            sheet = [f"- {T.car_name(c)}: {c.color}, {c.price:,} dirhams" for c in res.cars]
        else:
            sheet = fact_sheet(list(res.cars), self._branch_name).splitlines()
        if res.needs_narrowing and not (unchanged and s.narrow_asks > 0):
            s.narrow_asks += 1
            return {"core": f"I have {res.total} cars like that. Could you tell me your budget or a make?",
                    "facts": [f"Search for {wanted}: exactly {res.total} cars match (say the number {res.total}, never 'a few' or 'several'), too many to read out. Examples:", *sheet,
                              "Ask one narrowing question (budget, make, body type or colour) instead of listing them all."], "awaiting": True}
        more = res.total - len(res.cars)
        if not res.needs_narrowing:
            s.narrow_asks = 0
        items = [f"a {T.car_name(c)} at {T.money(c.price)}" for c in res.cars]
        listed = items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]
        ask = "Would you like to hear more, or book a test drive?" if len(items) == 1 else "Which one would you like to hear more about?"
        return {"core": f"I have {listed}. {ask}", "facts": [f"Search for {wanted}: {res.total} match(es):", *sheet,
                                                              *([f"There are {more} more not listed: say how many in total ({res.total}) and read out only these {len(res.cars)}."] if more > 0 else [])], "awaiting": True}

    def _t_get_car(self, s, args, text, dbg) -> dict:
        car = self._car_arg(s, args)
        if car is None:
            return {"core": "Which car do you mean?", "facts": ["No car could be identified from the request."], "awaiting": True}
        s.car = car.stock_id
        if all(c.stock_id != car.stock_id for c in s.shown):
            s.shown = [car]
        s.state = State.ANSWERING
        dbg.tools.append(f"get_car({car.stock_id}) -> {car.status.value}")
        facts = ["Prices are fixed (no negotiation).", *fact_sheet([car], self._branch_name).splitlines()]
        words = " ".join(ex.tokens_of(text))
        ask = "Would you like to book a test drive?"
        if HAGGLE.search(words) or _VALUE.search(words):                 # "best price?", "can you do 150?", "is that a good price?": the fixed price, said once
            return {"core": f"Our prices are fixed, so the {T.car_name(car)} is {T.money(car.price)}. {ask}", "facts": facts, "awaiting": True}
        just_said = any(who == "agent" and T.car_name(car) in t for who, t in s.history[-2:])
        if just_said:                                                    # the caller heard the full line a moment ago: answer only what was asked
            part = [x for key, x in (("km", f"It has {car.km:,} kilometres."), ("colour", f"It's {car.color}."), ("price", f"It's {T.money(car.price)}."),
                                     ("year", f"It's a {car.year}.")) if _ATTRIBUTE[key].search(words)]
            if part:
                return {"core": " ".join(part) + " " + ask, "facts": facts, "awaiting": True}
        return {"core": f"It's the {T.car_name(car)}, {car.color}, {car.km:,} kilometres, {T.money(car.price)}. {ask}", "facts": facts, "awaiting": True}

    def _slot_text(self, slots, now) -> str:
        return " or ".join(T.when_short(x, now) for x in slots)

    def _t_check_slots(self, s, args, text, dbg) -> dict:
        car = self._car_arg(s, args)
        if car is None:
            return {"core": "Which car would you like to test drive?", "facts": ["No car identified."], "awaiting": True}
        tz, now = self.svc.slots.tz(car.branch_id), self.svc.clock()
        when = str(args.get("when") or text)
        w = parse_when(when, now, tz)
        if not w.day:
            return {"core": "Which day would suit you?", "facts": ["The customer has not said a day yet."], "awaiting": True}
        free = list(self.svc.check_test_drive_slots(car.stock_id, w.day).slots)
        pod = part_of_day(when)
        if pod:
            free = [f for f in free if {"morning": f.hour < 12, "afternoon": 12 <= f.hour < 17, "evening": f.hour >= 17}[pod]]
        dbg.tools.append(f"check_slots({car.stock_id}, {w.day}) -> {len(free)} free")
        if not free:
            alts = list(self.svc.slots.alternatives(car, combine(w.day, datetime.min.time(), tz)))[:2]
            return {"core": f"We have no free slots then. How about {self._slot_text(alts, now)}?", "risk": "high", "awaiting": True,
                    "facts": [f"No free slots on {w.day:%A}.", f"Next free: {self._slot_text(alts, now)}."]}
        pick = [free[0], free[len(free) // 2]] if len(free) > 1 else [free[0]]
        return {"core": f"I can do {self._slot_text(pick, now)}. Which works better for you?", "risk": "high", "awaiting": True,
                "facts": [f"Free slots on {w.day:%A}: {self._slot_text(pick, now)} (and others)."]}

    def _t_prepare_booking(self, s, args, text, dbg) -> dict:
        car = self._car_arg(s, args)
        if car is None:
            w = parse_when(str(args.get("when") or text), self.svc.clock(), self.svc.slots.tz(self.svc.cfg.branches[0].id))
            asked = f" The customer asked for {w.day:%A}" + (f" at {w.time:%H:%M}" if w.time else "") + "." if w.day else ""
            return {"core": "Which car would you like to test drive?", "risk": "high", "awaiting": True,
                    "facts": ["No car has been chosen for the booking yet." + asked, "Ask which of the cars they mean. Nothing is booked yet: do not say it is booked, confirmed or arranged."]}
        s.car = car.stock_id
        name = str(args.get("name") or "").strip()
        if name and name.lower() not in ("null", "none"):
            s.draft_name = name.title() if name.islower() else name
        phone = str(args.get("phone") or "").strip()
        if phone and phone.lower() not in ("null", "none"):
            norm = normalize_phone(phone)
            if norm:
                s.draft_phone = norm
        tz, now = self.svc.slots.tz(car.branch_id), self.svc.clock()
        when = str(args.get("when") or "").strip()
        problems: list[str] = []
        if when and when.lower() not in ("null", "none"):
            w = parse_when(when, now, tz)
            if w.day and w.time:
                chosen = combine(w.day, w.time, tz)
                reason = self.svc.slots.invalid_reason(car.branch_id, chosen)
                free = self.svc.slots.free_slots(car, w.day) if reason is None else []
                if reason or chosen not in free:
                    alts = list(self.svc.slots.alternatives(car, chosen))[:2]
                    s.draft_slot = None
                    why = {"branch_closed": "We're closed then.", "too_soon": "That's a bit too soon.", "too_far_ahead": "That's too far ahead.",
                           "outside_hours_or_off_grid": "We book on the half hour, during opening hours."}.get(reason or "", "That slot isn't free.")
                    return {"core": f"{why} How about {self._slot_text(alts, now)}?", "risk": "high", "awaiting": True,
                            "facts": [f"The requested time is not available: {why}", f"Nearest free: {self._slot_text(alts, now)}."]}
                s.draft_slot = chosen
            elif w.day and not w.time:
                free = list(self.svc.slots.free_slots(car, w.day))
                pick = [free[0], free[len(free) // 2]] if len(free) > 1 else free[:1]
                if pick:
                    return {"core": f"On {w.day:%A} I can do {self._slot_text(pick, now)}. Which would you like?", "risk": "high", "awaiting": True,
                            "facts": [f"The customer gave a day but no time. Free on {w.day:%A}: {self._slot_text(pick, now)} (and others)."]}
            else:
                problems.append("the day and time were not understood")
        missing = [x for x, v in (("day and time", s.draft_slot), ("name", s.draft_name), ("mobile number", s.draft_phone)) if not v]
        s.state = State.SLOT if not s.draft_slot else State.CONTACT
        if missing and not s.draft_slot and not s.draft_name and not s.draft_phone:
            return {"core": f"Of course. Which day and time would suit you for the {T.car_name(car)}?", "risk": "high", "awaiting": True,
                    "facts": [f"The customer wants a test drive of the {T.car_name(car)}. Nothing is booked yet.", *fact_sheet([car], self._branch_name).splitlines(),
                              "Ask which day and time suit them. Do not say anything is booked or confirmed, and do not offer times yourself.",
                              "Do not read the car's details out again; the customer knows the car."]}
        if missing:
            need = " and ".join(missing)
            known = [f"{k}: {v}" for k, v in (("when", T.when_full(s.draft_slot) if s.draft_slot else None), ("name", s.draft_name),
                                              ("phone", speak_phone(s.draft_phone) if s.draft_phone else None)) if v]
            return {"core": f"To book the {T.car_name(car)} I still need your {need}.", "risk": "high", "awaiting": True,
                    "facts": [f"Booking draft for the {T.car_name(car)}. Known: {'; '.join(known) or 'nothing yet'}.", f"Still missing: {need}.",
                              *fact_sheet([car], self._branch_name).splitlines(),
                              *[f"Note: {p}." for p in problems], "Ask for the missing detail(s) only, in plain words (no 'I'll' or 'we'll'). Nothing is booked yet: do not say it is booked, confirmed or arranged.",
                              "Do not read the car's details out again; the customer knows the car."]}
        status = self.svc.car_status(car.stock_id)
        if status is not CarStatus.AVAILABLE:
            similar = self.svc.similar_to(car.stock_id)
            s.read_back = False
            return {"core": f"I'm sorry, that car has just been {status.value if status else 'sold'}.", "risk": "high", "awaiting": True,
                    "facts": [f"The car is no longer available ({status.value if status else 'sold'}).", *fact_sheet(list(similar[:2]), self._branch_name).splitlines()]}
        s.read_back = True
        s.state = State.CONFIRM
        line = (f"Let me confirm: a test drive of the {T.car_name(car)} on {T.when_full(s.draft_slot)} at our {self._branch_name(car)}, "
                f"for {s.draft_name}, phone {speak_phone(s.draft_phone)}. Is that right?")
        return {"core": line, "risk": "high", "awaiting": True,
                "facts": ["READ-BACK: say all of these details back to the customer and ask if they are right:", f"car: {T.car_name(car)}",
                          f"when: {T.when_full(s.draft_slot)}", f"where: {self._branch_name(car)}", f"name: {s.draft_name}", f"phone: {speak_phone(s.draft_phone)}"]}

    def _t_confirm_booking(self, s, args, text, dbg) -> dict:
        car = self.svc.get_car(s.car) if s.car else None
        if not (s.read_back and car and s.draft_slot and s.draft_name and s.draft_phone):
            return {"core": "I don't have a booking to confirm yet. Which car and time would you like?", "facts": ["There is no read-back booking to confirm."], "awaiting": True}
        out = self.svc.book_test_drive(car.stock_id, s.draft_name, s.draft_phone, s.draft_slot)
        dbg.tools.append(f"book_test_drive({car.stock_id}, {s.draft_slot:%a %H:%M}) -> {out.kind.value}")
        now = self.svc.clock()
        s.read_back = False
        if out.kind is BookingKind.BOOKED:
            s.booking, s.state = out.booking, State.CONFIRM
            s.draft_slot = None
            lic = " Please bring a valid driving licence." if self.svc.cfg.test_drive.licence_required else ""
            return {"core": f"You're booked for {T.when_short(out.booking.start, now)}.{lic}", "risk": "high", "awaiting": False,
                    "facts": [f"BOOKED: {T.car_name(car)} on {T.when_full(out.booking.start)} for {s.draft_name}.",
                              "Please bring a valid driving licence." if lic else "The booking is made."]}
        if out.kind in (BookingKind.SLOT_TAKEN, BookingKind.SLOT_INVALID):
            alts = list(out.alternatives)[:2]
            s.draft_slot = None
            return {"core": f"That slot has just been taken. How about {self._slot_text(alts, now)}?", "risk": "high", "awaiting": True,
                    "facts": [f"The slot was taken before the booking. Nearest free: {self._slot_text(alts, now)}."]}
        if out.kind is BookingKind.DUPLICATE and out.existing:
            return {"core": f"You already have a test drive for this car {T.when_short(out.existing.start, now)}.", "risk": "high", "awaiting": False,
                    "facts": [f"The customer already has a test drive for this car on {T.when_full(out.existing.start)}."]}
        s.draft_phone = None
        return {"core": "I couldn't use that phone number. Could you give me your mobile number again?", "risk": "high", "awaiting": True,
                "facts": [f"The booking was not made ({out.kind.value}). Ask for the mobile number again."]}

    def _t_cancel_draft(self, s, args, text, dbg) -> dict:
        s.draft_slot = s.draft_name = s.draft_phone = None
        s.read_back = False
        return {"core": "No problem. What would you like to change?", "facts": ["The booking draft was cleared; nothing is booked."], "awaiting": True}

    def _t_handoff(self, s, args, text, dbg) -> dict:
        h = self.svc.escalate_to_human(EscalationReason.REQUESTED_HUMAN, s.car, None, note="agent: customer asked for a person or was upset")
        s.handoff, s.ended, s.state = h, True, State.GOODBYE
        dbg.tools.append(f"escalate_to_human -> {h.handoff_id}")
        return {"core": "Of course. I'm connecting you with a colleague now.", "facts": ["The call is being handed to a colleague."], "risk": "high"}

    def _t_end_call(self, s, args, text, dbg) -> dict:
        s.ended, s.state = True, State.GOODBYE
        return {"core": "Thank you for calling AGMC. Goodbye!", "facts": ["The customer is finished; say a short goodbye."], "awaiting": False}
