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
import threading
import time
from datetime import datetime
from difflib import SequenceMatcher

from ..dialogue import templates as T
from . import answers as A
from . import guards
from . import wording as W
from ..dialogue import voice_rules
from ..dialogue.machine import TurnDebug, TurnResult
from ..dialogue.session import KnownBooking, Pending, Session
from ..dialogue.states import State
from ..dialogue.draft import clean_draft
from ..dialogue.writer import _DONE, FIRST_SENTENCE_DEADLINE, split_point, ReplyWriter, WriterInput, covers, fact_sheet, is_generic_closer, numbers, promise_in, verify_sentence
from ..models import BodyType, CarStatus, CarView, SearchFilters
from ..nlu import expected, extractors as ex, names
from ..nlu.datetime_parse import combine, parse_when, part_of_day
from ..tools.escalation import EscalationReason
from ..tools.phones import normalize_phone, speak_phone
from ..tools.results import BookingKind
from ..tools.service import DealerService

_PLAIN_YES = re.compile(r"(?:yes|yeah|yep|yup|sure|correct|right|perfect|ok|okay)(?: (?:yes|sure|please|that is|that's|it is|is|correct|right|perfect|confirm(?: it)?|go ahead|thanks|thank you))*")
# A yes to "Shall I book it?" that names the booking (28 Oct, the owner's test call: "yes, please book it" was not a plain yes, the model was too slow, and the code
# answered with the car's facts and another invitation). Used only while a read-back waits for the answer.
_BOOK_YES = re.compile(r"(?:(?:yes|yeah|yep|yup|sure|ok(?:ay)?|perfect|great|right|correct)(?: please)? )*(?:please )?"
                       r"(?:go ahead(?: and (?:book|confirm)(?: it| that)?)?|(?:book|confirm)(?: it| that| me in| the (?:test drive|booking)| a test drive| the test drive)?)"
                       r"(?: please| for me| thanks| thank you)*")
# The other ordinary ways to agree (29 Oct sweep: "that's correct", "sounds good", "absolutely", "lock it in", "sure, why not" were not a yes for the code). Whole utterances only:
# "that's not correct" or "sounds good but at 5" do not match.
_AGREE = re.compile(r"(?:(?:yes|yeah|yep|yup|ok(?:ay)?|sure|great|perfect|lovely|alright)(?: please)? )*"
                    r"(?:that'?s (?:all )?(?:correct|right|fine|perfect|good|great)|(?:all|that'?s all) (?:correct|good|right|fine)|"
                    r"sounds? (?:good|great|perfect|right|fine)|please do|absolutely|definitely|of course|go for it|do it|lock it in|"
                    r"(?:that )?works(?: for me)?|that works(?: for me)?|that'?s it|(?:sure )?why not|let'?s do (?:it|that)|go on then|i'?d (?:like|love) that|yes please)"
                    r"(?: (?:thanks|thank you|please|go ahead|cheers))*")
# "I want to see the car" right after an invitation to come and see it (same call): a visit, not more facts. Alone it can mean the photos (section 12), so only after one.
_SEE_CAR = re.compile(r"\b(?:see|look at|view|check out) (?:the car|this car|that car|it|her|the (?!(?:photos?|pictures?|pics|images?|videos?)\b)\w+)\b(?! (?:photos?|pictures?|pics|images?|videos?)\b)")
_CHANGE = re.compile(r"\b(?:no|not|change|different|another|other|cancel|wait|actually|instead|but)\b")
_VALUE = re.compile(r"\b(?:good price|fair price|good deal|worth it|worth the|too expensive|expensive|too much|overpriced|pricey)\b")
_ATTRIBUTE = {"km": re.compile(r"\b(?:kilomet\w*|km|mileage|miles|done|driven)\b"), "colour": re.compile(r"\b(?:colou?r)\b"),
              "price": re.compile(r"\b(?:price|cost|how much)\b"), "year": re.compile(r"\b(?:year|how old|model year)\b")}
_ABOUT_THIS_CAR = re.compile(r"\b(?:kilomet\w*|km|mileage|miles|colou?r|price|cost|how much|year|how old)\b")   # only what the code's line answers; engine, warranty, ... go to the model
_REPEAT = re.compile(r"\b(?:repeat (?:that|it|the|yourself)|say (?:that|it) again|come again|pardon|what did you say|didn't (?:catch|hear) (?:that|you)|"
                     r"did not (?:catch|hear)|one more time|sorry what)\b")
_NO = re.compile(r"(?:no|nope|don't|do not|keep it|leave it|never mind|not now)\b")
_MY_BOOKINGS = re.compile(r"\b(?:what (?:have i|did i) book\w*|my (?:bookings?|test drives?|appointments?)|do i have (?:a|any) (?:test drives?|bookings?|appointments?)|"
                          r"when is my|what time is my|which car did i book)\b")
_CANCEL = re.compile(r"\b(?:cancel\w*|call (?:it )?off|don't need (?:the|my) (?:test drive|booking)|remove (?:my|the) (?:booking|test drive))\b")
_MOVE = re.compile(r"\b(?:move|reschedule|re schedule|postpone|push (?:it|back)|bring (?:it )?forward|"
                   r"change (?:the|my) (?:time|day|date|booking|test drive|appointment)|different (?:time|day)|another (?:time|day))\b")
_COMPARE = re.compile(r"\b(?:compare|comparison|difference|differences|versus|vs|which (?:one )?is better|which (?:one )?should i|better between|"
                      r"how do they compare|side by side)\b")
_OFFERED_DRIVE = re.compile(r"\b(?:test drive|take a look|have a look|look at it|check it out|come (?:in|and see|by)|see it|in person|take (?:it|her) for a spin|a spin|drive it|"
                            r"schedule|book|set up|visit|viewing|try it)\b")
_THANKS = re.compile(r"(?:(?:ok|okay|great|perfect|alright|lovely|cool|awesome) )?(?:thank you|thanks|thank you so much|thanks a lot|cheers)(?: (?:so much|very much|a lot))?")
_LOOKING = re.compile(r"\b(?:looking for|look for|do you have|have you got|any|show me|what about|how about|interested in|i want|i'd like|i would like)\b")
# "What colours do you have?", "what colors are these cars?", "which colours does it come in": the colour of cars, asked, not a colour named.
_COLOUR_Q = re.compile(r"\b(?:what|which) colou?rs?\b|\bcolou?rs? (?:are|is|do|does|have|come|available|options?)\b|\b(?:available|other) colou?rs\b|"
                       r"\bin (?:what|which) colou?r\b|\bcolou?r (?:options|choices)\b")
_PLURAL_REF = re.compile(r"\b(?:they|these|those|them|both|each|the cars|these cars|those cars|the ones|all of them)\b")
# "any red cars", "other SUVs from any brand", "doesn't have to be a Toyota": the caller widens the search beyond the make asked for earlier.
_ANY_CARS = re.compile(r"\b(?:any|other|all|different|some)\b(?: \w+){0,2} (?:cars|vehicles|makes|brands)\b|\bany (?:make|brand)\b|"
                       r"\b(?:doesn't|does not|don't|do not) (?:have to|need to) be\b|\bnot (?:just|only) (?:a |an )?\w+s?\b")
# "which ones?", "what are they?", "name them" right after a count: read the list out.
_WHICH_ONES = re.compile(r"(?:so |and |ok(?:ay)? |yeah )?(?:which ones?(?: of (?:those|them|these))?(?: (?:are|is) (?:they|there|those|available|it))?|"
                         r"which (?:are|ones are) (?:they|those|there)|what are (?:they|those|them|the cars)|(?:name|list|read) them(?: out)?(?: for me)?|"
                         r"tell me (?:which ones|them|what they are)|which cars(?: are they)?)(?: then)?")
_STOCK_REF = re.compile(r"\b(?:do you have|have you got|you got|in stock|different|other|all your|your cars)\b")
_SINGULAR_REF = re.compile(r"\b(?:it|its|it's|this one|that one|this car|that car)\b")
# A turn cut off mid-phrase (the end-of-turn detector fired during a pause, or the recognizer lost the end): ends on a word that cannot end a
# request, or the recognizer's usual mishearing of "I'm looking for a..." as "I'm looking forward".
_UNFINISHED_END = {"a", "an", "the", "for", "to", "of", "with", "and", "or", "some", "my", "your", "about", "than", "under", "below"}
_LOOKING_FORWARD = re.compile(r"(?:so )?(?:i'?m|i am|we'?re|we are) (?:just )?looking forward(?: to)?")
_FILLER = {"the", "a", "an", "and", "so", "uh", "um", "umm", "hmm", "er", "erm", "ah", "oh", "to", "of", "in", "is", "it", "but", "or", "well"}
_BOOKING_CUE = re.compile(r"\b(?:book|booking|test drive|drive it|see it|come|visit|appointment|today|tomorrow|morning|afternoon|evening|tonight|"
                          r"weekend|this week|next week|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b")
# "come" that is not a visit: "come on, can you do 150?", "does it come with a warranty?", "does it come in white?", "it comes with a sunroof" (30 Oct, a test call:
# "Does it come with a warranty?" was taken as "can I come in" and answered "What day would you like to come in?")
_NOT_A_VISIT = re.compile(r"\bcome on\b|\bcomes? (?:with|equipped|standard|fitted)\b|\b(?:does|do|did|will|would) (?:it|they|this(?: one| car)?|that(?: one| car)?) come\b")


def _visit_cue(text: str):
    """A booking cue in the caller's words, once the ways of saying "come" that are not about visiting are taken out."""
    return _BOOKING_CUE.search(_NOT_A_VISIT.sub(" ", text))
# "any day", "whenever", "as soon as possible", "what's free", "when can I come": no day chosen, the caller wants to be offered one (27 Oct: six identical
# "which day?" questions in a row, because none of these meant anything to the booking logic)
_FLEXIBLE = re.compile(
    r"\b(?:any\s+(?:day|days|time|times|date)|every\s+day|when\s*ever|whatever\s+(?:day|time|works|suits|is\s+(?:free|best))|"
    r"as\s+soon\s+as\s+(?:possible|you\s+can|i\s+can)|asap|earliest|soonest|first\s+(?:available|free)|next\s+(?:available|free)|"
    r"you\s+(?:choose|pick|decide)|up\s+to\s+you|your\s+choice|no\s+preference|(?:does\s*not|doesn'?t)\s+matter|don'?t\s+mind|"
    r"what(?:'s|\s+is)\s+(?:free|available|open)|(?:what|which)\s+(?:day|days|time|times|slots?)(?:\s+and\s+(?:day|days|time|times))?\s+(?:is|are)\s+(?:free|available|open)|"
    r"when\s+(?:are\s+you|is\s+it)\s+(?:free|open|available)|when\s+can\s+i\s+(?:come|visit|see|drive|try|stop))\b")
# the caller leaves the choice to us (as opposed to asking what is free): with times on the table, the earlier one is taken
_DELEGATES = re.compile(r"\b(?:any\s+(?:day|days|time|times|date)|every\s+day|when\s*ever|whatever|you\s+(?:choose|pick|decide)|up\s+to\s+you|your\s+choice|"
                        r"no\s+preference|(?:does\s*not|doesn'?t)\s+matter|don'?t\s+mind|"
                        r"either(?:\s+(?:one|of\s+them|is\s+fine|works|will\s+do))?|whichever|both\s+(?:are\s+fine|work))\b")    # 29 Oct sweep: "either is fine" asked the day again
# a bare request for more, right after an answer about a car
_MORE_ABOUT = re.compile(r"(?:ok(?:ay)? |yes |yeah |and )?(?:tell me )?(?:something|anything|what) else(?: please| about it)?|(?:tell me )?more(?: please| about it)?|go on|keep going|and what else")
_CHEAPER = re.compile(r"\b(?:something|anything|any|options?|cars?|models?)\b.*\b(?:cheaper|less expensive|lower price)\b|"
                      r"\b(?:cheaper|less expensive) (?:ones?|options?|cars?|models?)\b|^(?:a )?cheaper(?: one)?$")
_OTHER_RANGE = re.compile(r"\b(?:what|which|any)\s+(?:other|else)\b.*\b(?:cars?|models?|makes?|brands?|have|got|stock|sell)\b|"
                          r"\bother (?:makes|brands|models|cars)\b|^what else\b|\bwhat else (?:is there|have you got|do you have)\b")
_REST = re.compile(r"\b(?:other makes|other brands|the others|the rest|remaining|which (?:other )?ones|what are (?:they|those|the others)|name them|"
                   r"the \d+ other|the (?:one|two|three|four|five|six|seven|eight|nine|ten) other|what else)\b")
_BYE = re.compile(r"\b(?:bye|goodbye|good bye|that's all|that is all|that's everything|i'm done|i am done|see you|take care|have a good day)\b")
_WHAT_STOCK = re.compile(r"\b(?:i need a car|i'm looking for a car|looking for a car|i want (?:a|to buy a) car|buy a car|what cars do you have|what (?:cars|vehicles|models) have you got|what do you have|what have you got|what's in stock|"
                         r"what is in stock|what do you sell|show me (?:what you have|your cars|everything)|what are my options)\b")
# "what do you have?" while a search is under way (after "Any specific model?") asks for what matches THAT search, not for the whole showroom
_ASKS_TO_SEE = re.compile(r"\b(?:what do you have|what have you got|what are my options|show me what you have|show me everything|everything you have|"
                          r"show me (?:them )?all|all of them|whatever you(?:'ve)? (?:have|got)|list them)\b")    # 29 Oct sweep: "show me everything" after "4 Toyotas" listed other makes
_FUEL = re.compile(r"\b(electric|ev|hybrid|petrol|diesel|plug in hybrid)\b")
_LOW_KM = re.compile(r"\b(?:low (?:mileage|kilomet\w*|km)|not (?:many|a lot of) (?:kilomet\w*|km|miles)|barely (?:used|driven))\b")
_FINANCE = re.compile(r"\b(?:finance|financing|loan|monthly|instal+ments?|installment plan|lease|leasing|down payment|emi)\b")
_NONE_CLAIM = re.compile(r"\b(?:don't|do not|doesn't|no longer|not) (?:have|carry|stock|sell)\b|\bno \w+ (?:in stock|available)\b|"
                         r"\bout of stock\b|\bnone (?:in stock|available|left)\b|\bsold out\b")
_HAS_OPENER = re.compile(r"(?:sure|ok(?:ay)?|right|alright|great|perfect|lovely|of course|no problem|thanks|thank you|done|all set|yes|"
                         r"no|sorry|just to check|hmm|so|well|absolutely|good|let's see|let me|ah|oh)\b", re.I)
_NEGATIVE = re.compile(r"\b(?:don't|doesn't|isn't|aren't|not|no|none|can't|haven't|hasn't|sorry|unfortunately|fixed)\b", re.I)
# Said the moment a turn needs the model's decision, while it is made (pre-rendered by the voice: voice/pipeline.py common_phrases).
# Only "I am looking it up" wording: it fits every answer that can follow ("Let me check. I don't have any red BMWs right now.").
CUT_OFF = ["Sorry, I missed the end of that. What are you looking for?", "Sorry, you cut out at the end there. What would you like?",
           "Sorry, I only caught the start of that. Could you say it again?"]      # a turn cut off mid-phrase (pre-rendered by the voice too)
BRIDGES = ["Let me check.", "Let me see.", "Let me have a look.", "One moment, let me look."]
STILL_THERE = ["Sorry, are you still there?", "Hello? Are you still there?"]          # a goodbye heard too quietly to believe (voice.end_call_min_rms): ask, do not hang up
_BRIDGE_CUE = re.compile(r"\b(?:do you have|have you got|got any|any|show me|looking for|look for|how much|price|cost|cheap\w*|expensive|budget|under|below|"
                         r"what|which|is there|are there|available|in stock|stock|compare|difference|better|km|kilomet\w*|mileage|colou?r|year|"
                         r"suv|sedan|hatchback|coupe|pickup|van|electric|hybrid|petrol|diesel|automatic|manual|seats?|when|free|open|times?|slots?)\b")
_REACTION_LEAD = re.compile(r"^(?:sure|ok(?:ay)?|right|alright|great|perfect|lovely|of course|absolutely|no problem)[,!.]\s+(?=\S)", re.I)
# Words that keep their capital after "Sure," whatever comes before them. Makes, models, showroom and caller names are added per call (_proper).
_KEEP_CAPS = {"i", "i'm", "i've", "i'll", "i'd", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "january", "february",
              "march", "april", "may", "june", "july", "august", "september", "october", "november", "december", "dubai", "sharjah", "abu", "dhabi",
              "sheikh", "zayed", "al", "uae", "wazir"}


def _lower_first(text: str, proper: frozenset[str] | set[str] = frozenset()) -> str:
    """'The Camry is white' -> 'the Camry is white' after an opener ("Sure, the Camry ..."). A name keeps its capital: 'I', a make or model,
    a day, a place, an acronym (AGMC, SUV), anything in `proper`."""
    first = text.split(" ", 1)[0]
    bare = first.strip(",.!?;:")
    word = bare.lower()
    if not bare[:1].isupper() or sum(ch.isupper() for ch in bare) >= 2 or any(ch.isdigit() for ch in bare) or word in _KEEP_CAPS or word in proper or word.split("'")[0] in proper:
        return text
    return text[0].lower() + text[1:]


PRICE_RULE = "Prices are fixed: there is no discount, best price or negotiation, and you must not say 'offer' or 'deal'."


def _rounding_off(said: str) -> str:
    """A short sentence before the facts of a reply the caller has heard cut in the middle of a word: it starts over politely. (After a comma, see `_unheard`.)"""
    return "Let me put that properly."


def _unheard(heard: str, facts: str) -> list[str]:
    """What the standard wording's fact sentences say that the caller has not heard yet, clause by clause, to carry on a reply cut after a comma. A clause already
    said ("It's the 2016 Porsche Macan in black") is left out; the rest of its sentence is kept ("with 91,000 kilometres on it, at 149,000 dirhams.")."""
    out = []
    for sentence in voice_rules.sentences(facts):
        if covers(heard, sentence):
            continue
        said = numbers(heard)                                    # a clause whose figures were all heard is said ("29,000 km" covers "with 29,000 kilometres on it")
        keep = [c for c in re.split(r",\s+", sentence.rstrip(" .!")) if not covers(heard, c) and not (numbers(c) and numbers(c) <= said)]
        if keep:
            out.append(", ".join(keep) + ".")
    return out


def _after_opener(sentence: str, proper: frozenset[str] | set[str] = frozenset(), new_sentence: bool = False) -> str:
    """The model's first sentence when an opener was already said: its own 'Sure,' / 'Great!' dropped. After a lead that ends its own sentence
    ("Let me check.") the rest starts with a capital; after one that ends on a comma ("Sure,") it carries on in lower case, names excepted."""
    rest = re.sub(r"^(?:sure(?: thing)?|ok(?:ay)?|right|alright|great|perfect|of course|absolutely|certainly|let's see)[,!.]?\s+", "", sentence.strip(), flags=re.I)
    if new_sentence:
        return rest[:1].upper() + rest[1:]
    return _lower_first(rest, proper)


_FACT_TOOLS = ("search_cars", "browse_makes", "compare_cars", "best_of", "my_bookings", "showroom_info", "colours")   # replies that are lists or figures
_HOUR_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
_GREETING = re.compile(r"(?:hi|hello|hey|hiya|good (?:morning|afternoon|evening)|salaam|salam|assalamu alaikum|marhaba)(?: there| wazir| everyone| again)?")
# Thinking out loud is not giving up (29 Oct sweep: "hmm let me think" while times were on offer cancelled the booking draft).
_THINKING = re.compile(r"\b(?:let me think|let me see|one sec(?:ond)?|just a (?:sec(?:ond)?|moment|minute)|give me a (?:sec(?:ond)?|minute|moment)|hold on|hang on|"
                       r"wait a (?:sec(?:ond)?|minute|moment)|hmm+|umm+|uhh+)\b")
_NEVER_MIND = re.compile(r"\b(?:never mind|nevermind|maybe later|not now|another time|i'll think about it|i will think about it|let me think|"
                         r"no thanks|no thank you|not right now|i'm just looking|just looking|just browsing)\b")
_FAMILY = re.compile(r"\b(?:family|families|kids|children|child seats?|school run)\b")
_OPINION = re.compile(r"\b(?:reliable|reliability|good (?:family )?car for (?:a )?famil\w+|good for (?:a )?famil\w+|(?:a )?good family car|family car|any good(?! for)|good car(?! for)|good choice|worth it|worth buying|would you recommend|do you recommend|good buy|dependable)\b")
_SPORTY = re.compile(r"\b(?:sporty|sports car|sport car|fast|quick|performance|powerful)\b")
_SEATS = re.compile(r"\b(seven|eight|7|8)[- ]?seat(?:s|er)?\b|\bthird row\b|\bbig family\b")
HAGGLE = re.compile(r"\b(?:best price|lowest|discount|cheaper|lower|negotia\w*|bargain|haggle|flexible on (?:the )?price|room on (?:the )?price|reduce|knock|deal|last price|final price|do (?:it )?for|can you do)\b")
# 30 Oct, the owner's call: "Can it be 70,000 dirhams instead?", "I can only do 60,000 dirhams.", "Can we do 50,000 dirhams for the Civic?" were read as new budgets
# (or as asking about the Civic), so the fixed price was never said. Words like these count as a push only with an amount below the car's price (offers_less).
_OFFER = re.compile(r"\b(?:(?:can|could|would|will)(?: we| you| it| that| i)? (?:do|be|go|make it|take|accept|let it go|sell it)|i can (?:only )?(?:do|pay|afford|go|stretch)|"
                    r"i could (?:only )?(?:do|pay|afford)|make it|instead|how about|what about)\b")


def offers_less(words: str, price: int) -> bool:
    """'can it be 70,000 instead?', 'I can only do 60,000': an amount below the car's price, offered for it."""
    asked = ex.asked_price(words, price)
    return bool(asked and asked < price and _OFFER.search(words))

CONTINUE_SYSTEM = """You are Wazir, the friendly phone assistant of AGMC, a used-car dealership. Your reply has already begun (the start is given to you); you only add the closing question,
in one short natural sentence. Ask EXACTLY the question you are given, only in other words: do not bring up any topic, feature, car or subject that is not in it, and do not ask a different question.
Never repeat facts, never mention prices, numbers, discounts, warranties, policies or promises, and never say anything is booked or confirmed."""
# Words a closing question may use whatever the topic: grammar, and the ordinary ways of asking "anything more?" or "would you like to come?". Any other word must already be in the
# question the code asked, in what was just said, in the caller's words, in the car's facts or in the last lines of the call: a sentence about "driving experience" or "the next
# available model" that the code never asked is a topic the model made up (27 Oct, five of nine closing questions drifted like that).
_GRAMMAR = frozenset("""the and but for with from that this these those there here have has had not yes how what which who when where why also just too very really quite about than into out
over again still its you your yours our ours they them their him her his she can could would should shall will may might must are was were been being does did doing get got let lets
any some one ones another other next either both each every all few many much such only own same then than once since while until before after because though although whether""".split())
_ASKING = frozenset("""like likes fancy want wants wish interested interest keen love happy ready sure maybe perhaps great good nice lovely else more anything something everything tell know hear
learn ask see seeing look looking visit visiting come coming check drive driving test book booking arrange set schedule time soon today whenever convenient meanwhile need help helps find
explore options option choose pick decide start continue ahead take try talk discuss show shown leave stop thanks please okay hello again sorry mind thought think say anyone anybody
     person firsthand showroom viewing appointment spin wheel yourself""".split())
CURRENCY = re.compile(r"\b(?:dirhams?|aed|dollars?|price|prices|cost|costs|pay|payment)\b", re.I)
POLICY = re.compile(r"\b(?:fixed|negotiat\w*|discount\w*|warrant\w*|guarant\w*|financ\w*|insur\w*|deliver\w*|return\w*|refund\w*|deposit|free|fees?|trade-?in|lowest|cheapest|best|special|promo\w*|offers?|deals?)\b", re.I)

# A follow-up in the caller's voice (30 Oct test call: "Tell me more about the Civic?" after "I've got the Civic at 54,500 dirhams"): an order or a want
# addressed to the agent, or the caller's own questions. The agent asks the caller ("Would you like to ...?", "Shall I ...?").
_CALLER_VOICE = re.compile(r"\s*(?:tell me|show me|give me|send me|i want|i'd like|i would like|i'm (?:looking|interested)|do you have|have you got|what about|how about|"
                           r"can (?:i|we) (?:see|get|have|book|hear|know|look|try|test|come|drive)|could (?:i|we) (?:see|get|have|book|hear|know|look|try|test|come|drive))\b", re.I)

EXACT_TOOLS = ("prepare_booking", "confirm_booking", "check_slots", "cancel_draft", "handoff", "end_call", "booking_aside")
TOOLS = ("search_cars", "browse_makes", "colours", "fix_booking", "which_car", "get_car", "best_of", "compare_cars", "check_slots", "prepare_booking", "confirm_booking", "cancel_draft", "my_bookings", "cancel_booking",
         "reschedule_booking", "showroom_info", "repeat", "handoff", "end_call", "none")

DECIDE_SYSTEM = """You are the brain of Wazir, the phone assistant of AGMC, a used-car dealership. Read the conversation and choose the ONE next action for the
customer's last message. Answer with ONLY the tool name, and for get_car, check_slots and prepare_booking the stock id of the car after it (like: get_car AG-101).

search_cars      the customer wants to find cars or narrows a search (make, model, type, colour, budget, year, kilometres). "any budget" / "doesn't matter" / "show me" too.
get_car          a question about one specific car: price, kilometres, colour, gearbox, fuel, anything.
check_slots      the customer asks which test-drive times are free.
prepare_booking  the customer wants a test drive, or gives a booking detail (day, time, name or phone number).
confirm_booking  the customer says yes to the read-back of the booking.
cancel_draft     the customer says no or wants to change the booking details while a booking is being read back.
compare_cars     the customer wants two cars compared ("which is better", "the difference between").
colours          the customer asks what colours the cars come in ("what colours do you have?", "what colours are they?").
my_bookings      the customer asks what they have booked.
cancel_booking   the customer wants to cancel a test drive they already booked.
reschedule_booking the customer wants to move a test drive they already booked to another day or time.
showroom_info    opening hours, where the showroom or branches are, phone / WhatsApp / email.
repeat           the customer asks you to repeat or did not hear what you said.
handoff          the customer asks for a person, or is upset.
end_call         the customer says goodbye or is finished.
none             greeting, thanks, small talk, anything that needs no lookup.
                 A question or doubt about what is in stock ("do you have more?", "you don't have anymore?", "you said four") is search_cars, not none.

The customer's words come from speech recognition and can be misspelled. The stock id must come from CODE HINT, CAR IN FOCUS or LAST SHOWN ("it", "this one",
"the second one" mean a car from there). Prices are fixed: nothing to negotiate.
"""


def plural_make(name: str) -> str:
    return name if name.endswith("s") else name + "s"


@__import__("contextlib").contextmanager
def _nothing():
    yield


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
    guards_quiet_turns = True        # handle_stream(uncertain=True): a goodbye heard at a very low level asks instead of ending the call

    def __init__(self, svc: DealerService, backend, reply_writer: ReplyWriter | None, agent_name: str | None = None) -> None:
        self.svc, self.backend, self.reply_writer = svc, backend, reply_writer
        self.model_only = False
        self.silent_fallback = True
        self.prefix_mode = False                         # experimental: code speaks the facts, the model only continues (see _prefix_reply)
        self.prefetch_on = False                         # dialogue.prefetch (see prefetch); app.py sets it from config
        self.prefetched = 0
        self._prefetch_turn: dict[int, int] = {}
        from concurrent.futures import ThreadPoolExecutor

        self._prefetcher = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefetch")
        self.openers = False                             # start fact replies with a short spoken acknowledgement ("Sure,") that plays at once (voice/kokoro_tts.py)
        self.model_speaks = False                        # ARCH_PLAN Phase 3 (config dialogue.model_speaks): at the full level the model writes the whole reply; see _speaking_input
        self.fidelity = {"sentences": 0, "leaks": 0, "replies": 0, "fell_back": 0}     # model_speaks: sentences checked by agent/guards.py and how many it stopped (for replays)
        self.governor = None                             # QualityGovernor: picks full / prefix / code per turn from the model's live speed (agent/governor.py)
        if self.reply_writer is not None:
            self.reply_writer.extra_check = self._stock_claims
        self._local = threading.local()                  # one engine serves every call of a worker: per-turn state lives per thread
        self.last_level = ""
        self.prefix_stats: dict = {}
        self.stream_high_risk = True
        self.nlu = None
        self.decide_tokens, self.decide_deadline_s = 16, 5.0
        self.bridge_after_ms = 0.0                       # > 0: when the model's decision is expected to keep the caller waiting longer than this, a short
                                                         # "Let me check." is said at once while it is made (0 = never; voice sets it from voice.bridge_after_ms)

    @property
    def last_level(self) -> str:
        """The quality level of the last turn on THIS thread (each turn runs in its own thread, so callers never see each other's)."""
        return getattr(self._local, "level", "")

    @last_level.setter
    def last_level(self, value: str) -> None:
        self._local.level = value

    @property
    def prefix_stats(self) -> dict:
        return getattr(self._local, "prefix_stats", {})

    @prefix_stats.setter
    def prefix_stats(self, value: dict) -> None:
        self._local.prefix_stats = value

    # ------------------------------------------------------------------------------------------------ entry points
    def new_session(self) -> Session:
        return Session()

    def opening(self, s: Session) -> str:
        name = self.svc.cfg.app.agent_name
        c = self.svc.get_car(s.car) if s.car else None
        if name:
            line = T.greeting(name, c)
        else:
            line = "Hello, welcome to AGMC. " + (f"I see you're looking at the {T.short_name(c)}. Would you like the price, or to book a test drive?"
                                                 if c is not None else "What kind of car are you after?")
        s.remember("agent", line)
        return line

    def probe_model(self, record: bool = True) -> float | None:
        """Milliseconds the model takes to the first verified sentence of a model-written reply, measured on a made-up turn (no caller, no session), or None
        when it cannot be measured. Fed to the governor (`record_probe`), so the level is chosen from this and never found out on a caller's turn.
        The prompt has the shape of a real `full` turn (the writer's long instructions, a car's facts, a short history): that is where the time goes."""
        if self.reply_writer is None:
            return None
        stock = list(self.svc.inventory.search(SearchFilters(), 12))
        if not stock:
            return None
        # A different car and question every time: the same prompt twice is answered from Ollama's cache (480 ms against 1,100 ms for a new one, measured), but a real turn's
        # ending is always new. The instructions at the start are the same, as in a real call. (Not "tell me about it": the model answers that with "I'll tell you...",
        # which the checks reject.)
        self._probe_count = getattr(self, "_probe_count", -1) + 1
        n = self._probe_count
        cars = [stock[(n + i) % len(stock)] for i in range(3)]
        car = cars[0]
        caller = ("Is the first one a good car?", "How many kilometres does the first one have?", "What colour is the first one?", "Does the first one have any accidents?")[n % 4]
        w = WriterInput(caller=caller, risk="low", awaiting=True,
                        core=f"It's the {T.car_name(car)}, {car.color}, with {car.km:,} kilometres, at {T.money(car.price)}. Would you like to hear more?",
                        history=[("agent", "Which one would you like to hear more about?"), ("customer", caller)],
                        facts="\n".join(["Prices are fixed (no negotiation).", *fact_sheet(cars, self._branch_name).splitlines()]))      # as long as a real turn's: three cars
        t0 = time.perf_counter()
        gen = None
        try:
            gen = self.reply_writer.stream(w)
            first = next(iter(gen), None)
        except Exception:
            return None
        finally:
            if gen is not None and hasattr(gen, "close"):
                gen.close()                                              # stops the model: only the first sentence is wanted
        ms = (time.perf_counter() - t0) * 1000
        stopped = self.reply_writer.last_stream.get("stopped", "")
        if first is None and (not stopped or stopped.startswith("backend:") or stopped.startswith("high-risk")):
            return None                                                  # nothing was written, or the model could not be reached: not a measurement
        # A first sentence that a check then rejected still took this long to write, and a model that missed the first-sentence deadline is measured at the deadline
        # ("at least this slow"): both say what the governor needs, how fast the model is. (Seen on the Mac: the canned car's first sentence was rejected as
        # "new first-person promise", so the start-up probe said "not possible" with a perfectly working model.)
        if record and self.governor is not None:
            self.governor.record_probe(ms)
        return ms

    def predecide(self, s: Session, text: str) -> dict | None:
        """The model's decision for `text`, made ahead of the turn (voice/speculate.py: during Smart Turn's hold). None when there is nothing
        to prepare: the code routes these words itself, or the model has no room (a guess that may be thrown away must not crowd out real turns)."""
        if self._fast_decide(s, text) is not None:
            return None
        if self.governor is not None:
            if not self.governor.has_room():
                return None
            t0 = time.perf_counter()
            with self.governor.using_model():
                decision = self._decide(s, text)
            self.governor.record_decide((time.perf_counter() - t0) * 1000)
            return decision
        return self._decide(s, text)

    def handle_stream(self, s: Session, text: str, predecided=None, uncertain: bool = False):
        """Yield the spoken sentences, then the TurnResult (same contract as DialogueEngine.handle_stream).
        `predecided`: a future holding the decision already made for these words (see `predecide`), used instead of asking the model now.
        `uncertain`: the words came from a very quiet segment (room noise can be recognised as words): they may not end the call."""
        t0 = time.perf_counter()
        self._prefetch_turn.pop(id(s), None)                     # the caller spoke: prefetches still queued for this call are not sent
        dbg = TurnDebug(state_before=s.state.value)
        if s.ended:
            dbg.note = "session already ended"
            yield "Goodbye."
            yield TurnResult("Goodbye.", dbg, True)
            return
        words = ex.tokens_of(text)
        if not words or all(w in _FILLER for w in words):                    # "the", "um": a fragment the recognizer picked up, not a request
            line = "Sorry, I didn't catch that. Could you say it again?"
            dbg.intent, dbg.via, dbg.note = "none", "rules", "fragment: asked to repeat"
            s.remember("customer", text)
            s.remember("agent", line)
            yield line
            yield TurnResult(line, dbg, False)
            return
        if self._unfinished(s, words):                                    # "I'm looking for a..." / "I'm looking forward.": the end was cut off
            line = W.pick(s, "cut_off", CUT_OFF)
            dbg.intent, dbg.via, dbg.note = "none", "rules", "cut off mid-phrase: asked again"
            s.remember("customer", text)
            s.remember("agent", line)
            yield line
            yield TurnResult(line, dbg, False)
            return
        joined_words = " ".join(words)
        asked = expected.last_question(s.last_agent()).lower()
        if asked and W.INVITES.search(asked) and (_NO.match(joined_words) or _NEVER_MIND.search(joined_words)) and not _PLAIN_YES.match(joined_words):
            s.no_invite_until = len(s.history) + 12                        # "No, I want to know something else": no more invitations for the next few exchanges
        elif _visit_cue(joined_words):
            s.no_invite_until = 0                                          # the caller is talking about visiting: asking again is fine
        check = self._sounds_like_shown(s, text)
        if check is not None:                                              # "the Audi A4" right after a list with the RAV4 in it: ask, don't guess
            heard, near = check
            s.shown, s.car, s.state = [near, heard], None, State.ANSWERING
            line = f"Just to check, did you mean the {T.short_name(near)} or the {T.short_name(heard)}?"
            dbg.intent, dbg.via, dbg.note = "none", "rules", "sounds like a car just listed: asked which"
            s.remember("customer", text)
            s.remember("agent", line)
            yield line
            yield TurnResult(line, dbg, False)
            return
        decision = self._fast_decide(s, text)
        bridge = ""
        if decision is None and predecided is not None:
            try:
                decision = predecided.result(timeout=self.decide_deadline_s)     # started during the hold: done, or nearly
            except Exception:
                decision = None
            if decision is not None:
                dbg.note = (dbg.note + "; " if dbg.note else "") + "decision made during the hold"
        no_model = False
        if decision is None and self.governor is not None and not self.governor.model_allowed():
            decision, no_model = self._code_decision(s, text), True             # the model is overwhelmed: the code decides and speaks
            dbg.note = (dbg.note + "; " if dbg.note else "") + "model overloaded: shed, decided by code"
        if decision is None:
            bridge = self._bridge(s, text)
            if bridge:                                                     # the caller hears this while the model decides (and writes)
                yield bridge
            t_decide = time.perf_counter()
            if self.governor is not None:
                with self.governor.using_model():
                    decision = self._decide(s, text)
                self.governor.record_decide((time.perf_counter() - t_decide) * 1000)
            else:
                decision = self._decide(s, text)
            if decision.get("error"):                                      # no decision within the deadline: the code decides, nothing waits longer
                decision, no_model = self._code_decision(s, text), True
                dbg.note = (dbg.note + "; " if dbg.note else "") + "model too slow: decided by code"
        tool = decision.get("tool", "none")
        if tool not in TOOLS and tool != "booking_aside":
            tool = "none"
        if tool in ("reschedule_booking", "cancel_booking") and not s.known_bookings and (s.state in (State.SLOT, State.CONTACT) or s.draft_slot or s.offered_slots):
            # 28 Oct: nothing is booked yet, a booking is only being arranged. "Do." (for "two") and "I want to change the time", said while times were on offer, went to the tools
            # for bookings that exist and were answered "I don't see any test drives booked to move."
            if tool == "reschedule_booking":
                if s.draft_slot and re.search(r"\b(?:change|different|another|other|move)\b", joined_words):
                    s.draft_slot = None                                      # "change the time": the time chosen so far is let go and times are offered again
                tool = "prepare_booking"
            else:
                tool = "cancel_draft"
        if tool not in ("prepare_booking", "check_slots"):
            s.day_asks = 0                                                  # the booking talk was left: the next request for a day is a first ask again
        if uncertain and tool == "end_call":                               # 17:49 in the 27 Oct log: a segment at level 0.013-0.015 hung up two calls
            line = W.pick(s, "still_there", STILL_THERE)
            dbg.intent, dbg.via, dbg.note = "none", "rules", "goodbye heard at a very low level: asked if the caller is there, did not end the call"
            s.remember("customer", text)
            s.remember("agent", line)
            yield line
            yield TurnResult(line, dbg, False)
            return
        tool_text = decision.get("as_if") or text                         # a yes to an offer: the tool runs as if the offer's own words had been said (the caller's words are what is remembered)
        args = self._args_for(s, tool, decision.get("car_id"), tool_text) if tool in ("search_cars", "get_car", "check_slots", "prepare_booking") else {}
        if tool == "booking_aside":
            args = {"kind": decision.get("kind")}
        if tool == "search_cars" and not args and s.shown and decision.get("via") != "rules" and not (ex.wants_listing(text) or ex.wants_others(text) or expected.nothing_in_particular(text)):
            tool = "none"                                           # nothing searchable in it ("Bravo!", "throttle"): speech-recognition noise, not a request to list again
        picked = ex.selection(text, s.shown) if decision.get("via") != "rules" else None      # code's own routing already looked at this
        if picked and tool in ("none", "check_slots", "search_cars"):
            tool, args = "get_car", {"car_id": picked}                # "tell me about the second one" is about that car, whatever the model called it
        elif tool == "none" and s.car and (HAGGLE.search(" ".join(ex.tokens_of(text))) or self._offer_on(s, s.car, " ".join(ex.tokens_of(text)))):
            tool, args = "get_car", {"car_id": s.car}                 # "what's your best price" / "any discount": the answer is the fixed price
        dbg.intent, dbg.via = tool, "agent:rules" if decision.get("via") == "rules" else "agent"
        dbg.nlu_ms = (time.perf_counter() - t0) * 1000
        result = self._run(s, tool, args if isinstance(args, dict) else {}, tool_text, dbg)
        if decision.get("also") and s.car:
            result["core"] = f"{result['core']} {self._side_answer(s, decision['also'], 'car')}".strip()
        if decision.get("thanked"):                                        # "You're welcome." and then the question the booking is waiting on, as it was
            rest = _REACTION_LEAD.sub("", result["core"], count=1)
            result["core"] = "You're welcome. " + rest[:1].upper() + rest[1:]
        if decision.get("fallback"):                                       # the code could not tell what was asked and the model was not asked
            result["core"] = W.pick(s, "overload", ["Sorry, I didn't quite get that. Are you after a particular make, type of car or budget?",
                                                    "Could you tell me a bit more: a make, a type of car, or a budget?"])
        dbg.state_after = s.state.value
        spoken: list[str] = [bridge] if bridge else []
        if bridge:                                                       # "Let me check. Sure! What day ..." -> "Let me check. What day ..."
            result["core"] = _REACTION_LEAD.sub("", result["core"], count=1)
            result["core"] = result["core"][:1].upper() + result["core"][1:]
            dbg.note = (dbg.note + "; " if dbg.note else "") + "bridge said while the model decided"
        prefix, question = self._split_core(result["core"])
        level = "code" if no_model else self.governor.choose() if self.governor is not None else ("prefix" if self.prefix_mode else "full")
        if level == "full" and tool in _FACT_TOOLS and prefix:
            level = "prefix"           # a list of cars, a count, a comparison: pure facts, said by code; the model adds the question
        self.last_level = level
        speaks = self.model_speaks and level == "full"
        quiet = speaks and not self._slow_writer()             # model_speaks: an acknowledgement only while the model is slow to its first sentence
        opener, lead = ("", bridge) if bridge else ("" if quiet else self._opener(s, tool, result["core"]), "")
        if opener and level == "full":
            lead = opener                                       # said on its own while the model writes (see the full branch)
        elif opener:                                            # "Sure, we've got 4 BMWs ...": the caller hears something at once
            result["core"] = f"{opener} {_lower_first(result['core'], self._proper(s))}"
            prefix, question = self._split_core(result["core"])
        dbg.note = (dbg.note + "; " if dbg.note else "") + f"level {level}"
        if not s.ended and tool in EXACT_TOOLS and (self.governor is not None or self.prefix_mode):
            for part in _clauses(result["core"]):             # slot-filling questions, the read-back and the confirmation are spoken by code, word for word, at every level
                spoken.append(part)
                yield part
            self.prefix_stats = {"continuation": "exact-wording"}
        elif level == "code" and not s.ended:
            for part in _clauses(result["core"]):             # the model is overloaded or down: the code's own wording, at once
                spoken.append(part)
                yield part
        elif self.prefix_mode and not s.ended and tool in EXACT_TOOLS:
            for part in _clauses(result["core"]):             # slot-filling questions, the read-back and the confirmation are spoken by code, word for word; the model only chose the tool
                spoken.append(part)
                yield part
            self.prefix_stats = {"continuation": "exact-wording"}
        elif level == "prefix" and self.reply_writer is not None and not s.ended and prefix:
            dbg.risk = result["risk"]
            for sentence in self._prefix_reply(s, text, result, prefix, question, dbg):
                spoken.append(sentence)
                yield sentence
        elif self.reply_writer is not None and not s.ended:
            facts, stable = "\n".join(result["facts"]), ""
            core, awaiting = result["core"], result["awaiting"]
            if speaks:
                core, awaiting, facts, stable = self._speaking_input(s, text, result)
            w = WriterInput(caller=text, core=core, risk=result["risk"], history=list(s.history), facts=facts, awaiting=awaiting, stable=stable,
                            history_base=s.forgotten)
            allowed = f"{stable}\n{facts}\n{core}\n{text}"
            dbg.risk = result["risk"]
            t_write = time.perf_counter()
            if lead and not bridge:                             # said while the model writes; the model's own opener is then dropped
                spoken.append(lead)
                yield lead
            for attempt in range(2):                                    # a reply the checks rejected outright: ask the model again, more carefully (still the model's words)
                if attempt:
                    why = self.reply_writer.last_stream.get("stopped", "")
                    w = dataclasses.replace(w, facts=w.facts + f"\nYour previous attempt was rejected because: {why}. Write it again without that.")
                with (self.governor.using_model() if self.governor is not None else _nothing()):
                    for sentence in self.reply_writer.stream(w, temperature=0.2 if attempt else None):
                        if speaks:
                            self.fidelity["sentences"] += 1
                            leak = guards.leaks(sentence, allowed, self._car_words())
                            if leak:                                    # fail closed: not said; what was said stands, the code's wording follows if nothing was
                                self.fidelity["leaks"] += 1
                                dbg.note = (dbg.note + "; " if dbg.note else "") + "fact guard: " + "; ".join(leak)
                                break
                        if lead and len(spoken) == 1:
                            sentence = _after_opener(sentence, self._proper(s), new_sentence=not lead.rstrip().endswith(","))
                            if not sentence:
                                continue
                        if len(spoken) == (1 if lead else 0) and self.governor is not None:
                            self.governor.record_full((time.perf_counter() - t_write) * 1000)
                        spoken.append(sentence)
                        yield sentence
                info = self.reply_writer.last_stream
                dbg.lead_ms = info.get("first_ms", 0.0)
                if info.get("stopped"):
                    dbg.note = (dbg.note + "; " if dbg.note else "") + f"stream{' (retry)' if attempt else ''}: " + info["stopped"]
                if info.get("stopped") == FIRST_SENTENCE_DEADLINE:        # the model is slow, not wrong: asking again would only double the wait. The code's own wording follows below,
                    if self.governor is not None:                           # and the governor learns at once that the model-written level does not fit right now
                        self.governor.record_full((time.perf_counter() - t_write) * 1000)
                    break
                if len(spoken) > (1 if lead else 0):
                    break
            if lead and spoken == [lead]:                       # the model gave nothing usable after the opener: the code's facts
                spoken.append(result["core"] if bridge else _lower_first(result["core"], self._proper(s)))
                dbg.note = (dbg.note + "; " if dbg.note else "") + "FALLBACK: the model gave nothing usable, so the code-written wording was spoken"
                yield spoken[-1]
            if spoken and not spoken[-1].rstrip().endswith(("?", ".", "!")):
                # The reply was cut mid-sentence (the time limit or a check): finish with the facts the code has, then the question.
                core_facts, core_q = self._split_core(result["core"])
                if core_facts:
                    # What was said cannot be taken back. After a clause ("It's a 2016 Porsche Macan in black,") the reply carries on with what the caller has not heard,
                    # clause by clause: "with 91,000 kilometres on it, at 149,000 dirhams." (29 Oct: it said "and here is the full picture." and then the whole sentence again,
                    # name and colour included). After a half-word it starts over politely and gives the facts as a sentence of their own.
                    if spoken[-1].rstrip().endswith(","):
                        parts = _unheard(" ".join(spoken), core_facts)
                        if not parts:                           # all of it was said: the reply ends there (it said "and that's the main thing.", a filler; owner, 29 Oct)
                            spoken[-1] = spoken[-1].rstrip(" ,") + "."
                    else:
                        parts = [_rounding_off(spoken[-1]), core_facts]
                    for part in parts:
                        spoken.append(part)
                        yield part
                close = self._split_core(core)[1] if speaks else (core_q or W.anything_else(s))
                if close:                                       # (when the model speaks, a question is added only if the call is waiting on one)
                    spoken.append(close)
                    yield close
            if spoken:
                dbg.draft = " ".join(spoken)
                dbg.copy_ratio = round(SequenceMatcher(None, dbg.draft.lower(), result["core"].lower()).ratio(), 2)
        if speaks:
            self.fidelity["replies"] += 1
        if spoken == ([bridge] if bridge else []) and (not self.model_only or self.silent_fallback):
            if speaks:
                self.fidelity["fell_back"] += 1
            spoken.append(result["core"])
            dbg.note = (dbg.note + "; " if dbg.note else "") + "FALLBACK: the model gave nothing usable, so the code-written wording was spoken"
            yield result["core"]
        elif spoken == ([bridge] if bridge else []):
            dbg.note = (dbg.note + "; " if dbg.note else "") + "model_only: the model gave no usable reply, so nothing was said"
        reply = " ".join(spoken)
        s.remember("customer", text)
        s.remember("agent", reply)
        dbg.about = s.focus
        dbg.violations = voice_rules.check(reply, max_sentences=voice_rules.MAX_SENTENCES + 2, max_words=90)
        dbg.total_ms = (time.perf_counter() - t0) * 1000
        self.prefetch(s)                                         # the next reply's prompt start, read by the model server while this one is spoken
        yield TurnResult(reply, dbg, s.ended)

    def _proper(self, s: Session) -> set[str]:
        """Words that are names in this call, so they keep their capital after an opener: makes and models in stock, showroom names, the caller."""
        if not hasattr(self, "_names"):
            vocab = self.svc.inventory.vocabulary()
            names = {w for make, models in vocab.items() for x in (make, *models) for w in x.lower().split()}
            names |= {w for b in self.svc.cfg.branches for w in b.name.lower().split()} - {"showroom"}
            self._names = names
        return self._names | set((s.draft_name or "").lower().split())

    def _names_only_this_car(self, s: Session, words: list[str]) -> bool:
        """No car is named, or only the one in focus ("tell me more about the Land Rover" while discussing the Range Rover Sport)."""
        named = ex.find_vehicle(words, self.svc.inventory.vocabulary())
        car = self.svc.get_car(s.car) if s.car else None
        if not named.get("make"):
            return True
        return car is not None and named["make"] == car.make.lower() and named.get("model") in (None, car.model.lower())

    @staticmethod
    def _unfinished(s: Session, words: list[str]) -> bool:
        """The turn stops mid-phrase: "i'm looking for a", "show me the", "something under". Not while a booking detail is being given
        (a name or a number can end anywhere) and not a one-word answer ("any")."""
        if len(words) < 2 or s.read_back or s.state in (State.SLOT, State.CONTACT, State.CONFIRM):
            return False
        return words[-1] in _UNFINISHED_END or bool(_LOOKING_FORWARD.fullmatch(" ".join(words)))

    def _bridge(self, s: Session, text: str) -> str:
        """A short "Let me check." said at once when this turn must wait for the model's decision and the wait is expected to be noticeable.

        Natural, not robotic: only when the governor's live measurements say the caller would otherwise wait longer than `bridge_after_ms`
        (on a fast GPU it is never said), only for a request that sounds like a lookup (not "thanks", "hello", "yes"), never two turns running,
        and never the same wording twice in a row."""
        if self.bridge_after_ms <= 0 or self.governor is None or s.ended:
            return ""
        wait = self.governor.expected_wait_ms()
        if wait is None or wait < self.bridge_after_ms:
            return ""
        words = " ".join(ex.tokens_of(text))
        looks_up = bool(_BRIDGE_CUE.search(words)) or bool(ex.find_vehicle(ex.tokens_of(text), self.svc.inventory.vocabulary()).get("make"))
        if not looks_up or _THANKS.fullmatch(words) or _BYE.search(words) or _GREETING.fullmatch(words):
            return ""
        turn = sum(1 for who, _ in s.history if who == "customer")
        if s.last_variant.get("bridge_turn") == turn - 1 and wait < 2 * self.bridge_after_ms:
            return ""                                                   # said last turn: say it again only when the wait is really long
        line = W.pick(s, "bridge", BRIDGES)
        s.last_variant["bridge_turn"] = turn
        return line

    def _opener(self, s: Session, tool: str, core: str) -> str:
        """A short acknowledgement before a factual answer, when the answer does not already start with one. It is rendered in advance by
        the voice, so it plays the moment the turn ends while the rest of the reply is still being made."""
        if not self.openers or s.ended or tool not in _FACT_TOOLS + ("get_car",) or not core or _HAS_OPENER.match(core):
            return ""
        first = re.split(r"(?<=[.!?])\s", core, maxsplit=1)[0]
        negative = bool(_NEGATIVE.search(first))
        for _ in range(3):                                          # not 'So, our prices are fixed, so ...': no word said twice
            o = W.opener(s, negative=negative)
            if not set(o.lower().strip(",.").split()) & set(first.lower().replace(",", "").split()[:8]):
                return o
        return ""

    def _aside_kind(self, s: Session, text: str, words: list[str], joined: str) -> str | None:
        """While a booking is being filled in or read back: is this a side remark rather than a booking answer? "think" (one sec, let me think), "car" (a question
        about the car in focus or its price), "showroom" (hours, where). None when the words carry anything for the booking (a day or time, a name or number, a yes or
        no), or nothing a side answer can serve. 29 Oct sweep: such questions either ended the booking (the model's choice) or were not answered (the code's)."""
        if not s.car or parse_when(text, self.svc.clock(), self.svc.slots.tz(self.svc.cfg.branches[0].id)).found:
            return None
        c = ex.contact(text)
        if c.phone or c.heard_digits or ex.names_someone(text) or self._confirms(joined) or re.match(r"(?:no|nope|yes|yeah|actually|wait|cancel|change)\b", joined):
            if not (_THINKING.search(joined) and re.match(r"(?:wait|hmm|umm|uhh)\b", joined)):
                return None
        if _THINKING.search(joined) and len(words) <= 6:
            return "think"
        topics = [t for t in A.topics_in(joined) if t != "about"]
        if topics or A.asked_feature(joined) or HAGGLE.search(joined) or _VALUE.search(joined) or any(p.search(joined) for p in _ATTRIBUTE.values()):
            return "car"
        if A.showroom_topics(joined):
            return "showroom"
        return None

    def _new_time_in(self, s: Session, text: str, joined: str) -> bool:
        """At the read-back: do the words give a different day or time ("actually can we make it 9 instead", "Saturday instead")?"""
        w = parse_when(text, self.svc.clock(), self.svc.slots.tz(self.svc.cfg.branches[0].id))
        if w.day or w.time:
            return True
        return bool(re.search(r"\b(?:make it|do|how about|what about|change (?:it )?to|move (?:it )?to) (\d{1,2}|" + "|".join(_HOUR_WORDS) + r")\b(?! (?:days?|weeks?))", joined))

    def _booking_prompt(self, s: Session) -> str:
        """The question the booking was waiting on, asked again after a side answer: the times already offered, the missing detail, or the read-back's question."""
        if s.read_back:
            return "Shall I book it?"
        if s.offered_slots:
            return f"For the test drive, I can do {self._slot_text(s.offered_slots, self.svc.clock())}. Which suits you better?"
        if s.state is State.SLOT:
            return "Which day would suit you for the test drive?"
        missing = [x for x, have in (("name", s.draft_name), ("mobile number", s.draft_phone)) if not have]
        if not missing:
            return "Shall I book it?"
        ask = _REACTION_LEAD.sub("", W.ask_missing(s, missing, s.draft_name), count=1)
        return ask[:1].upper() + ask[1:]

    def _side_answer(self, s: Session, text: str, kind: str | None) -> str:
        """The answer to a side remark in the middle of a booking, from the data only ("" when there is nothing to say)."""
        words = " ".join(ex.tokens_of(text))
        car = self.svc.get_car(s.car) if s.car else None
        d = self.svc.details(s.car) if car is not None else None
        if kind == "think":
            return "No problem, take your time."
        if kind == "showroom":
            return " ".join(A.showroom_answer(self.svc.showroom(), words) or [])
        if car is None:
            return ""
        if HAGGLE.search(words) or _VALUE.search(words):
            return f"Our prices are fixed, so it's {T.money(car.price)}."
        parts = A.detail_answer(car, d, words) or [x for key, x in (("km", f"It has {car.km:,} kilometres."), ("colour", f"It's {car.color}."),
                                                                   ("price", f"It's {T.money(car.price)}."), ("year", f"It's a {car.year}.")) if _ATTRIBUTE[key].search(words)]
        return " ".join(parts)

    def _t_booking_aside(self, s, args, text, dbg) -> dict:
        """A side answer in the middle of a booking, then the booking's own question again. Nothing about the booking changes."""
        kind = args.get("kind")
        answer = self._side_answer(s, text, kind)
        dbg.tools.append(f"booking_aside({kind})")
        prompt = self._booking_prompt(s)
        return {"core": f"{answer or 'Let me check that at the showroom for you.'} {prompt}", "risk": "high", "awaiting": True,
                "facts": [f"A side question in the middle of the booking, answered from the data: {answer}", f"Then the booking's question again: {prompt}"]}

    def _slow_writer(self) -> bool:
        """Will the caller wait noticeably for the model's first sentence? The governor's live measure of it against `bridge_after_ms`, the same line the
        "Let me check." bridge uses. Phase 3 dropped the opener said while the model writes ("Sure," is rendered in advance and plays at once) because said every time
        it sounds scripted; it comes back only when the wait is long (29 Oct: on the laptop the first sound of a model-written reply took 1.9 s at the median, none under 1 s).
        On a server fast enough for the full level it is never said."""
        if self.bridge_after_ms <= 0 or self.governor is None:
            return False
        return self.governor.snapshot()["full_first_ms"] >= self.bridge_after_ms

    @staticmethod
    def _confirms(words: str) -> bool:
        """A yes to the read-back: "yes", "yes please", and also "yes, please book it", "go ahead", "book it", "confirm it"."""
        return bool(_PLAIN_YES.fullmatch(words) or _BOOK_YES.fullmatch(words) or _AGREE.fullmatch(words))

    @staticmethod
    def _invited_recently(s: Session) -> bool:
        """Did one of the agent's last two replies invite the caller to come and see the car or to book a test drive?"""
        agent = [t for who, t in s.history if who == "agent"][-2:]
        return any(W.INVITES.search(t.lower()) for t in agent)

    def _car_words(self) -> set[str]:
        if not hasattr(self, "_car_word_set"):
            self._car_word_set = guards.car_names(self.svc.inventory.vocabulary())
        return self._car_word_set

    def _car_facts(self, car: CarView, d) -> list[str]:
        """The facts that hold while this car is discussed: its line and its whole spec sheet (the start of the model's prompt, see _speaking_input and prefetch)."""
        return [fact_sheet([car], self._branch_name),
                f"Everything on the spec sheet of the {T.car_name(car)} (say only what the caller needs now): "
                + " ".join(x for x in (A.group_sentence(g, d) for g in A.FACT_GROUPS) if x) + (f" It has {A.DRIVE.get(d.drive.upper(), d.drive)}." if d.drive else "")
                + " Anything not listed here is not recorded: say you don't have it on record."]

    def prefetch(self, s: Session) -> int:
        """While the agent speaks, send the start of the next likely prompt to the model server, which keeps what it reads (29 Oct, the owner's call: the first
        question about a new car waited up to 3 s, most of it the model reading that car's facts). Up to three cars named in the reply just given, and the car being discussed,
        each with the conversation so far: exactly what the next reply's prompt starts with (ReplyWriter.prompt_start). One token is asked for and thrown away.
        In the background, one at a time; stops when the caller has started the next turn, and is skipped while another call is using the model.
        Returns how many prompts were queued."""
        if not (self.prefetch_on and self.model_speaks and self.reply_writer is not None) or s.ended or s.state in (State.SLOT, State.CONTACT, State.CONFIRM) or s.read_back:
            return 0
        said = s.last_agent()
        named = [c.stock_id for c in s.shown if c.model in said][:3]    # cars just read out or offered ("...the Mazda CX-5 at 89,900 and the Tucson..."): the caller may pick one
        ids = list(dict.fromkeys([*named, *([s.car] if s.car else [])]))
        starts = []
        for cid in ids:
            car, d = self.svc.get_car(cid), self.svc.details(cid)
            if car is not None and d is not None:
                stable = "\n".join([*self._car_facts(car, d), PRICE_RULE])
                starts.append(self.reply_writer.prompt_start(WriterInput(caller="", core="", risk="low", history=list(s.history), stable=stable, history_base=s.forgotten)))
        if not starts:
            return 0
        turn = self._prefetch_turn[id(s)] = len(s.history)
        writer = self.reply_writer

        def run():
            for user in starts:
                if self._prefetch_turn.get(id(s)) != turn or (self.governor is not None and self.governor.snapshot()["in_flight"] > 0):
                    return                                       # the caller has moved on, or a reply needs the model now
                try:
                    writer.backend.lead_in(writer.system, user, 1, None, 5.0)
                    self.prefetched += 1
                except Exception:
                    return                                       # the server is busy or down: a prefetch is never worth an error

        self._prefetcher.submit(run)
        return len(starts)

    def _speaking_input(self, s: Session, text: str, result: dict) -> tuple[str, bool, str, str]:
        """(standard wording, awaiting an answer, facts for this turn, facts that hold for this part of the call) for a reply the model writes whole
        (ARCH_PLAN Phase 3, `model_speaks`).

        * Only the generic "anything else?" closer is dropped (the model may end on a statement). An invitation to visit stays: `wording.offer_test_drive` gives one only when
          it is due (none after a no, never two answers running, at most one in three), otherwise the generic closer, so the pacing decides. The booking steps and a
          clarifying question ("Which car do you mean?") keep theirs. (First version dropped invitations too: 0 in 1,000 replies, and 2% of replies asked anything.)
        * The facts are the whole spec sheet of the car in focus, not the next one or two groups, with what the caller has already heard and what they asked about, so a
          follow-up ("and the boot?", "what about the colour") can land on the aspect asked.
        * Where the booking stands, so the model neither offers a second test drive nor talks as if one were booked.
        * Each fact once, and the ones that do not change while one car is discussed (its line and spec sheet, the price rule) apart from this turn's, so the writer
          puts them before the conversation and the model server reuses them (section 30, 29 Oct: the car in focus was given three times, about 300 of the 707 tokens
          read fresh on every turn; the other cars' spec sheets were given too)."""
        core, awaiting = result["core"], result["awaiting"]
        facts_part, question = self._split_core(core)
        booking_step = s.state in (State.SLOT, State.CONTACT, State.CONFIRM) or s.read_back
        if question and not booking_step and is_generic_closer(question) and facts_part:
            core, awaiting = facts_part, False
        car = self.svc.get_car(s.car) if s.car else None
        d = self.svc.details(s.car)
        stable, lines = [], []
        focus = f"- {T.car_name(car)}:" if car is not None and d is not None else None
        for line in result["facts"]:
            if line.startswith("Prices are fixed"):
                stable.append(line)
            elif focus and line.startswith(focus):
                continue                                 # given once, in the stable part below
            elif line.startswith("Answered from the spec sheet: ") and line.split(": ", 1)[1] in core:
                continue                                 # the standard wording says it already
            elif line.startswith("- ") and " dirhams" in line and line.startswith(tuple(f"- {T.car_name(c)}:" for c in s.shown)):
                lines.append(line[:line.index(" dirhams") + len(" dirhams")])      # another car in the call: name, colour and price
            else:
                lines.append(line)
        if lines and lines[-1] == "Cars in this call (true facts you may use):":
            lines.pop()
        if focus:
            stable[:0] = self._car_facts(car, d)
            told = s.said_about.get(s.car, [])
            if told:
                lines.append("The caller has already heard about its " + ", ".join(told) + ". Do not repeat those unless they ask again.")
        words = " ".join(ex.tokens_of(text))
        topics = [t for t in A.topics_in(words) if t != "about"] + [{"km": "mileage"}.get(k, k) for k, p in _ATTRIBUTE.items() if p.search(words)]
        if topics:                                      # every aspect named ("the colour, mileage, engine and price": all four, not only the spec-sheet one)
            lines.append("The caller is asking about: " + ", ".join(topics) + ". Answer all of that first.")
        if s.booking is not None or s.known_bookings:
            lines.append("A test drive is already booked in this call or on the website: do not offer another one.")
        elif booking_step or s.draft_slot or s.offered_slots:
            lines.append("A test drive is being arranged: do not say it is booked or confirmed, and do not suggest times; the system asks for those.")
        if not awaiting:
            lines.append("The call is not waiting for an answer: end on a statement unless a question is really needed.")
        return core, awaiting, "\n".join(lines), "\n".join(stable)

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

    @staticmethod
    def _novel_words(sentence: str, *context: str) -> set[str]:
        """Content words of `sentence` that are neither grammar, an ordinary way of asking, nor in any of the `context` texts (a word counts with its endings: see / seeing)."""
        def stem(w: str) -> str:
            w = re.sub(r"(?:'s|'d|'ll|'re|'ve|n't)$", "", w)
            for suffix in ("ing", "ed", "es", "s"):
                if len(w) > len(suffix) + 3 and w.endswith(suffix):
                    return w[: -len(suffix)]
            return w
        known = {stem(w) for text in context for w in re.findall(r"[a-z0-9']+", text.lower())} | {stem(w) for w in _GRAMMAR | _ASKING}
        return {w for w in re.findall(r"[a-z']+", sentence.lower()) if len(w) > 2 and stem(w) not in known}

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
        novel = self._novel_words(sentence, question, prefix, caller, facts, history)
        if novel:
            problems.append("brings up something the code did not ask about (" + ", ".join(sorted(novel)[:3]) + "): ask only the question given")
        said = {w for w in re.findall(r"[a-z0-9']+", prefix.lower()) if len(w) > 3}
        mine = {w for w in re.findall(r"[a-z0-9']+", sentence.lower()) if len(w) > 3}
        if mine and len(mine & said) / len(mine) >= 0.6:
            problems.append("repeats what was just said")
        theirs = {w for w in re.findall(r"[a-z0-9']+", caller.lower()) if len(w) > 3}
        if len(mine & theirs) >= 2 and len(mine & theirs) / len(mine) >= 0.6:     # 30 Oct test call: "I don't have the warranty on record. Does it come with a warranty?"
            problems.append("repeats the customer's own question back: ask the customer the question given")
        if len(sentence.split()) > 18:
            problems.append("too long for a follow-up")
        if question and not sentence.rstrip().endswith("?"):
            problems.append("does not ask the question the call is waiting on")
        if _CALLER_VOICE.match(sentence):
            problems.append("sounds like the customer, not you: ask the customer")
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
                f"The call now needs this answered: \"{question}\"\nContinue your reply with ONE short sentence (at most 14 words) that asks exactly that, in your own words, and nothing else: "
                "no other topic, no feature, no other car. Do not repeat anything you already said, and mention no prices, numbers, policies or promises.")
        reasons = ""
        for attempt in range(2):
            u = user + (f"\nYour previous attempt was rejected because: {reasons}. Write it again without that." if reasons else "")
            buf, m = "", None
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
        dbg.note = (dbg.note + "; " if dbg.note else "") + f"prefix mode: follow-up rejected ({reasons}); the code's own question was said"
        yield question                                         # never end on a statement: the caller must know what is being asked

    # ------------------------------------------------------------------------------------------------ the decision
    def _code_decision(self, s: Session, text: str) -> dict:
        """The action when the model is not asked (overloaded or too slow): a search if the words say what to look for, the car being discussed
        if there is one, otherwise a guiding question."""
        words = " ".join(ex.tokens_of(text))
        if s.read_back:                                                    # "Shall I book it?" is waiting: never answer with the car's facts (28 Oct test call)
            if self._confirms(words):
                return {"tool": "confirm_booking", "via": "rules"}
            if re.match(r"(?:no|nope|not quite|wrong|that's wrong|actually)\b", words):
                return {"tool": "fix_booking", "via": "rules"}
            return {"tool": "prepare_booking", "car_id": s.car, "via": "rules"}    # not clear: the details are read back again and the question asked again
        if s.state in (State.SLOT, State.CONTACT) or s.draft_slot:
            return {"tool": "prepare_booking", "car_id": s.car, "via": "rules"}    # a booking is being arranged: carry on with it (it asks for what is missing), not the car's facts
        about_it = s.car and re.search(r"\b(?:it|its|it's|this one|that one|this car|that car)\b", words) and not _CHEAPER.search(words)
        if not about_it and (ex.search_filters(text, self.svc.inventory.vocabulary()) or _FUEL.search(words) or _LOW_KM.search(words) or _CHEAPER.search(words)):
            return {"tool": "search_cars", "via": "rules"}             # 30 Oct replay: "Is it white?" about the Camry searched for white cars (and the Camry was lost)
        if s.car:
            return {"tool": "get_car", "car_id": s.car, "via": "rules"}
        if s.filters or s.shown:                                           # a search is under way (29 Oct: "4 Toyotas... What budget?" then the code asked "a make, a type of car,
            return {"tool": "search_cars", "via": "rules"}                 # or a budget?" as if nothing had been said): read out what it has found
        return {"tool": "none", "via": "rules", "fallback": True}

    def _code_hint(self, text: str) -> tuple[dict, list[str]]:
        """The make / model the customer's words match (spelling-tolerant), and up to 5 stock ids of it."""
        found = ex.find_vehicle(ex.tokens_of(text), self.svc.inventory.vocabulary())
        if not found.get("make"):
            return found, []
        return found, [c.stock_id for c in self.svc.inventory.search(SearchFilters(make=found["make"], model=found.get("model")), 5)]

    def _options(self, s: Session, hint_ids: list[str]) -> list[str]:
        """Every answer the decision may give: each tool, and the tools about one car with each car that is in play right now (the code hint,
        the car in focus, the cars just listed). With a server that can constrain its output the model cannot name anything else."""
        ids = list(dict.fromkeys([*hint_ids, *([s.car] if s.car else []), *(c.stock_id for c in s.shown)]))[:12]
        car_tools = ("get_car", "check_slots", "prepare_booking")
        return [*TOOLS, *(f"{t} {i}" for t in car_tools for i in ids)]

    def _context(self, s: Session, text: str = "", hint: tuple[dict, list[str]] | None = None) -> str:
        tz = self.svc.slots.tz(self.svc.cfg.branches[0].id)
        now = self.svc.clock().astimezone(tz)
        focus = self.svc.get_car(s.car) if s.car else None
        draft = (f"car {s.car or '-'}, name {s.draft_name or '-'}, phone {s.draft_phone or '-'}, "
                 f"when {T.when_full(s.draft_slot) if s.draft_slot else '-'}, read back to the customer: {'yes, waiting for their yes/no' if s.read_back else 'no'}")
        shown = "; ".join(f"{i + 1}) {c.stock_id} {c.year} {c.make} {c.model}" for i, c in enumerate(s.shown)) or "none"
        convo = "\n".join(f"{'AGENT' if who == 'agent' else 'CUSTOMER'}: {t}" for who, t in s.history[-4:]) or "(start of the call)"
        # The customer's words are matched to the stock by code (spelling-tolerant), so the model never has to know the whole inventory.
        found, ids = hint if hint is not None else self._code_hint(text)
        hint = ""
        if found.get("make"):
            hint = f"CODE HINT (spelling corrected): the customer's words match make {found['make']}" + (f", model {found['model']}" if found.get("model") else "") + f", ids {ids}\n"
        return (f"NOW: {now:%A %d %B %Y, %H:%M} (Dubai). Open every day.\n{hint}"
                f"CAR IN FOCUS: {focus.stock_id + ' ' + T.car_name(focus) if focus else 'none'}\nLAST SHOWN: {shown}\n"
                f"SEARCH SO FAR: {s.filters or 'nothing'}\nBOOKING DRAFT: {draft}\nCONVERSATION:\n{convo}\n")

    def _fast_decide(self, s: Session, text: str) -> dict | None:
        """The turns whose action the code can tell for certain, without asking the model (saves its 0.25-0.4 s decision call):
        a pick from the list just read out, the next booking detail while a booking is being filled in, and a plain yes to the read-back.
        Anything else, or anything mixed ("yes, but at five"), goes to the model as before."""
        words = ex.tokens_of(text)
        joined = " ".join(words)
        if s.pending is not None and s.pending.kind in ("cancel_which", "move_which"):   # the answer to "which one?" comes first, whatever it names
            return {"tool": "cancel_booking" if s.pending.kind == "cancel_which" else "reschedule_booking", "via": "rules"}
        booking_flow = s.read_back or s.state in (State.SLOT, State.CONTACT)          # a booking is being filled in or read back
        offered_rest = re.search(r"\b(?:go through the rest|name the others)\b", s.last_agent().lower())
        if offered_rest and not booking_flow and (self._confirms(joined) or re.fullmatch(r"(?:yes |yeah |ok(?:ay)? )?(?:go on|go ahead|the rest|the others|name them|all of them|go through them)(?: please)?", joined)):
            if "name the others" in s.last_agent().lower():
                return {"tool": "browse_makes", "as_if": "the others", "via": "rules"}     # "yes" to "...or shall I name the others?"
            return {"tool": "search_cars", "as_if": "what other ones do you have", "via": "rules"}   # "yes" to "...or shall I go through the rest?"
        aside = self._aside_kind(s, text, words, joined) if booking_flow else None
        if aside:                                                                # "is it automatic?", "one sec" while times are on offer: answer, and the booking goes on
            return {"tool": "booking_aside", "kind": aside, "car_id": s.car, "via": "rules"}
        if not booking_flow and _COLOUR_Q.search(joined) and not ex.search_filters(text, self.svc.inventory.vocabulary()).get("color"):
            if s.car and _SINGULAR_REF.search(joined) and not _PLURAL_REF.search(joined):
                return {"tool": "get_car", "car_id": s.car, "via": "rules"}       # "what colour is it?": the car just discussed
            return {"tool": "colours", "via": "rules"}                           # "what colours are these cars?" / "what colours do you have?"
        if not booking_flow:                                                     # managing bookings and comparing come before "which car": they name cars too
            if s.known_bookings and _CANCEL.search(joined) and not _MOVE.search(joined):
                return {"tool": "cancel_booking", "via": "rules"}
            if s.known_bookings and _MOVE.search(joined):
                return {"tool": "reschedule_booking", "via": "rules"}
            if _MY_BOOKINGS.search(joined) and not _CANCEL.search(joined) and not _MOVE.search(joined):
                return {"tool": "my_bookings", "via": "rules"}
            if _COMPARE.search(joined):
                return {"tool": "compare_cars", "via": "rules"}
        if not booking_flow and _CHEAPER.search(joined) and not re.search(r"\b(?:the|that) cheaper one\b", joined):
            return {"tool": "search_cars", "via": "rules"}                       # "something cheaper": a new search, before "the cheaper one" (a pick)
        if not booking_flow and s.car and _OPINION.search(joined):
            return {"tool": "get_car", "car_id": s.car, "via": "rules"}           # "is it reliable?" about the car just described
        if not booking_flow and s.car and "about" in A.topics_in(joined) and self._names_only_this_car(s, words):
            return {"tool": "get_car", "car_id": s.car, "via": "rules"}           # "anything else about the car I should know?" / "what else should I know about it?"
        picked = ex.selection(text, s.shown)
        cue = _NOT_A_VISIT.sub(" ", joined)                                      # "Come on, can you do 150?", "does it come with...": not a visit
        if not booking_flow and _BOOKING_CUE.search(cue) and re.search(r"\b(?:see|come|visit|drive|try|book|test)\b", cue):
            # 28 Oct: "Yes, I want to come and see the Macan." is a booking, not a pick from the list: naming the car must not turn it into "tell me about it" (three times in one call)
            named_car = ex.find_vehicle(words, self.svc.inventory.vocabulary())
            only = list(self.svc.inventory.search(SearchFilters(make=named_car.get("make"), model=named_car["model"]), 2)) if named_car.get("model") else []
            target = picked or (only[0].stock_id if len(only) == 1 else None) or s.car
            if target:
                return {"tool": "prepare_booking", "car_id": target, "via": "rules"}
        if picked and s.state not in (State.SLOT, State.CONTACT, State.CONFIRM):
            return {"tool": "get_car", "car_id": picked, "via": "rules"}
        named = ex.find_vehicle(words, self.svc.inventory.vocabulary())
        if not booking_flow and s.shown:                                         # "the A4" (heard "the a four") right after a list with the A4 in it
            hit = [c for c in s.shown if c.model.lower() == (named.get("model") or "") or re.search(
                rf"\b(?:{re.escape(c.model.lower())}|{re.escape(' '.join(ex.tokens_of(c.model)))})\b", joined)]
            if len(hit) == 1:
                return {"tool": "get_car", "car_id": hit[0].stock_id, "via": "rules"}     # "the A4" right after a list with the A4 in it
        if not booking_flow and named.get("make") and not named.get("model") and len(words) <= 5 and not s.car \
                and not _OTHER_RANGE.search(joined) and not A.superlative_topic(joined):
            return {"tool": "search_cars", "via": "rules"}                        # "Toyota" / "maybe a Toyota": that make
        if _GREETING.fullmatch(joined):
            return {"tool": "none", "via": "rules"}
        if not booking_flow and (s.narrow_asks > 0 or s.shown) and s.filters and not s.car and _WHICH_ONES.fullmatch(joined):
            return {"tool": "search_cars", "via": "rules"}                       # "which ones?" after "There are 8 SUVs ..." (or a list): read them out
        if not booking_flow and s.car and s.described == s.car and _MORE_ABOUT.fullmatch(joined):
            return {"tool": "get_car", "car_id": s.car, "via": "rules"}            # "something else please" after facts about this car: the next facts
        if not booking_flow and s.makes_said and _REST.search(joined):
            return {"tool": "browse_makes", "via": "rules"}                       # "what are the 9 other makes?"
        if not booking_flow and _OTHER_RANGE.search(joined) and not ex.find_vehicle(words, self.svc.inventory.vocabulary()).get("make") \
                and not re.search(r"\b(?:cheaper|bigger|smaller|similar|like (?:it|this|that))\b", joined) \
                and not set(ex.search_filters(text, self.svc.inventory.vocabulary())) - {"make", "model"}:   # "any other red cars?" is a search for red
            return {"tool": "browse_makes", "via": "rules"}                       # "what other car models do you have?": the rest of the stock, not more of one make
        if _BYE.search(joined) and len(words) <= 8 and not booking_flow:
            return {"tool": "end_call", "via": "rules"}
        if not booking_flow and _WHAT_STOCK.search(joined) and not ex.find_vehicle(words, self.svc.inventory.vocabulary()).get("make"):
            if s.filters and s.shown and not s.car and _ASKS_TO_SEE.search(joined):
                return {"tool": "search_cars", "via": "rules"}                    # 27 Oct: "I'm not looking for any specific model, what do you have?" after the Toyotas: list the Toyotas, not the other makes
            return {"tool": "browse_makes", "via": "rules"}                       # "what cars do you have?"
        if not booking_flow:
            named0 = ex.find_vehicle(words, self.svc.inventory.vocabulary())
            if named0.get("model") and not _COMPARE.search(joined) and not _OTHER_RANGE.search(joined):
                hits = list(self.svc.inventory.search(SearchFilters(make=named0.get("make"), model=named0["model"]), 2))
                if len(hits) == 1 and hits[0].stock_id != s.car:
                    return {"tool": "get_car", "car_id": hits[0].stock_id, "via": "rules"}   # "BMW X5", "tell me about the Land Cruiser"
        if not booking_flow and s.car and (HAGGLE.search(joined) or self._offer_on(s, s.car, joined)) and not _CHEAPER.search(joined) \
                and not set(ex.search_filters(text, self.svc.inventory.vocabulary()) or {}) - {"max_price", "min_price"} \
                and not ex.find_vehicle(words, self.svc.inventory.vocabulary()).get("make"):
            return {"tool": "get_car", "car_id": s.car, "via": "rules"}      # "can you do 150,000?" is an offer on this car, not a new budget
        about_it = s.car and re.search(r"\b(?:it|its|it's|this one|that one|this car|that car)\b", joined)
        searchable = set(ex.search_filters(text, self.svc.inventory.vocabulary())) - {"make", "model"}
        if not booking_flow and not about_it and (ex.search_filters(text, self.svc.inventory.vocabulary()) or _FUEL.search(joined) or _LOW_KM.search(joined)) \
                and (not _OTHER_RANGE.search(joined) or searchable) and not A.superlative_topic(joined):
            return {"tool": "search_cars", "via": "rules"}                       # "an SUV under 100k", "a white one", "anything electric"
        if not booking_flow and about_it and (ex.search_filters(text, self.svc.inventory.vocabulary()) or {}).get("color") \
                and re.search(r"\b(?:is it|is this|is that|it's|its|does it come)\b", joined) and not _CHEAPER.search(joined):
            return {"tool": "get_car", "car_id": s.car, "via": "rules"}      # "Is it white?" about this car: its colour (30 Oct test call: the model listed every colour we have)
        if not booking_flow and (_FINANCE.search(joined) or (not s.car and _OPINION.search(joined) and not _FAMILY.search(joined))):
            return {"tool": "none", "via": "rules"}                              # finance, "something reliable": answered by code in _t_none
        if not booking_flow and not s.car and _FAMILY.search(joined):
            return {"tool": "best_of", "via": "rules"}                           # "something for my family": the roomiest we have
        if _NEVER_MIND.search(joined) and len(words) <= 10:
            if booking_flow and not (_THINKING.search(joined) and not re.search(r"\b(?:never mind|nevermind|not now|another time|maybe later|i'll think about it|i will think about it)\b", joined)):
                return {"tool": "cancel_draft", "via": "rules"}
            if booking_flow:
                return {"tool": "booking_aside", "kind": "think", "car_id": s.car, "via": "rules"}
            return {"tool": "none", "via": "rules"}                              # "never mind, maybe later": no pressure, nothing to change
        if A.superlative_topic(joined) and (s.shown or s.car):
            return {"tool": "best_of", "via": "rules"}                           # "which one is the fastest / cheapest / newest?": from the data
        if _REPEAT.search(joined) and len(words) <= 8:
            return {"tool": "repeat", "via": "rules"}
        if s.pending is not None and s.pending.kind == "time_check" and s.car and (_PLAIN_YES.fullmatch(joined) or re.fullmatch(r"(?:yes |yeah )?(?:that's right|correct|exactly)", joined)):
            return {"tool": "prepare_booking", "car_id": s.car, "via": "rules"}  # "yes" to "did you say 5 PM?"
        if s.pending_cancel is not None and (_PLAIN_YES.fullmatch(joined) or _NO.match(joined)):
            return {"tool": "cancel_booking", "via": "rules"}                    # the answer to "shall I cancel ...?"
        if s.pending is not None and s.pending.kind in ("cancel_which", "move_which"):
            return {"tool": "cancel_booking" if s.pending.kind == "cancel_which" else "reschedule_booking", "via": "rules"}
        q = expected.last_question(s.last_agent()).lower()
        if s.car and not booking_flow and _OFFERED_DRIVE.search(q):
            w_now = parse_when(text, self.svc.clock(), self.svc.slots.tz(self.svc.cfg.branches[0].id))
            if w_now.found:
                return {"tool": "prepare_booking", "car_id": s.car, "via": "rules"}   # "tomorrow at four" right after "want to book a test drive?"
        offered = _OFFERED_DRIVE.search(q) or (s.described == s.car and q and not re.search(
            r"\b(?:which|what|how|when|where|who|anything else|hear|know|tell|features?|more|details?|about|other|else|similar|compare)\b", q))
        if s.car and not s.read_back and s.state not in (State.SLOT, State.CONTACT, State.CONFIRM) and (_PLAIN_YES.fullmatch(joined) or _AGREE.fullmatch(joined)) and offered:
            return {"tool": "prepare_booking", "car_id": s.car, "via": "rules"}       # "yes" to "want to check it out?" / "shall I set up a test drive?"
        if s.car and not booking_flow and (_PLAIN_YES.fullmatch(joined) or _AGREE.fullmatch(joined)) and re.search(r"\bhear more\b", q) and not re.search(r"\bwhich\b", q):
            return {"tool": "get_car", "car_id": s.car, "via": "rules"}           # "yes" to "would you like to hear more about it?" (30 Oct, Bedrock call: the model confirmed a booking that did not exist)
        if _THANKS.fullmatch(joined) and (s.state in (State.SLOT, State.CONTACT, State.CONFIRM) or s.read_back):
            return {"tool": "prepare_booking", "car_id": s.car, "via": "rules", "thanked": True}   # mid-booking (29 Oct: "Thank you." after the number dropped the question for the name)
        if _THANKS.fullmatch(joined):
            return {"tool": "none", "via": "rules"}                               # "thank you" after a booking is not a request to confirm it again
        if not s.car and len(s.shown) >= 2 and not booking_flow and _PLAIN_YES.fullmatch(joined) and re.search(r"\bwhich\b", q):
            return {"tool": "which_car", "via": "rules"}                          # "yes" to "which one would you like?"
        if not s.car and s.shown and not booking_flow and (_visit_cue(joined) or parse_when(text, self.svc.clock(),
                self.svc.slots.tz(self.svc.cfg.branches[0].id)).found) and re.search(r"\b(?:book|test drive|see|come|visit|tomorrow|today|am|pm|morning|afternoon|evening|day)\b", joined):
            return {"tool": "prepare_booking", "via": "rules"}                    # a day or a booking with no car chosen: prepare_booking asks which car
        if not booking_flow and (_CHEAPER.search(joined) or ((_SPORTY.search(joined) or _SEATS.search(joined)) and not s.car)):
            return {"tool": "search_cars", "via": "rules"}                       # "something cheaper", "anything with seven seats", "something sporty"
        if not booking_flow and s.car and _visit_cue(joined) and re.search(r"\b(?:see|come|visit|drive|try|book|test)\b", joined):
            return {"tool": "prepare_booking", "car_id": s.car, "via": "rules"}   # "can I see it this weekend?"
        if not booking_flow and s.car and _SEE_CAR.search(joined) and self._invited_recently(s):
            return {"tool": "prepare_booking", "car_id": s.car, "via": "rules"}   # "I want to see the car" after "Would you like to come and see it?"
        if not booking_flow:
            if A.showroom_topics(joined) and not _visit_cue(joined) and not re.search(r"\bmy (?:number|phone|mobile|whatsapp|email)\b", joined):
                return {"tool": "showroom_info", "via": "rules"}
        if s.shown and not s.car and s.state not in (State.SLOT, State.CONTACT, State.CONFIRM) and expected.nothing_in_particular(text):
            return {"tool": "search_cars", "via": "rules"}                       # "any range or style" after a narrowing question: read the list out
        found = ex.find_vehicle(words, self.svc.inventory.vocabulary())
        current = self.svc.get_car(s.car) if s.car else None
        if (found.get("make") and _LOOKING.search(joined) and not found.get("model") and not _visit_cue(joined)
                and (current is None or current.make.lower() != found["make"])):
            return {"tool": "search_cars", "via": "rules"}                       # "now I'm looking for any BMW": a new search, not the car before
        if not s.car or s.ended:
            return None
        detail = (A.topics_in(joined) or A.asked_feature(joined)) and self.svc.details(s.car) is not None
        if (s.state not in (State.SLOT, State.CONTACT, State.CONFIRM) and (_ABOUT_THIS_CAR.search(joined) or HAGGLE.search(joined) or _VALUE.search(joined) or detail)
                and not _visit_cue(joined) and not found.get("make")):
            return {"tool": "get_car", "car_id": s.car, "via": "rules"}      # "how many kilometres has it done?" is about the car just discussed
        if s.read_back and (both := re.match(r"(yes(?: please)?|yeah|yep|sure|ok(?:ay)?|correct|that'?s right|perfect)(?: and| but| also| oh and)? (.+)", joined)) \
                and self._confirms(both[1]) and self._aside_kind(s, both[2], both[2].split(), both[2]) == "car":
            return {"tool": "confirm_booking", "also": both[2], "via": "rules"}       # "yes, and does it have a warranty?": booked, and answered (29 Oct sweep)
        if s.state is State.CONFIRM and s.read_back and self._confirms(" ".join(words)):
            return {"tool": "confirm_booking", "via": "rules"}
        if s.read_back and self._new_time_in(s, text, joined):
            return {"tool": "prepare_booking", "car_id": s.car, "via": "rules"}    # "actually can we make it 9 instead": the time changes and is read back again (29 Oct sweep)
        if s.read_back and re.match(r"(?:no|nope|not quite|wrong|that's wrong|actually)\b", joined):
            return {"tool": "fix_booking", "via": "rules"}                         # "no, the name is Omer" / "no, it's 055..." / "no"
        if s.state in (State.SLOT, State.CONTACT):
            tz = self.svc.slots.tz(self.svc.cfg.branches[0].id)
            w = parse_when(text, self.svc.clock(), tz)
            bare_hour = s.offered_slots and re.fullmatch(r"(?:at |around |about )?(?:\d{1,2}|" + "|".join(_HOUR_WORDS) + r")(?: o'?clock| please)?", joined)
            if w.time is not None or bare_hour or (s.offered_slots and self._offered_pick(s, text) is not None):
                return {"tool": "prepare_booking", "car_id": s.car, "via": "rules"}   # "no, 2 PM", "the afternoon one": a time, whatever comes before it
            if s.state is State.SLOT and _FLEXIBLE.search(joined):
                return {"tool": "prepare_booking", "car_id": s.car, "via": "rules"}   # "any day", "what's free": offer the next free times
            if s.state is State.CONTACT and ex.names_someone(text):
                return {"tool": "prepare_booking", "car_id": s.car, "via": "rules"}   # "No, but the booking under Fatima": a name, whatever comes before it (29 Oct)
            if not _CHANGE.search(" ".join(words)):
                ct = ex.contact(text)
                if w.found or ct.phone or (s.state is State.CONTACT and (ct.name or ct.heard_digits)):     # (digits that are not a whole number: ask again, do not leave it to the model)
                    return {"tool": "prepare_booking", "car_id": s.car, "via": "rules"}
        return None

    def _decide(self, s: Session, text: str) -> dict:
        """The model names the tool (and the car). Everything else - filters, name, phone, day and time - is taken from the customer's words by code."""
        hint = self._code_hint(text)
        user = self._context(s, text, hint) + f"CUSTOMER (now): {text}\nTOOL:"
        options = self._options(s, hint[1])
        try:
            choose = getattr(self.backend, "choose", None)              # a server that can constrain its answer to the listed actions (vLLM)
            raw = choose(DECIDE_SYSTEM, user, options, self.decide_tokens, self.decide_deadline_s) if choose else None
            if raw is None:
                raw = self.backend.lead_in(DECIDE_SYSTEM, user, self.decide_tokens, None, self.decide_deadline_s)
        except Exception as e:                                          # the model did not answer in time: do nothing rather than guess
            return {"tool": "none", "args": {}, "error": type(e).__name__}
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
        if raw.strip() in options:                                      # an exact listed answer: "get_car S1042", "search_cars"
            tool, _, car_id = raw.strip().partition(" ")
            return {"tool": tool, "car_id": car_id or None}
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
            args["heard_digits"] = ct.heard_digits if not ct.phone else None
            asked_name = s.state is State.CONTACT or bool(re.search(r"\b(?:name|who should i put)\b", expected.last_question(s.last_agent()), re.I))
            # Without "my name is", the leading words are a name only when a name was asked for or a number comes with them
            # ("Mohammed, 050 ..."): "Yeah, I want to come and see it" after "Want to come and see it?" is not a name.
            args["name"] = names.correct(ct.name) if (cue or ((asked_name or ct.phone) and not when_found)) else None   # "tomorrow at four" is never a name; "Achmed" is Ahmed (nlu/names.py)
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
        # What the model is given, once each (27 Oct: the car in focus was listed twice, as the focus and as one of the cars shown, and "prices are fixed" was said in two wordings;
        # about 150 tokens read by the model on every turn for nothing). The tool's own weaker price line is dropped in favour of the stronger one below.
        focus = [self.svc.get_car(s.car)] if s.car and self.svc.get_car(s.car) else []
        unique = list({c.stock_id: c for c in focus + list(s.shown)}.values())
        context = [f"- {T.car_name(c)}: {c.color}, {c.price:,} dirhams{self._spec_facts(c)}" for c in unique][:5]
        out["facts"] = [*[f for f in out.get("facts", []) if not f.startswith("Prices are fixed (no negotiation)")],
                        PRICE_RULE,
                        *(["Cars in this call (true facts you may use):", *context] if context else [])]
        return out

    def _spec_facts(self, c: CarView) -> str:
        """The site's spec sheet for a car, for the model to answer from ("is it a good family car?"): seats, drive, economy, boot, warranty."""
        d = self.svc.details(c.stock_id)
        if d is None:
            return ""
        parts = [f"{d.seats} seats" if d.seats is not None else "", A.DRIVE.get(d.drive.upper(), d.drive) if d.drive else "", c.fuel,
                 f"{d.power_hp} hp" if d.power_hp is not None else "", f"uses {d.consumption}" if d.consumption else "",
                 f"boot {d.boot_litres} litres" if d.boot_litres is not None else "", A.spoken(d.accident) if d.accident else "",
                 f"{d.owners} previous owner(s)" if d.owners is not None else "", f"warranty {A.spoken_warranty(d.warranty)}" if d.warranty else ""]
        return "".join(f", {x}" for x in parts if x)                    # what the sheet does not record is left out, not guessed

    def _t_browse_makes(self, s, args, text, dbg) -> dict:
        """What else is in stock, by make: after a booking or a search, "what other cars do you have?" means the rest of the showroom."""
        current = (self.svc.get_car(s.car).make if s.car and self.svc.get_car(s.car) else None) or (s.filters.get("make") or "").title() or None
        counts = {}
        for mk in self.svc.inventory.vocabulary():
            n = self.svc.inventory.count(SearchFilters(make=mk))
            if n:
                counts[mk] = n
        names = {mk: next((c.make for c in self.svc.inventory.search(SearchFilters(make=mk), 1)), mk.title()) for mk in counts}
        words = " ".join(ex.tokens_of(text))
        if s.makes_said and _REST.search(words):                    # "what are the 9 other makes?": the ones not named yet, all of them
            rest = sorted((mk for mk in counts if names[mk] not in s.makes_said), key=lambda mk: -counts[mk])
            if rest:
                said = [names[mk] for mk in rest]
                s.makes_said += said
                listed = said[0] if len(said) == 1 else ", ".join(said[:-1]) + " and " + said[-1]
                return {"core": f"The others are {listed}. Do any of those catch your eye?",
                        "facts": ["In stock by make: " + ", ".join(f"{names[mk]} {counts[mk]}" for mk in rest)], "awaiting": True}
        others = sorted((mk for mk in counts if not current or names[mk].lower() != current.lower()), key=lambda mk: -counts[mk])
        total = sum(counts[mk] for mk in others)
        shown = [names[mk] for mk in (others if len(others) - 6 <= 2 else others[:6])]     # one or two more makes are named too
        more = len(others) - len(shown)
        parts = shown + ([f"{more} other makes"] if more > 0 else [])
        listed = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1] if parts else ""
        lead = f"Besides {current}, we've got {total} more cars" if current else f"We've got {total} cars"
        s.filters, s.shown, s.car, s.state = {}, [], None, State.INTEREST
        s.makes_said = ([current] if current else []) + shown
        return {"core": f"{lead}, from {listed}. Which make would you like to hear about" + (", or shall I name the others?" if more else "?"),
                "facts": [f"In stock by make: " + ", ".join(f"{names[mk]} {counts[mk]}" for mk in others)], "awaiting": True}

    def _t_best_of(self, s, args, text, dbg) -> dict:
        pool = list(s.shown) or ([self.svc.get_car(s.car)] if s.car else [])
        if not pool:                                                      # chosen from the stock, within what was asked for (29 Oct: "a family SUV under 100,000" got the Carnival, a van at 126,000)
            asked = {**s.filters, **ex.search_filters(text, self.svc.inventory.vocabulary())}
            wanted = {k: v for k, v in asked.items() if k in SearchFilters.model_fields and v not in (None, "")}
            pool = [self.svc.get_car(c.stock_id) for c in self.svc.inventory.search(SearchFilters(**wanted), 200)]
            if not pool:
                return self._t_search_cars(s, args, text, dbg)           # nothing matches: said as a search with no results
            text = text if A.superlative_topic(" ".join(ex.tokens_of(text))) else "best for a family"
        if s.filters and len(pool) < 2:
            pool = [self.svc.get_car(c.stock_id) for c in self.svc.inventory.search(SearchFilters(**{k: v for k, v in s.filters.items() if k in ("make", "model", "max_price")}), 50)]
        line = A.best_of([c for c in pool if c], self.svc.details_for(c.stock_id for c in pool if c), " ".join(ex.tokens_of(text)))
        if not line:
            return self._t_none(s, args, text, dbg)
        winner, sentence = line
        if not (list(s.shown) or s.car):                                  # chosen from the whole stock, not from a list just read out
            sentence = sentence.replace(" of these", " we have")
        s.car, s.state = winner.stock_id, State.ANSWERING
        return {"core": f"{sentence} Would you like to hear more about it?", "facts": [sentence], "awaiting": True}

    def _t_none(self, s, args, text, dbg) -> dict:
        question = "What are you looking for today?" if not s.car and not s.shown else W.anything_else(s)
        if _GREETING.fullmatch(" ".join(ex.tokens_of(text))):
            return {"core": f"Hi! {question}", "facts": ["The customer said hello."], "awaiting": True}
        said = " ".join(ex.tokens_of(text))
        if _FINANCE.search(said):
            return {"core": f"I can't quote finance over the phone, but our team can go through the options with you at the showroom. {question}",
                    "facts": ["Finance terms are not quoted by the assistant."], "awaiting": False}
        if _OPINION.search(said) and not s.car:
            return {"core": "Every car we sell passes a 150-point inspection and comes with an AGMC warranty. What kind of car are you after?",
                    "facts": ["All cars pass the 150-point inspection and carry an AGMC warranty."], "awaiting": True}
        if _NEVER_MIND.search(" ".join(ex.tokens_of(text))):
            return {"core": f"No problem at all, take your time. {question}", "facts": ["The customer is not ready; no pressure."],
                    "awaiting": False}
        if _THANKS.fullmatch(" ".join(ex.tokens_of(text))):
            return {"core": f"You're welcome! {question}", "facts": ["The customer said thank you. Say you're welcome and ask if there is anything else."], "awaiting": False}
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
        if not make and not args.get("model") and _ANY_CARS.search(" ".join(ex.tokens_of(text))) and \
                not ex.find_vehicle(ex.tokens_of(text), vocab).get("make"):
            f.pop("make", None)                                          # "do you have any red cars?" after Toyotas: any make now
            f.pop("model", None)
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
        words = " ".join(ex.tokens_of(text))
        base = (self.svc.get_car(s.car) if s.car else None) or (min(s.shown, key=lambda c: c.price) if s.shown else None)
        if _CHEAPER.search(words) and base is not None and not args.get("max_price"):
            f["max_price"] = filters_price = base.price - 1                  # "something cheaper": under the car being discussed
            filters = filters.model_copy(update={"max_price": filters_price})
        if _LOW_KM.search(words) and not f.get("max_km"):
            f["max_km"] = 40000
            filters = filters.model_copy(update={"max_km": 40000})
        fuel_m = _FUEL.search(words)
        fuel = {"ev": "electric", "plug in hybrid": "hybrid"}.get(fuel_m.group(1), fuel_m.group(1)) if fuel_m else f.get("fuel")
        if fuel:
            f["fuel"] = fuel
        need_seats = 7 if (_SEATS.search(words) or f.get("seats")) else None          # kept for "any budget" after "seven seats"
        sporty = bool(_SPORTY.search(words) or f.get("sporty"))
        if fuel or need_seats or sporty:                                 # in the database query: the fuel, and what only the spec sheet records (seats, power)
            spec = {"fuel": fuel, "min_seats": need_seats, "min_power": 250 if sporty else None, "sort": "power" if sporty else "price"}
            if (need_seats or sporty) and not self.svc.inventory.count(SearchFilters(include_unavailable=True, min_seats=0 if need_seats else None,
                                                                                    min_power=0 if sporty else None)):
                spec = {"fuel": fuel}                                    # no car has it recorded (the demo stock): it cannot be searched on
            elif need_seats or sporty:
                f["seats"] = f"{need_seats}+" if need_seats else f.get("seats")
                f["sporty"] = True if sporty else f.get("sporty")
                f = {k: v for k, v in f.items() if v is not None}
            filters = filters.model_copy(update=spec)
        res = self.svc.search_cars(filters)
        full = None
        dbg.tools.append(f"search_cars({f}) -> {res.total} match(es)")
        wanted = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in f.items()) or "anything"
        if not res.cars and _CHEAPER.search(words) and base is not None:
            wider = self.svc.search_cars(SearchFilters(max_price=base.price - 1))
            s.filters = {"max_price": base.price - 1}
            if wider.cars:
                s.shown, s.car, s.state = list(wider.cars), None, State.ANSWERING
                same = f"The {T.short_name(base)} is our cheapest {base.make} at {T.money(base.price)}. " if f.get("make") else ""
                listed = W.list_line(s, list(wider.cars), wider.total)
                lead = listed if listed.startswith("I") else listed[0].lower() + listed[1:]
                ask_one = "Would you like to hear more about it?" if len(wider.cars) == 1 else "Which one would you like to hear more about?"
                return {"core": f"{same}Cheaper than that, {lead} {ask_one}",
                        "facts": [f"Nothing cheaper within {wanted}; cheaper across all makes: {wider.total}."], "awaiting": True}
            return {"core": f"The {T.short_name(base)} at {T.money(base.price)} is the cheapest car we have right now. Would you like to hear more about it?",
                    "facts": ["Nothing in stock is cheaper."], "awaiting": True}
        if not res.cars and ex.wants_others(text) and f.get("make"):
            return self._t_browse_makes(s, args, text, dbg)
        if not res.cars and ex.wants_others(text) and len(s.shown) >= 1 and f:          # "any other red cars?" after every red one was read out
            what = W.what_they_asked(list(s.shown), f)
            names = [f"the {T.short_name(c)}" for c in s.shown]
            listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
            only = f"That's the only one of our {what} right now: {listed}." if len(names) == 1 else f"Those are all the {what} I have right now: {listed}."
            return {"core": f"{only} {'Would you like to hear more about it?' if len(names) == 1 else 'Which one would you like to hear more about?'}",
                    "facts": [f"Search for {wanted}: no others beyond the {len(names)} already read out."], "awaiting": True}
        if not res.cars:
            # Nothing matches EVERYTHING asked so far. If a filter from earlier in the call is what empties it (a budget from "something
            # cheaper", then "do you have a BMW?"), look again with only what was just asked, and say both: never "we have no BMWs" when we do.
            said_now = {k: v for k, v in ex.search_filters(text, self.svc.inventory.vocabulary()).items() if v not in (None, "")}
            if make:
                said_now["make"] = make
            carried = {k: v for k, v in f.items() if k not in said_now and k in ("make", "max_price", "min_price", "year_min", "max_km", "color", "body_type", "model")}
            if said_now and carried:
                try:
                    wider = self.svc.search_cars(SearchFilters(**{k: (BodyType(v) if k == "body_type" else v) for k, v in said_now.items()
                                                                if k in ("make", "model", "color", "max_price", "year_min", "max_km", "body_type")}))
                except ValueError:
                    wider = None
                if wider is not None and wider.cars:
                    s.filters, s.shown, s.car, s.state = dict(said_now), list(wider.cars), (wider.cars[0].stock_id if wider.total == 1 else None), State.ANSWERING
                    limits = self._filters_said(carried)
                    what = W.what_they_asked(list(wider.cars), said_now)
                    lead = f"{T.cap(limits)} I don't have one, but" if limits else "Not with everything you mentioned, but"
                    if set(carried) == {"make"} and set(said_now) <= {"color", "body_type"}:     # "I don't have a red Toyota, but ..."
                        name = next((c.make for c in self.svc.inventory.search(SearchFilters(make=carried["make"]), 1)), str(carried["make"]).title())
                        body = str(said_now.get("body_type") or "")
                        thing = " ".join(x for x in (said_now.get("color"), name, body.upper() if body == "suv" else body) if x)
                        lead = f"I don't have {'an' if thing[:1].lower() in 'aeiou' else 'a'} {thing}, but"
                    elif carried.get("make") or carried.get("body_type"):   # 30 Oct: "As a Toyota I don't have one, but" -> "I don't have a Toyota under 60,000 dirhams, but"
                        both = {**carried, **said_now}
                        name = next((c.make for c in self.svc.inventory.search(SearchFilters(make=both["make"]), 1)), str(both["make"]).title()) if both.get("make") else ""
                        body = str(both.get("body_type") or "")
                        thing = " ".join(x for x in (both.get("color"), name, body.upper() if body == "suv" else body) if x) or "car"
                        rest = self._filters_said({k: v for k, v in both.items() if k in ("max_price", "year_min", "max_km")})
                        lead = f"I don't have {'an' if thing[:1].lower() in 'aeiou' else 'a'} {thing}{' ' + rest if rest else ''}, but"
                    if wider.needs_narrowing:
                        body = f"{lead} there are {wider.total} {what}, from {T.money(min(c.price for c in wider.cars))}. What budget would work for you?"
                    else:
                        listed = W.list_line(s, list(wider.cars), wider.total)
                        ask_one = "Would you like to hear more about it?" if len(wider.cars) == 1 else "Which one would you like to hear more about?"   # one car: not "which one" (30 Oct)
                        body = f"{lead} {listed[0].lower() + listed[1:] if not listed.startswith('I') else listed} {ask_one}"
                    return {"core": body, "awaiting": True,
                            "facts": [f"With everything asked ({wanted}) nothing matches. Without the earlier {limits or 'filters'}: {wider.total} {what} in stock.",
                                      "Do NOT say there are none of them in stock: there are."]}
            stock_of_make = self.svc.inventory.count(SearchFilters(make=f.get("make"))) if f.get("make") else 0
            if stock_of_make and f.get("color"):                     # "a red one" when no BMW is red: say what colours the BMWs are
                colours = sorted({c.color for c in self.svc.inventory.search(SearchFilters(make=f["make"]), 50)})
                name = next((c.make for c in self.svc.inventory.search(SearchFilters(make=f["make"]), 1)), f["make"].title())
                said = colours[0] if len(colours) == 1 else ", ".join(colours[:-1]) + " and " + colours[-1]
                s.filters = {k: v for k, v in f.items() if k != "color"}
                return {"core": f"None of our {plural_make(name)} is {f['color']}: they're {said}. Would any of those colours work for you?", "awaiting": True,
                        "facts": [f"{name} cars in stock come in {said}; none in {f['color']}."]}
            return {"core": "I don't have anything that matches that. Would you like to try another make or type of car?", "awaiting": True,
                    "facts": [f"Search for {wanted}: no cars in stock match ALL of it."] +
                             ([f"There ARE {stock_of_make} {f['make']} cars in stock; only this combination has none. Do not say there are none."] if stock_of_make else [])}
        s.state = State.ANSWERING
        s.shown = list(res.cars)
        s.car = res.cars[0].stock_id if res.total == 1 else None
        if len(res.cars) > 1 or res.needs_narrowing:                   # a list is names and prices only: no kilometres etc. for the model to mangle mid-sentence
            sheet = [f"- {T.car_name(c)}: {c.color}, {c.price:,} dirhams" for c in res.cars]
        else:
            sheet = fact_sheet(list(res.cars), self._branch_name).splitlines()
        if res.needs_narrowing and not (unchanged and s.narrow_asks > 0):
            s.narrow_asks += 1
            return {"core": f"{W.count_line(s, res.total, list(res.cars), f)} {W.narrow_question(f)}",
                    "facts": [f"Search for {wanted}: exactly {res.total} cars match (say the number {res.total}, never 'a few' or 'several'), too many to read out. Examples:", *sheet,
                              "Ask one narrowing question instead of listing them all, about something the customer has not said yet: "
                              + (", ".join(k for k, have in (("budget", f.get("max_price") or f.get("min_price")), ("make", f.get("make")), ("body type", f.get("body_type")),
                                                              ("colour", f.get("color"))) if not have) or "model") + "."], "awaiting": True}
        more = res.total - len(res.cars)
        if not res.needs_narrowing:
            s.narrow_asks = 0
        from ..tools.results import NAME_THE_REST_UP_TO, SearchResult
        if 0 < more <= NAME_THE_REST_UP_TO:                               # 4 or 5 of them: read them all, not "plus one more"
            every = full if full is not None else [c.view() for c in self.svc.inventory.search(filters, res.total)]
            res = SearchResult(total=res.total, cars=tuple(every[:res.total]), needs_narrowing=False)
            s.shown, more = list(res.cars), 0
            sheet = [f"- {T.car_name(c)}: {c.color}, {c.price:,} dirhams" for c in res.cars]
        ask = W.hear_more_or_book(s) if len(res.cars) == 1 else (
            "Which one would you like to hear more about, or shall I go through the rest?" if more > 0 else "Which one would you like to hear more about?")
        listed = W.list_line(s, list(res.cars), res.total)               # "..., and 5 more" when more than two are left
        if more > 0:
            return {"core": f"{listed} {ask}", "facts": [f"Search for {wanted}: {res.total} match(es):", *sheet,
                                                              *([f"There are {more} more not listed: say how many in total ({res.total}) and read out only these {len(res.cars)}."] if more > 0 else [])], "awaiting": True}
        return {"core": f"{listed} {ask}", "facts": [f"Search for {wanted}: {res.total} match(es):", *sheet], "awaiting": True}

    def _t_get_car(self, s, args, text, dbg) -> dict:
        car = self._car_arg(s, args)
        if car is None:
            return {"core": "Which car do you mean?", "facts": ["No car could be identified from the request."], "awaiting": True}
        s.car = car.stock_id
        if all(c.stock_id != car.stock_id for c in s.shown):
            s.shown = [car]
        s.state = State.ANSWERING
        s.makes_said = []                                                # a car is being discussed: "what else" is no longer "the other makes"
        dbg.tools.append(f"get_car({car.stock_id}) -> {car.status.value}")
        facts = ["Prices are fixed (no negotiation).", *fact_sheet([car], self._branch_name).splitlines()]
        words = " ".join(ex.tokens_of(text))
        ask = lambda: W.offer_test_drive(s)        # the closing question, chosen only when it is used (an invitation to visit is not made in two answers running, nor after a no)
        d = self.svc.details(car.stock_id)
        told = s.said_about.setdefault(car.stock_id, [])
        topics = A.topics_in(words)
        pushed = bool(HAGGLE.search(words)) or self._offer_on(s, car, words)
        if pushed or _VALUE.search(words):                               # "best price?", "can you do 150?", "is that a good price?": the fixed price, said once
            rec = s.haggles.setdefault(car.stock_id, {"pushes": 0, "offered": []})
            if pushed:
                rec["pushes"] += 1
                rec["asked"] = ex.asked_price(words, car.price) or rec.get("asked")     # "any discount at all?" after "can you do 150?": still about 150,000
                if rec["pushes"] >= 2:                                   # asked again: the price stays, cars near the asked price are offered instead
                    return self._cheaper_instead(s, car, words, rec, facts, dbg)
            return {"core": f"Our prices are fixed, so the {T.car_name(car)} is {T.money(car.price)}. {ask()}", "facts": facts, "awaiting": True}
        asked_colour = (ex.search_filters(text, self.svc.inventory.vocabulary()) or {}).get("color")
        if asked_colour and car.color and re.search(r"\b(?:is it|is this|is that|it's|its|does it come)\b", words):    # "Is it white?" about this car (30 Oct replay)
            same = str(asked_colour).lower() in car.color.lower()
            return {"core": f"{'Yes' if same else 'No'}, it's {car.color.lower()}. {ask()}", "facts": facts, "awaiting": True}
        if d is not None and topics and set(topics) == {"about"} and not A.asked_feature(words):
            return self._more_about(s, car, d, told, facts, ask)         # "tell me more", "anything else about it?": a group of facts not told yet
        detail = A.detail_answer(car, d, words)
        if detail:                                                       # warranty, accident, engine, boot, "does it have a sunroof?": from the spec sheet
            A.mark_said(told, topics)
            return {"core": " ".join(detail) + " " + ask(), "facts": facts + [f"Answered from the spec sheet: {' '.join(detail)}"], "awaiting": True}
        just_said = s.described == car.stock_id
        if just_said:                                                    # the caller heard the full line a moment ago: answer only what was asked
            part = [x for key, x in (("km", f"It has {car.km:,} kilometres."), ("colour", f"It's {car.color}."), ("price", f"It's {T.money(car.price)}."),
                                     ("year", f"It's a {car.year}.")) if _ATTRIBUTE[key].search(words)]
            if part:
                return {"core": " ".join(part) + " " + ask(), "facts": facts, "awaiting": True}
            if d is not None and _FAMILY.search(words):                  # "is it a good family car?": what a family would ask about
                said = [x for x in (A.group_sentence("practical", d), A.group_sentence("condition", d)) if x]
                if said:
                    A.mark_said(told, ["boot", "accident", "service"])
                    return {"core": " ".join(said) + " " + ask(), "facts": facts, "awaiting": True}
            if d is not None:                                            # "is it reliable?", "something else about it": the next facts not told yet, the car's condition first for an opinion question
                return self._more_about(s, car, d, told, facts, ask, prefer=("condition",) if _OPINION.search(words) else ())
        s.described = car.stock_id
        return {"core": f"{W.car_line(s, car)} {ask()}", "facts": facts, "awaiting": True}

    def _cheaper_instead(self, s: Session, car: CarView, words: str, rec: dict, facts: list[str], dbg: TurnDebug) -> dict:
        """The caller asked for a lower price on this car a second time (owner, 29 Oct). The price never changes; cars that cost about what was asked
        ("can you do 150?") are offered instead, each at its own fixed price: the same type of car first, then the nearest in price. With no amount
        named, cars somewhat cheaper than this one. Each push names cars not named yet; with none left, the closest ones are named again."""
        asked = rec.get("asked")
        target = asked if asked and asked < car.price else round(car.price * 0.85 / 5000) * 5000
        pool = [v for v in (self.svc.get_car(c.stock_id) for c in self.svc.inventory.search(SearchFilters(max_price=car.price - 1), 50))
                if v is not None and v.stock_id != car.stock_id and v.price < car.price]
        stays = f"The {T.short_name(car)} stays at {T.money(car.price)}, as our prices are fixed."
        dbg.tools.append(f"cheaper_instead({car.stock_id}, push {rec['pushes']}, asked {asked or 'nothing'}) -> {len(pool)} cheaper in stock")
        if not pool:
            return {"core": f"{stays} It's the lowest-priced car we have. {W.offer_test_drive(s)}", "facts": facts, "awaiting": True}
        fresh = [c for c in pool if c.stock_id not in rec["offered"]]
        rank = lambda c: (c.price > target * 1.05, c.body_type != car.body_type, abs(c.price - target))
        near = sorted(fresh or pool, key=rank)[:2]
        named = " and ".join(f"the {T.car_name(c)} at {T.money(c.price)}" for c in near)
        if not fresh:
            offer = f"The closest I have are still {named}."
        elif all(c.price > target * 1.05 for c in near):
            offer = f"The closest I have to {T.money(target)} {'is' if len(near) == 1 else 'are'} {named}."
        elif rec["offered"]:
            offer = f"There's also {named}."
        elif asked:
            offer = f"If you'd like something nearer {T.money(target)}, there's {named}."
        else:
            offer = f"If you'd like something for less, there's {named}."
        rec["offered"] = list(dict.fromkeys([*rec["offered"], *(c.stock_id for c in near)]))
        s.shown = near                                                  # "the first one" / "tell me about the Q7" now means one of these
        close = "Would you like to hear about it?" if len(near) == 1 else "Would you like to hear about one of them?"
        return {"core": f"{stays} {offer} {close}", "facts": facts + fact_sheet(near, self._branch_name).splitlines(), "awaiting": True}

    def _more_about(self, s: Session, car, d, told: list[str], facts: list[str], ask, prefer: tuple[str, ...] = ()) -> dict:
        """The next one or two groups of facts about this car that the caller has not heard; when there are none, say so rather than repeat one."""
        s.described = car.stock_id
        again = A.group_sentence(prefer[0], d) if prefer and prefer[0] in told else None
        if again:                                                        # asked "is it reliable?" and the condition was already told: it IS the answer, said again briefly
            return {"core": again + " " + ask(), "facts": facts + [f"Answered from the spec sheet: {again}"], "awaiting": True}
        sentences = A.next_facts(d, told, prefer)
        if not sentences:
            return {"core": W.nothing_more(s), "facts": facts + ["Everything on the spec sheet has already been told to the caller. Do not repeat any of it."], "awaiting": True}
        return {"core": " ".join(sentences) + " " + ask(), "facts": facts + [f"Answered from the spec sheet: {' '.join(sentences)}"], "awaiting": True}

    def _sounds_like_shown(self, s: Session, text: str):
        """(the car heard, the listed car it sounds like) when the caller seems to name a car just read out but the recognizer heard another
        one: 'A4' after a list with the RAV4, 'Camri'... None when the heard car is itself in the list, or nothing listed sounds like it."""
        if not s.shown or s.state in (State.SLOT, State.CONTACT, State.CONFIRM):
            return None
        found = ex.find_vehicle(ex.tokens_of(text), self.svc.inventory.vocabulary())
        model = (found.get("model") or "").lower()
        if not model or any(c.model.lower() == model for c in s.shown):
            return None
        heard = next((self.svc.get_car(c.stock_id) for c in self.svc.inventory.search(SearchFilters(make=found.get("make"), model=model), 1)), None)
        if heard is None or (found.get("make") and _LOOKING.search(" ".join(ex.tokens_of(text)))):   # "do you have an Audi A4?": a real new request
            return None

        def alike(a: str, b: str) -> bool:
            digits_a, digits_b = re.sub(r"\D", "", a), re.sub(r"\D", "", b)
            return SequenceMatcher(None, a, b).ratio() >= 0.6 or (digits_a and digits_a == digits_b and (a.endswith(b) or b.endswith(a)))

        near = next((c for c in s.shown if alike(c.model.lower().replace(" ", ""), model.replace(" ", ""))), None)
        return (heard, near) if near is not None else None

    @staticmethod
    def _offered_pick(s: Session, text: str):
        """Which of the times just offered the caller means: "the first one", "the second", "the morning one", "the later one"."""
        words = " ".join(ex.tokens_of(text))
        slots = sorted(s.offered_slots)
        said = {int(w) for w in re.findall(r"\b(\d{1,2})\b", words)} | {n for w, n in _HOUR_WORDS.items() if re.search(rf"\b{w}\b", words)
                                                                          and (w != "one" or re.search(r"\bone (?:o'?clock|pm|p m|in the)\b", words))}   # "the morning one" names no hour
        hits = [x for x in slots if (x.hour % 12 or 12) in said]
        if len(hits) == 1:                                       # "9", "nine", "nine please": the offered time with that hour
            return hits[0]
        if said and not hits:                                    # an hour that was not offered ("9 in the morning" when 12 and 4 were): not "the morning one" (28 Oct replay: booked 12)
            return None
        if re.search(r"\b(?:first|earlier|earliest|morning)\b", words):
            return slots[0]
        if re.search(r"\b(?:second|last|later|latest|afternoon|evening)\b", words) and len(slots) > 1:
            return slots[-1]
        return None

    def _booked_car(self, s: Session) -> str:
        car = self.svc.get_car(s.booking.stock_id) if s.booking else None
        return T.car_name(car) if car else "other car"

    def _slot_text(self, slots, now) -> str:
        return " or ".join(T.when_short(x, now) for x in slots)

    def _offer_next_slots(self, s: Session, car, text: str = "", intro: str = "") -> dict:
        """Stop asking which day and offer the next free times: for \"any day\" / \"what's free\", for a day with nothing free, and for a second question in a row.
        The earliest and a middle one of the first day that has room (the earliest two for \"as soon as possible\"); a part of the day in the words is respected."""
        tz, now = self.svc.slots.tz(car.branch_id), self.svc.clock()
        alts = list(self.svc.slots.alternatives(car, now, k=24))
        pod = part_of_day(text)
        if pod:
            part = [a for a in alts if {"morning": a.hour < 12, "afternoon": 12 <= a.hour < 17, "evening": a.hour >= 17}[pod]]
            alts = part or alts
        s.day_asks = max(s.day_asks, 1)                         # (not 0: the next answer with no day in it is then offered times again, not asked "which day?" again: ask, offer, ask, offer, 28 Oct)
        if not alts:
            return {"core": (intro + " " if intro else "") + "I don't see any free times in the next two weeks. Would you like me to put you through to a colleague?", "risk": "high", "awaiting": True,
                    "facts": ["There are no free test-drive slots in the booking window."]}
        day = alts[0].astimezone(tz).date()
        same = [a for a in alts if a.astimezone(tz).date() == day]
        soonest = bool(re.search(r"\b(?:soon|soonest|earliest|asap|first)\b", " ".join(ex.tokens_of(text))))
        pick = same[:2] if soonest else ([same[0], same[len(same) // 2]] if len(same) > 1 else same[:1])
        s.offered_slots, s.slot_day, s.state = pick, day, State.SLOT
        times = " or ".join(T.clock_part(x) for x in pick)
        return {"core": W.offer_slots(s, T.day_name(day, now), times, intro), "risk": "high", "awaiting": True,
                "facts": ([intro] if intro else []) + [f"Free on {day:%A}: {times} (and others). The customer had not chosen a day; these are offered."]}

    def _t_check_slots(self, s, args, text, dbg) -> dict:
        car = self._car_arg(s, args)
        if car is None:
            return {"core": self._which_car_to_drive(s), "facts": ["No car identified."], "awaiting": True}
        tz, now = self.svc.slots.tz(car.branch_id), self.svc.clock()
        when = str(args.get("when") or text)
        w = parse_when(when, now, tz)
        if not w.day:
            if s.day_asks >= 1 or _FLEXIBLE.search(" ".join(ex.tokens_of(when))):       # already asked once, or the caller left it to us: offer, do not ask again
                return self._offer_next_slots(s, car, when)
            s.day_asks += 1
            return {"core": "Which day would suit you?", "facts": ["The customer has not said a day yet."], "awaiting": True}
        free = list(self.svc.check_test_drive_slots(car.stock_id, w.day).slots)
        pod = part_of_day(when)
        if pod:
            free = [f for f in free if {"morning": f.hour < 12, "afternoon": 12 <= f.hour < 17, "evening": f.hour >= 17}[pod]]
        dbg.tools.append(f"check_slots({car.stock_id}, {w.day}) -> {len(free)} free")
        if not free:
            alts = list(self.svc.slots.alternatives(car, combine(w.day, datetime.min.time(), tz)))[:2]
            return {"core": f"{W.checking(s)} We have no free slots then. How about {self._slot_text(alts, now)}?", "risk": "high", "awaiting": True,
                    "facts": [f"No free slots on {w.day:%A}.", f"Next free: {self._slot_text(alts, now)}."]}
        pick = [free[0], free[len(free) // 2]] if len(free) > 1 else [free[0]]
        return {"core": f"{W.checking(s)} I can do {self._slot_text(pick, now)}. Which works better for you?", "risk": "high", "awaiting": True,
                "facts": [f"Free slots on {w.day:%A}: {self._slot_text(pick, now)} (and others)."]}

    def _t_prepare_booking(self, s, args, text, dbg) -> dict:
        car = self._car_arg(s, args)
        if car is None:
            w = parse_when(str(args.get("when") or text), self.svc.clock(), self.svc.slots.tz(self.svc.cfg.branches[0].id))
            asked = f" The customer asked for {w.day:%A}" + (f" at {w.time:%H:%M}" if w.time else "") + "." if w.day else ""
            return {"core": self._which_car_to_drive(s), "risk": "high", "awaiting": True,
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
        slot_set_now = False
        delegated = False
        if when and when.lower() not in ("null", "none"):
            w = parse_when(when, now, tz)
            day, tm = w.day, w.time
            if s.pending is not None and s.pending.kind == "time_check" and day is None and tm is None and _PLAIN_YES.match(" ".join(ex.tokens_of(text))):
                day, tm = s.pending.value.date(), s.pending.value.time()      # "yes" to "did you say 5 PM?": that time
            if tm is None and day is None and s.offered_slots:          # "the first one", "the afternoon one", "the later one"
                pick = self._offered_pick(s, when)
                if pick is not None:
                    day, tm = pick.date(), pick.time()
            if tm is None and day is None and s.offered_slots and _DELEGATES.search(" ".join(ex.tokens_of(when))):
                first = sorted(s.offered_slots)[0]                      # "any day suits me" / "you choose" right after two times were offered: take the earlier one
                day, tm = first.date(), first.time()
                delegated = True
            if tm is None and day is None and (s.offered_slots or s.draft_slot):
                said = " ".join(ex.tokens_of(text))
                bare = re.fullmatch(r"(?:at |around |about )?(\d{1,2}|" + "|".join(_HOUR_WORDS) + r")(?: o'?clock| please)?", said) or (
                    len(said.split()) <= 8 and re.search(r"\b(?:do|make it|at|how about|what about|rather|instead)\b", said)        # "can we do 3 instead" (29 Oct sweep: asked the day again)
                    and re.search(r"\b(?:do|make it|at|how about|what about) (\d{1,2}|" + "|".join(_HOUR_WORDS) + r")\b(?! (?:days?|weeks?|people|seats?|cars?))", said))
                if bare:
                    h = int(bare.group(1)) if bare.group(1).isdigit() else _HOUR_WORDS[bare.group(1)]
                    from datetime import time as _time
                    tm = _time(h + 12 if h < 9 else h, 0)                 # a bare "2" in opening hours (9 AM - 8 PM) is 2 PM
            if tm is not None and day is None and (s.slot_day or s.offered_slots or s.draft_slot):   # "2 PM" right after "tomorrow I can do 9 AM or 2 PM"; "make it 9" at the read-back: that day
                day = s.slot_day or (s.offered_slots[0].date() if s.offered_slots else s.draft_slot.date())
            if day and tm:
                chosen = combine(day, tm, tz)
                reason = self.svc.slots.invalid_reason(car.branch_id, chosen)
                free = self.svc.slots.free_slots(car, day) if reason is None else []
                if reason or chosen not in free:
                    alts = list(self.svc.slots.alternatives(car, chosen))[:2]
                    s.draft_slot, s.state = None, State.SLOT                     # waiting for another time: the next answer is a time
                    if s.booking is not None and s.booking.start == chosen:          # the caller's own test drive from earlier in this call
                        return {"core": f"You already have a test drive booked then, for the {self._booked_car(s)}. How about {self._slot_text(alts, now)}?",
                                "risk": "high", "awaiting": True, "facts": ["The customer already booked another test drive at that time in this call.",
                                                                            f"Nearest free: {self._slot_text(alts, now)}."]}
                    why = {"branch_closed": "We're closed then.", "too_soon": "That's a bit too soon.", "too_far_ahead": "That's too far ahead.",
                           "outside_hours_or_off_grid": "We book on the half hour, during opening hours."}.get(reason or "", "That slot isn't free.")
                    return {"core": f"{why} How about {self._slot_text(alts, now)}?", "risk": "high", "awaiting": True,
                            "facts": [f"The requested time is not available: {why}", f"Nearest free: {self._slot_text(alts, now)}."]}
                offered = list(s.offered_slots)
                confirmed = s.pending is not None and s.pending.kind == "time_check" and s.pending.value == chosen
                if offered and chosen not in offered and not confirmed:
                    s.pending, s.state = Pending("time_check", chosen), State.SLOT
                    choices = " or ".join(T.clock(x) for x in offered)
                    return {"core": f"I'd offered {choices}. Just to check, did you say {T.clock(chosen)}?", "risk": "high", "awaiting": True,
                            "facts": [f"Heard {T.clock(chosen)}, which was not one of the offered times ({choices}). It is free, but check first."]}
                if delegated and not confirmed:                                  # "any day would work" is about the day, not a choice between the two times: ask before taking one
                    s.pending, s.state = Pending("time_check", chosen), State.SLOT
                    return {"core": f"Then shall we say {T.when_short(chosen, now)}?", "risk": "high", "awaiting": True,
                            "facts": [f"The customer left the day to us after two times were offered. Propose {T.when_short(chosen, now)} and wait for a yes."]}
                s.draft_slot = chosen
                s.day_asks = 0
                slot_set_now = True
                s.offered_slots, s.slot_day, s.pending = [], None, None
            elif day and not tm:
                free = list(self.svc.slots.free_slots(car, day))
                pick = [free[0], free[len(free) // 2]] if len(free) > 1 else free[:1]
                if pick:
                    s.offered_slots, s.slot_day, s.state = pick, day, State.SLOT
                    times = " or ".join(T.clock_part(x) for x in pick)
                    s.day_asks = 0
                    return {"core": f"{W.checking(s)} {T.cap(T.day_name(day, now))} I can do {times}. Which suits you better?", "risk": "high", "awaiting": True,
                            "facts": [f"The customer gave a day but no time. Free on {day:%A}: {times} (and others)."]}
                return self._offer_next_slots(s, car, when, intro=f"There's nothing free {T.day_name(day, now)}.")      # "and today" after closing: say so, offer what there is
            elif _FLEXIBLE.search(" ".join(ex.tokens_of(when))):
                return self._offer_next_slots(s, car, when)
            else:
                problems.append("the day and time were not understood")
        if args.get("heard_digits") and not s.draft_phone and s.draft_slot:       # 28 Oct: "050, 0790, 054, 056" for 0507932546 was dropped and the agent asked "Anything else you'd like to know?"
            s.state = State.CONTACT
            return {"core": "I didn't catch the whole number. Could you say it again, a few digits at a time?", "risk": "high", "awaiting": True,
                    "facts": ["The mobile number was not heard completely (it is not a valid UAE number). Ask the customer to say it again, a few digits at a time."]}
        missing = [x for x, v in (("day and time", s.draft_slot), ("name", s.draft_name), ("mobile number", s.draft_phone)) if not v]
        s.state = State.SLOT if not s.draft_slot else State.CONTACT
        if missing and not s.draft_slot and not s.draft_name and not s.draft_phone:
            if s.day_asks >= 1:                                          # the day was asked for last turn and not given: offer times, do not ask again
                return self._offer_next_slots(s, car, when)
            s.day_asks += 1
            return {"core": W.ask_day(s, car), "risk": "high", "awaiting": True,
                    "facts": [f"The customer wants a test drive of the {T.car_name(car)}. Nothing is booked yet.", *fact_sheet([car], self._branch_name).splitlines(),
                              "Ask which day and time suit them. Do not say anything is booked or confirmed, and do not offer times yourself.",
                              "Do not read the car's details out again; the customer knows the car."]}
        if missing:
            need = " and ".join(missing)
            known = [f"{k}: {v}" for k, v in (("when", T.when_full(s.draft_slot) if s.draft_slot else None), ("name", s.draft_name),
                                              ("phone", speak_phone(s.draft_phone) if s.draft_phone else None)) if v]
            said = T.when_short(s.draft_slot, now).replace(":00 ", " ") if slot_set_now and s.draft_slot else ""
            return {"core": W.ask_missing(s, missing, s.draft_name, said), "risk": "high", "awaiting": True,
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
        line = W.read_back(car, s.draft_slot, self._branch_name(car), s.draft_name, speak_phone(s.draft_phone))
        if s.replacing is not None and s.replacing.stock_id == car.stock_id:
            line = (f"Let me confirm: I'll move your test drive of the {T.car_name(car)} from {T.when_short(s.replacing.start, now)} "
                    f"to {T.when_full(s.draft_slot)}. Is that right?")
        return {"core": line, "risk": "high", "awaiting": True,
                "facts": ["READ-BACK: say all of these details back to the customer and ask if they are right:", f"car: {T.car_name(car)}",
                          f"when: {T.when_full(s.draft_slot)}", f"where: {self._branch_name(car)}", f"name: {s.draft_name}", f"phone: {speak_phone(s.draft_phone)}"]}

    def _t_confirm_booking(self, s, args, text, dbg) -> dict:
        car = self.svc.get_car(s.car) if s.car else None
        if not (s.read_back and car and s.draft_slot and s.draft_name and s.draft_phone):
            if s.booking is not None:                                    # already booked in this call: say so, not "no booking yet"
                return {"core": f"You're all set for {T.when_short(s.booking.start, self.svc.clock())}. Is there anything else I can help with?",
                        "facts": ["The test drive is already booked; nothing new to confirm."], "awaiting": False}
            return {"core": "I don't have a booking to confirm yet. Which car and time would you like?", "facts": ["There is no read-back booking to confirm."], "awaiting": True}
        old = s.replacing if s.replacing is not None and s.replacing.stock_id == car.stock_id else None
        if old is not None and old.booking_id:                # a move: free the old slot on this server first (the duplicate check would refuse the new one)
            self.svc.cancel_booking(old.booking_id)
            dbg.tools.append(f"cancel_booking({old.booking_id}) for the move")
        out = self.svc.book_test_drive(car.stock_id, s.draft_name, s.draft_phone, s.draft_slot)
        dbg.tools.append(f"book_test_drive({car.stock_id}, {s.draft_slot:%a %H:%M}) -> {out.kind.value}")
        now = self.svc.clock()
        s.read_back = False
        if out.kind is not BookingKind.BOOKED and old is not None and old.booking_id:       # the new slot failed: put the old one back
            back = self.svc.book_test_drive(car.stock_id, old.name or s.draft_name, old.phone or s.draft_phone, old.start)
            if back.kind is BookingKind.BOOKED:
                old.booking_id = back.booking.booking_id
        if out.kind is BookingKind.BOOKED:
            s.booking, s.state = out.booking, State.ANSWERING          # booked: back to talking about cars (questions, another booking)
            s.draft_slot = None
            s.known_bookings.append(KnownBooking(car.stock_id, out.booking.start, s.draft_name or "", s.draft_phone or "", booking_id=out.booking.booking_id))
            if old is not None:
                self._forget(s, old)
                s.replacing = None
                return {"core": f"Done. Your test drive of the {T.car_name(car)} is now {T.when_short(out.booking.start, now)}. {W.anything_else(s)}", "risk": "high", "awaiting": False,
                        "facts": [f"MOVED: {T.car_name(car)} from {T.when_full(old.start)} to {T.when_full(out.booking.start)}."]}
            lic = " Please bring a valid driving licence." if self.svc.cfg.test_drive.licence_required else ""
            return {"core": W.booked(s, T.when_short(out.booking.start, now).replace(":00 ", " "), bool(lic)), "risk": "high", "awaiting": False,
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

    def _stock_claims(self, sentence: str) -> list[str]:
        """A model sentence saying a make or model is not in stock, when it is (26 Oct: "we don't have any BMWs" with four BMWs on sale)."""
        low = sentence.lower()
        if not _NONE_CLAIM.search(low):
            return []
        toks = ex.tokens_of(sentence)
        found = ex.find_vehicle(toks, self.svc.inventory.vocabulary()) or {}
        if not found.get("make"):                                       # "BMWs", "Toyotas"
            found = ex.find_vehicle([t[:-1] if t.endswith("s") and len(t) > 3 else t for t in toks], self.svc.inventory.vocabulary()) or {}
        if found.get("make") and self.svc.inventory.count(SearchFilters(make=found["make"], model=found.get("model"))) > 0:
            return [f"says there are no {found.get('model') or found['make']} cars, but there are"]
        return []

    def _offer_on(self, s: Session, car, words: str) -> bool:
        """An offer below this car's price: in so many words, or just the amount ("come on, 50,000?") once the caller has already pushed on it."""
        car = self.svc.get_car(car) if isinstance(car, str) else car
        if not car:
            return False
        if offers_less(words, car.price):
            return True
        asked = ex.asked_price(words, car.price)
        return bool(s.haggles.get(car.stock_id, {}).get("pushes") and asked and asked < car.price and len(words.split()) <= 6)

    @staticmethod
    def _filters_said(f: dict) -> str:
        """'under 62,499 dirhams', 'from 2021', 'white': the filters from earlier in the call, said the way a person would."""
        parts = []
        if f.get("make"):
            make = str(f["make"])
            parts.append(f"as {'an' if make[:1] in 'aeiou' else 'a'} {make.title() if len(make) > 3 else make.upper()}")
        if f.get("max_price"):
            parts.append(f"under {T.money(int(f['max_price']) + 1 if int(f['max_price']) % 500 == 499 else int(f['max_price']))}")
        if f.get("year_min"):
            parts.append(f"from {f['year_min']} on")
        if f.get("max_km"):
            parts.append(f"under {int(f['max_km']):,} kilometres")
        if f.get("color"):
            parts.append(f"in {f['color']}")
        if f.get("body_type"):
            parts.append(f"as {'an' if str(f['body_type'])[:1] in 'aeiou' else 'a'} {str(f['body_type']).upper() if f['body_type'] == 'suv' else f['body_type']}")
        return " and ".join(parts)

    def _which_car_to_drive(self, s: Session) -> str:
        """Which car, naming the ones just discussed, so the caller can simply say one."""
        names = [T.short_name(c) for c in s.shown[:3]]
        if not names:
            return "Which car would you like to test drive?"
        listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " or the " + names[-1]
        return f"Which car would you like to test drive: the {listed}?"

    def _t_which_car(self, s, args, text, dbg) -> dict:
        names = [T.short_name(c) for c in s.shown[:3]]
        listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " or the " + names[-1]
        return {"core": f"Which one: the {listed}?", "facts": [f"The customer said yes but has not said which car: {listed}."], "awaiting": True}

    def _t_fix_booking(self, s, args, text, dbg) -> dict:
        """'No' at the read-back: keep what was right, take the corrected detail if it was said, and read back again (or ask what to change)."""
        car = self.svc.get_car(s.car) if s.car else None
        ct = ex.contact(text)
        tz = self.svc.slots.tz(self.svc.cfg.branches[0].id)
        when = parse_when(text, self.svc.clock(), tz)
        named = bool(re.search(r"\b(?:name|it's|i'm|call me)\b", " ".join(ex.tokens_of(text)))) and ct.name
        if car is None or not (named or ct.phone or when.found):
            s.read_back, s.state = False, State.CONTACT
            return {"core": "Sorry about that. What should I change: the day and time, your name, or your number?", "risk": "high", "awaiting": True,
                    "facts": ["The customer said the read-back was wrong. Nothing is booked. Keep the draft."]}
        if named:
            s.draft_name = names.correct(ct.name)
        if ct.phone:
            s.draft_phone = ct.phone
        s.read_back = False
        out = self._t_prepare_booking(s, {"car_id": car.stock_id, "when": text if when.found else ""}, text, dbg)
        if out["core"].startswith("So that's"):                       # the corrected read-back: said as a correction, not the same lines again
            out["core"] = "Sorry about that. " + out["core"].replace("Shall I book it?", "Is that all correct now?")
        return out

    def _t_cancel_draft(self, s, args, text, dbg) -> dict:
        s.draft_slot = s.draft_name = s.draft_phone = None
        s.day_asks = 0
        s.read_back = False
        if _NEVER_MIND.search(" ".join(ex.tokens_of(text))):
            s.state = State.ANSWERING
            return {"core": f"No problem, I won't book anything. {W.anything_else(s)}", "facts": ["The booking was dropped; nothing is booked."],
                    "awaiting": False}
        return {"core": "No problem. What would you like to change?", "facts": ["The booking draft was cleared; nothing is booked."], "awaiting": True}

    # ---- bookings the caller already has --------------------------------------------------------------------------------------
    def _forget(self, s: Session, kb: KnownBooking) -> None:
        """A booking is gone (cancelled or moved): off the list, and onto the list the server tells the page about."""
        s.known_bookings = [k for k in s.known_bookings if k is not kb]
        s.cancelled.append(kb)
        if s.booking is not None and kb.booking_id and s.booking.booking_id == kb.booking_id:
            s.booking = None

    def _booking_text(self, kb: KnownBooking) -> str:
        car = self.svc.get_car(kb.stock_id)
        return f"the {T.car_name(car) if car else 'car'} {T.when_short(kb.start, self.svc.clock())}"

    def _which_booking(self, s: Session, text: str) -> KnownBooking | None:
        """The booking the caller means: the only one, or the one whose car or day they named."""
        if len(s.known_bookings) == 1:
            return s.known_bookings[0]
        words = " ".join(ex.tokens_of(text))
        hits = []
        for kb in s.known_bookings:
            car = self.svc.get_car(kb.stock_id)
            names = {car.make.lower(), car.model.lower(), *car.model.lower().split()} if car else set()
            if any(re.search(rf"\b{re.escape(n)}\b", words) for n in names if len(n) > 1) or kb.start.strftime("%A").lower() in words:
                hits.append(kb)
        return hits[0] if len(hits) == 1 else None

    def _t_my_bookings(self, s, args, text, dbg) -> dict:
        if not s.known_bookings:
            return {"core": "I don't see any test drives booked yet. Would you like to book one?", "facts": ["The customer has no test drives booked."], "awaiting": True}
        items = [self._booking_text(kb) for kb in sorted(s.known_bookings, key=lambda k: k.start)]
        listed = items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]
        lead = "a test drive of" if len(items) == 1 else f"{len(items)} test drives:"
        return {"core": f"You have {lead} {listed}. Is there anything else I can help with?", "facts": [f"The customer's test drives: {listed}."], "awaiting": False}

    def _t_cancel_booking(self, s, args, text, dbg) -> dict:
        words = " ".join(ex.tokens_of(text))
        if s.pending_cancel is not None:
            kb, s.pending_cancel = s.pending_cancel, None
            if _PLAIN_YES.fullmatch(words) or words.startswith("yes"):
                if kb.booking_id:
                    out = self.svc.cancel_booking(kb.booking_id)
                    dbg.tools.append(f"cancel_booking({kb.booking_id}) -> {out.kind.value}")
                said = self._booking_text(kb)
                self._forget(s, kb)
                return {"core": f"Done. Your test drive of {said} is cancelled. Is there anything else I can help with?", "risk": "high",
                        "awaiting": False, "facts": ["CANCELLED: " + said]}
            return {"core": "No problem, I've kept it. Is there anything else I can help with?", "facts": ["The booking was kept."], "awaiting": False}
        if not s.known_bookings:
            return {"core": "I don't see any test drives booked to cancel. Would you like to book one?", "facts": ["Nothing to cancel."], "awaiting": True}
        kb = self._which_booking(s, text)
        if kb is None:
            s.pending = Pending("cancel_which")
            options = " or ".join(self._booking_text(k) for k in s.known_bookings[:3])
            return {"core": f"Which one would you like to cancel: {options}?", "facts": ["The customer has several test drives."], "awaiting": True}
        s.pending, s.pending_cancel = None, kb
        return {"core": f"Shall I cancel your test drive of {self._booking_text(kb)}?", "risk": "high", "awaiting": True,
                "facts": [f"Asking before cancelling: {self._booking_text(kb)}. Nothing is cancelled yet."]}

    def _t_reschedule_booking(self, s, args, text, dbg) -> dict:
        if not s.known_bookings:
            return {"core": "I don't see any test drives booked to move. Would you like to book one?", "facts": ["Nothing to move."], "awaiting": True}
        kb = self._which_booking(s, text)
        if kb is None:
            s.pending = Pending("move_which")
            options = " or ".join(self._booking_text(k) for k in s.known_bookings[:3])
            return {"core": f"Which one would you like to move: {options}?", "facts": ["The customer has several test drives."], "awaiting": True}
        s.pending, s.replacing, s.car = None, kb, kb.stock_id
        s.draft_slot, s.read_back = None, False
        s.draft_name = kb.name or s.draft_name
        s.draft_phone = (normalize_phone(kb.phone) or s.draft_phone) if kb.phone else s.draft_phone
        s.state = State.SLOT
        tz = self.svc.slots.tz(self.svc.cfg.branches[0].id)
        if parse_when(text, self.svc.clock(), tz).found:                  # "move it to Sunday at ten": straight to the new time
            return self._t_prepare_booking(s, {"car_id": kb.stock_id, "when": text}, text, dbg)
        return {"core": f"Sure. Which day and time would suit you instead of {T.when_short(kb.start, self.svc.clock())}?", "risk": "high", "awaiting": True,
                "facts": [f"Moving the test drive of {self._booking_text(kb)}. Ask for the new day and time. Nothing has changed yet."]}

    # ---- answers the code writes from the site's data --------------------------------------------------------------------------
    def _t_colours(self, s, args, text, dbg) -> dict:
        """The colours, from the stock data. About the cars just listed when the caller points at them ("these", "they") or nothing else
        is in play; otherwise across the stock, or one make / body type if named ("what colours do the Toyotas come in?")."""
        joined = " ".join(ex.tokens_of(text))
        f = {k: v for k, v in ex.search_filters(text, self.svc.inventory.vocabulary()).items() if k in ("make", "model", "body_type")}
        if len(s.shown) >= 2 and not f and (_PLURAL_REF.search(joined) or not (s.car or _STOCK_REF.search(joined))):
            cars = s.shown[:4]
            parts = [f"the {T.short_name(c)} is {c.color}" for c in cars]
            line = (", ".join(parts[:-1]) + " and " + parts[-1]) + ("" if len(s.shown) <= 4 else f", plus {len(s.shown) - 4} more")
            line = line[:1].upper() + line[1:] + "."
            return {"core": f"{line} Which one would you like to hear more about?", "facts": [line], "awaiting": True}
        stock = self.svc.inventory.search(SearchFilters(**f), 500)
        counts: dict[str, int] = {}
        for c in stock:
            counts[c.color] = counts.get(c.color, 0) + 1
        colours = sorted(counts, key=lambda k: (-counts[k], k))
        what = W.what_they_asked(stock, f) if f else "cars"
        if not colours:
            line = f"I don't have any {what} in stock right now."
            return {"core": f"{line} Would you like to look at another make or type?", "facts": [line], "awaiting": True}
        if len(colours) == 1:
            line = f"The {what} I have are all {colours[0]}."
        elif len(colours) <= 4:
            line = f"The {what} come in {', '.join(colours[:-1])} and {colours[-1]}."
        else:
            more = len(colours) - 5
            line = f"Our {what} come in {', '.join(colours[:4])} and {colours[4]}" + (f", plus {more} other colour{'s' if more > 1 else ''}." if more > 0 else ".")
        return {"core": f"{line} Is there a colour you'd like?", "facts": [line], "awaiting": True}

    def _t_showroom_info(self, s, args, text, dbg) -> dict:
        site = self.svc.showroom()
        lines = A.showroom_answer(site, " ".join(ex.tokens_of(text))) or [f"Our showroom is on {site.location or 'Sheikh Zayed Road'}."]
        return {"core": " ".join(lines) + " " + W.anything_else(s), "facts": lines, "awaiting": False}

    def _t_repeat(self, s, args, text, dbg) -> dict:
        last = s.last_agent()
        return {"core": last or "Sorry, could you say that again?", "facts": ["The customer asked to hear the last reply again."], "awaiting": True}

    def _cars_named(self, s: Session, text: str) -> list[CarView]:
        """The cars a comparison is about: named models, positions in the last list, or "both" / "them" (the last two shown)."""
        words = " ".join(ex.tokens_of(text))
        out: list[CarView] = []
        for i, w in enumerate(("first", "second", "third")):
            if re.search(rf"\b{w}\b", words) and i < len(s.shown):
                out.append(s.shown[i])
        for make, models in self.svc.inventory.vocabulary().items():
            for m in sorted(models, key=len, reverse=True):
                said = " ".join(ex.tokens_of(m))                        # "rav4" is heard and normalised as "rav four"
                if re.search(rf"\b(?:{re.escape(m)}|{re.escape(said)})\b", words):
                    pool = [c for c in s.shown if c.model.lower() == m] or [self.svc.get_car(c.stock_id) for c in self.svc.inventory.search(SearchFilters(make=make, model=m), 1)]
                    car = pool[0] if pool else None
                    if car is not None and all(c.stock_id != car.stock_id for c in out):
                        out.append(car)
        current = self.svc.get_car(s.car) if s.car else None
        if len(out) == 1 and current is not None and current.stock_id != out[0].stock_id:
            out.insert(0, current)
        if len(out) < 2 and re.search(r"\b(?:both|them|these|those|the two|first two)\b", words) and len(s.shown) >= 2:
            out = list(s.shown[:2])
        return out[:2]

    def _t_compare_cars(self, s, args, text, dbg) -> dict:
        cars = self._cars_named(s, text)
        if len(cars) < 2:
            return {"core": "Which two cars would you like me to compare?", "facts": ["Two cars are needed for a comparison."], "awaiting": True}
        a, b = cars
        lines = A.compare(a, self.svc.details(a.stock_id), b, self.svc.details(b.stock_id))
        s.shown, s.car, s.state = [a, b], None, State.ANSWERING
        dbg.tools.append(f"compare({a.stock_id}, {b.stock_id})")
        return {"core": " ".join(lines) + " Which one would you like to hear more about?", "awaiting": True,
                "facts": ["Prices are fixed (no negotiation).", *fact_sheet([a, b], self._branch_name).splitlines()]}

    def _t_handoff(self, s, args, text, dbg) -> dict:
        h = self.svc.escalate_to_human(EscalationReason.REQUESTED_HUMAN, s.car, None, note="agent: customer asked for a person or was upset")
        s.handoff, s.ended, s.state = h, True, State.GOODBYE
        dbg.tools.append(f"escalate_to_human -> {h.handoff_id}")
        return {"core": "Of course. I'm connecting you with a colleague now.", "facts": ["The call is being handed to a colleague."], "risk": "high"}

    def _t_end_call(self, s, args, text, dbg) -> dict:
        s.ended, s.state = True, State.GOODBYE
        return {"core": "Thank you for calling AGMC. Goodbye!", "facts": ["The customer is finished; say a short goodbye."], "awaiting": False}
