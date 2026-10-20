# Demo guide (text chat, fake data)

**What this is:** the real dialogue engine, tools and policy engine, driven by typing. The language model only chooses the *intent*;
every reply is templated, every price comes from the policy engine, and every booking is read back before it is made. All cars,
prices, customers and branches are **fake**. It is **text only**: voice (STT/TTS) is a later phase.

**Model backend for the demo: `openvino_gpu`** = Qwen3-4B-Instruct-2507 (our int4 OpenVINO conversion) on the Intel Arc iGPU.
Fallback if the iGPU misbehaves: `--backend ollama` (same checkpoint on the CPU; slower, about 2-3 s per model turn).

## Before the demo (2 minutes)
1. **Plug in AC** and set Windows power mode to **Best performance** (battery mode cut sustained speed to ~30%).
2. **Stop anything heavy that is not the demo.** Other benchmark or conversion jobs steal CPU and shared power from the iGPU and
   make model turns jittery (measured: see "What to expect"). The pre-flight check lists them.
3. Double-click **`demo_check.cmd`** (or run `.venv\Scripts\python.exe scripts\demo_preflight.py`). You want `READY`.
4. Close the labeling page if it is still running (harmless, but it is one more process): see the stop command at the bottom.

## Run it (from `C:\dev\voice-agent`)
| What | Command |
|---|---|
| **Scripted demo** (the 5 scenarios, paced, with the live trace) | double-click `demo_replay.cmd`, or `.venv\Scripts\python.exe -m dealer_core.cli.replay --backend openvino_gpu --pause 1.5 --debug` |
| One scenario only (for example the price haggle) | `demo_replay.cmd --only 3` |
| **Interactive chat** with the live trace | double-click `demo_chat.cmd`, or `.venv\Scripts\python.exe -m dealer_core.cli.chat --backend openvino_gpu --debug` |
| Reliability check, 20 runs of every scenario, no transcripts | `.venv\Scripts\python.exe -m dealer_core.cli.replay --backend openvino_gpu --runs 20 --quiet` |
| Fallback backend | add `--backend ollama` (replace `openvino_gpu`) |

Start-up takes about 10-13 seconds (it loads the model and warms every state so the first customer turn is not slow). In the
interactive chat, `/debug` toggles the trace, `/reset` starts a new call, `/state` shows the session, `/quit` leaves.

## What the trace shows (`--debug`)
```
  | state   : answering -> negotiating            the conversation state before and after this turn
  | intent  : make_offer   handled by LLM (openvino_gpu)   NLU 946.8 ms     what the caller meant, and WHO decided: RULES or the LLM
  | tool    : propose_price(S1042, 125000) -> counter 146100               the backend tool called, with its result
  | policy  : counter amount=146100 round=2 rounds_left=1                  the policy engine's decision (no LLM involved)
  | time    : 947.8 ms total
```

## The five scenarios (the script in `scenarios/demo.yaml`)
1. **Search for an SUV within a budget.** "I'm looking for an SUV under 150 thousand dirhams" -> the agent counts 6 matches and asks a
   narrowing question instead of reading them all ("Toyota please" -> reads two). Answered by rules: instant.
2. **Ask about a car.** Kilometres, colour, GCC spec, service history; and "has it had any accidents?" gets an honest "I don't have
   accident records", not an invention. Instant.
3. **Haggle below the floor.** Each offer gets a small, stepped counter, always "subject to confirmation", never the limit; the
   fourth, still-too-low offer hands the call to a human. The floor price is never spoken (it is not even visible to the layer
   that writes the reply).
4. **Book a test drive with read-back.** Slot, name and phone are read back in full before anything is booked; the car's
   availability is re-checked at that moment; the confirmation goes to WhatsApp (mocked).
5. **Off-topic, then angry -> human.** A polite redirect, then an immediate hand-off.

## What to tell your boss (honestly)
* The model **only picks the intent** (about six tokens of output). Everything the agent says is written by templates; every number is
  from the database or the policy engine. That is why it cannot invent a price, a discount or a fact.
* State-changing steps (price, booking, cancel, hand-off) deliberately go through the model rather than keyword rules, because the
  rules are not yet proven accurate enough on real speech; that is the 0.7-1.5 s you see on those turns. Plain questions are instant.
* It is **text only** and uses **fake data**. Voice (speech in/out) is the next phase; the latency work for it is done.

## Known limits (do not demo these)
* **Policy questions** (warranty, financing, documents): there is no knowledge base yet (Phase 2). The agent says it does not have the
  confirmed details and offers a colleague. It does not make terms up.
* **Opening hours / "which day is best"**, **"I didn't say that" corrections**: no dedicated handling yet (known gaps).
* Arabic is detected and answered with "English only, shall I connect a colleague?". It cannot help in Arabic.
* Ad-libbing: scripted turns are verified; off-script phrases are covered by the unit tests but not guaranteed. If the agent misroutes
  one, type `/reset` and carry on with the script.
* Dates shown in the demo follow a frozen clock (the next Monday at 10:00) so the booking looks the same on any day.

## Stop the labeling page if it is still running
```powershell
taskkill /T /F /PID (Get-Content C:\dev\voice-agent\logs\labeler.pid)
```
