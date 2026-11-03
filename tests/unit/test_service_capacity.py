"""scripts/service_capacity.py (ARCH_REVIEW section 101): how much one copy of a shared service carries. Against fake local servers: nothing leaves the machine."""
import importlib.util
import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("service_capacity", ROOT / "scripts" / "service_capacity.py")
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)


def test_a_level_holds_n_clients_asking_again_at_once_and_counts_what_was_answered():
    def once(i, k):
        time.sleep(0.05)
        return {"first": 0.01, "total": 0.05}

    r = sc.run_level(2, 0.6, once)
    assert 16 <= r["answered"] <= 26 and r["errors"] == 0 and 28 <= r["per_second"] <= 45
    assert r["first_p50"] == 0.01 and r["total_p90"] == 0.05


def test_errors_are_counted_not_hidden_and_a_level_that_only_fails_stops_early():
    n = {"k": 0}

    def flaky(i, k):
        n["k"] += 1
        if n["k"] % 2:
            raise RuntimeError("HTTP 429")
        return {"first": 0.01, "total": 0.02}

    r = sc.run_level(1, 0.8, flaky)
    assert r["errors"] > 0 and r["answered"] > 0 and r["error_kinds"] == ["RuntimeError: HTTP 429"]

    def never(i, k):
        raise RuntimeError("refused")

    t = time.monotonic()
    r = sc.run_level(2, 30.0, never)
    assert time.monotonic() - t < 10 and r["answered"] == 0 and r["errors"] > 8              # it did not hammer a service that refuses everything for 30 s


def test_the_callers_one_copy_carries_use_the_busiest_clean_level_at_70_percent():
    rows = [{"concurrency": 1, "answered": 20, "errors": 0, "per_second": 5.0}, {"concurrency": 4, "answered": 40, "errors": 0, "per_second": 10.0},
            {"concurrency": 16, "answered": 50, "errors": 9, "per_second": 30.0}]                          # the level with errors is not counted
    c = sc.carried(rows, per_turn=3.0, turn_every=15.0)
    assert c["saturated_per_second"] == 10.0 and c["at_concurrency"] == 4 and c["planning_per_second"] == 7.0 and c["callers"] == 35
    assert sc.carried([{"concurrency": 1, "answered": 0, "errors": 5, "per_second": 0}], 3, 15)["callers"] is None
    assert sc.carried(rows, per_turn=1.0, turn_every=15.0)["callers"] == 105                                 # one request a turn: three times the callers


def test_the_voice_server_is_counted_in_seconds_of_speech_not_in_sentences():
    rows = [{"concurrency": 1, "answered": 30, "errors": 0, "per_second": 2.26, "audio_seconds_per_second": 6.97}]
    c = sc.carried(rows, per_turn=3.0, turn_every=15.0, audio_per_turn=6.0)
    assert c["callers"] == int(6.97 * 0.7 * 15 / 6.0) == 12 and c["callers_if_sentences_are_these_long"] == 7 and c["audio_seconds_per_turn"] == 6.0
    assert sc.carried(rows, per_turn=3.0, turn_every=15.0)["callers"] == 7                         # without it, as before


class Fake(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    lock = threading.Lock()

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        self.rfile.read(n)
        if self.path.endswith("/audio/speech"):
            with Fake.lock:                                        # one sentence at a time, as the voice server does
                time.sleep(0.1)
                body = b"\x01\x00" * 24000                         # one second of audio at 24 kHz
        else:
            time.sleep(0.02)
            body = json.dumps({"text": "hello"}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}/v1"


def test_a_serial_voice_server_keeps_the_same_rate_however_many_ask_and_the_wait_grows():
    httpd, url = serve()
    try:
        once = sc.voice_once(url, "pocket", "eve", 24000)
        one, four = sc.run_level(1, 1.0, once), sc.run_level(4, 1.5, once)
    finally:
        httpd.shutdown()
    assert 7 <= one["per_second"] <= 11 and 7 <= four["per_second"] <= 11                    # 0.1 s a sentence, one at a time
    assert four["first_p50"] > 2.5 * one["first_p50"]                                         # four waiting: about four times the wait
    assert one["audio_seconds_per_second"] > 6                                                # one second of audio per answer


def test_the_recogniser_test_sends_the_clip_the_model_and_the_hot_words_the_workers_way():
    seen = []

    class Seen(Fake):
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            seen.append(self.rfile.read(n))
            body = json.dumps({"text": "ok"}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Seen)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        once = sc.recogniser_once(f"http://127.0.0.1:{httpd.server_address[1]}/v1", "qwen3-asr", [b"RIFFfakewav"], "Toyota, Camry")
        assert once(0, 0)["first"] > 0
    finally:
        httpd.shutdown()
    body = seen[0]
    assert b'name="model"' in body and b"qwen3-asr" in body and b'name="hotwords"' in body and b"Toyota, Camry" in body and b"RIFFfakewav" in body


def test_the_model_test_reads_the_first_text_and_the_whole_reply_and_refuses_an_empty_one():
    class Backend:
        def lead_in_stream(self, system, user, max_tokens, temperature, deadline):
            time.sleep(0.03)
            yield "Here is "
            time.sleep(0.02)
            yield "a reply."

    prompts = [{"system": "S", "user": "U", "max_tokens": 50}]
    rec = sc.model_once(Backend(), prompts)(0, 0)
    assert 0.025 <= rec["first"] <= 0.2 and rec["total"] >= rec["first"] + 0.015 and rec["chars"] == len("Here is a reply.")

    class Empty:
        def lead_in_stream(self, *a):
            return iter(())

    try:
        sc.model_once(Empty(), prompts)(0, 0)
        raise AssertionError("an empty reply was accepted")
    except RuntimeError as e:
        assert "wrote nothing" in str(e)


def test_the_model_dry_run_says_what_it_would_send_and_sends_nothing():
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "service_capacity.py"), "model", "--dry-run", "--levels", "1", "2", "--seconds", "10"],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and "COSTS MONEY" in out.stdout and "million input tokens" in out.stdout


def test_the_hot_word_list_comes_from_the_stock_database():
    db = ROOT / "data" / "stock.db"
    if not db.exists():
        import pytest
        pytest.skip("no local stock database")
    names = sc.stock_names().split(", ")
    assert "Porsche" in names and "CR-V" in names and len(names) >= 90
