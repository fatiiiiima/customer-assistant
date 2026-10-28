"""The deployment must send the agent's tool decisions to the model server it sends the replies to (docs/ARCH_REVIEW.md section 19).

`--backend` on the worker command selects only the reply model; `dialogue.decision_backend` is separate and defaults to the Intel-GPU model, which a
GPU-server pod does not have. The three deployment files each set it in the environment; these tests load the configuration the way the worker does
(the website overlay plus that environment, not the laptop's .env) and check it names the served model."""
import re
from pathlib import Path

import pytest
import yaml

from dealer_core.config import load_config

ROOT = Path(__file__).resolve().parents[2]
DEPLOY = ROOT / "deploy"


def k8s_worker() -> tuple[dict[str, str], str]:
    docs = list(yaml.safe_load_all((DEPLOY / "k8s" / "voice.yaml").read_text(encoding="utf-8")))
    c = next(d for d in docs if d and d.get("kind") == "Deployment" and d["metadata"]["name"] == "voice-worker")["spec"]["template"]["spec"]["containers"][0]
    return {e["name"]: str(e["value"]) for e in c["env"] if "value" in e}, " ".join(c["args"])


def compose_worker() -> tuple[dict[str, str], str]:
    w = yaml.safe_load((DEPLOY / "docker-compose.yml").read_text(encoding="utf-8"))["services"]["worker"]
    return {k: str(v) for k, v in w["environment"].items()}, " ".join(w["command"])


def dockerfile_worker() -> tuple[dict[str, str], str]:
    text = (DEPLOY / "Dockerfile").read_text(encoding="utf-8")
    env = dict(re.findall(r"^ENV\s+(?:\w+=\S+\s+)*?(VA__\w+)=(\S+)", text, flags=re.M))
    return env, re.search(r"^CMD\s+(.*)$", text, flags=re.M).group(1).replace('"', " ").replace(",", " ")


@pytest.mark.parametrize("source", [k8s_worker, compose_worker, dockerfile_worker], ids=["k8s", "compose", "dockerfile"])
def test_decisions_go_to_the_served_model(source):
    env, command = source()
    cfg = load_config(overlay=ROOT / "config" / "portal.yaml", environ=env)       # environ given: the laptop's .env is not read
    served = re.search(r"--backend\s+(\w+)", command).group(1)
    assert served == "gpu_server"
    assert cfg.dialogue.decision_backend == served
    assert cfg.llm.chain(cfg.dialogue.decision_backend)[0] == "gpu_server"        # the first backend tried for a decision is the server, not the iGPU model


def test_the_laptop_profile_is_unchanged():
    """The website overlay stays the laptop's: its decisions still go to the Intel-GPU model unless the environment says otherwise."""
    cfg = load_config(overlay=ROOT / "config" / "portal.yaml", environ={})
    assert cfg.dialogue.decision_backend == "openvino_gpu"
