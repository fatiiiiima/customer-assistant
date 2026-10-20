"""Probe which OpenVINO devices exist on this laptop and what they report.

Does not assume acceleration helps: it only lists devices. Benchmarks (bench_openvino.py)
decide whether CPU / iGPU / NPU is actually faster for each workload.
"""
import json

import openvino as ov

core = ov.Core()
out = {"openvino": ov.__version__, "devices": {}}
for d in core.available_devices:
    props = {}
    for key in ("FULL_DEVICE_NAME", "DEVICE_TYPE", "OPTIMIZATION_CAPABILITIES", "DEVICE_ARCHITECTURE"):
        try:
            props[key] = str(core.get_property(d, key))
        except Exception as e:  # not every device reports every property
            props[key] = f"n/a ({type(e).__name__})"
    out["devices"][d] = props
print(json.dumps(out, indent=2))
