#!/usr/bin/env python3
"""
Lightweight hardware probe for BioVision.
Only imports torch and psutil — no dlib, ultralytics, or other heavy deps.
Outputs a single JSON line to stdout.
"""
import json
import sys


def probe() -> dict:
    device = "cpu"
    gpu_name = None

    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            device = "cuda"
            try:
                gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                gpu_name = "NVIDIA GPU"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
            gpu_name = "Apple Silicon MPS"
    except Exception:
        pass  # torch not installed — default to cpu

    ram_gb = None
    try:
        import psutil  # noqa: PLC0415

        ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
    except Exception:
        # psutil unavailable — try platform-specific fallbacks
        try:
            import subprocess  # noqa: PLC0415

            if sys.platform == "darwin":
                out = subprocess.check_output(
                    ["sysctl", "-n", "hw.memsize"], text=True, timeout=3
                )
                ram_gb = round(int(out.strip()) / (1024**3), 1)
            elif sys.platform.startswith("linux"):
                with open("/proc/meminfo", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            kb = int(line.split()[1])
                            ram_gb = round(kb / (1024**2), 1)
                            break
        except Exception:
            pass

    return {"device": device, "gpu_name": gpu_name, "ram_gb": ram_gb}


if __name__ == "__main__":
    result = probe()
    print(json.dumps(result))
