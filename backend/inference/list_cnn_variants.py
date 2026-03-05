"""
Report selectable CNN backbone variants for the current system.

Stdout JSON:
{
  "ok": true,
  "torch_available": bool,
  "torchvision_available": bool,
  "device": "cuda" | "mps" | "cpu",
  "gpu_name": str | null,
  "gpu_memory_gb": float | null,
  "default_variant": str | null,
  "variants": [
    {
      "id": "simplebaseline",
      "label": "SimpleBaseline (ResNet-50)",
      "description": "...",
      "selectable": bool,
      "recommended": bool,
      "reason": str | null
    }
  ]
}
"""

from __future__ import annotations

import json


def _base_variants():
    return [
        {
            "id": "simplebaseline",
            "ctor": "resnet50",
            "label": "SimpleBaseline (ResNet-50)",
            "description": "Balanced accuracy/speed baseline for most datasets.",
        },
        {
            "id": "mobilenet_v3_large",
            "ctor": "mobilenet_v3_large",
            "label": "MobileNetV3 Large",
            "description": "Fastest option; useful on CPU or lower-memory systems.",
        },
        {
            "id": "efficientnet_b0",
            "ctor": "efficientnet_b0",
            "label": "EfficientNet-B0",
            "description": "Compact backbone with strong generalization on medium datasets.",
        },
        {
            "id": "hrnet_w32",
            "ctor": "hrnet_w32",
            "label": "HRNet-W32",
            "description": "Highest-capacity option; best with stronger GPU resources.",
        },
    ]


def main():
    payload = {
        "ok": True,
        "torch_available": False,
        "torchvision_available": False,
        "device": "cpu",
        "gpu_name": None,
        "gpu_memory_gb": None,
        "default_variant": None,
        "variants": [],
    }

    variants = _base_variants()

    try:
        import torch  # type: ignore

        payload["torch_available"] = True
    except Exception:
        for v in variants:
            payload["variants"].append(
                {
                    "id": v["id"],
                    "label": v["label"],
                    "description": v["description"],
                    "selectable": False,
                    "recommended": False,
                    "reason": "PyTorch is not installed.",
                }
            )
        print(json.dumps(payload))
        return

    try:
        from torchvision import models as tv_models  # type: ignore

        payload["torchvision_available"] = True
    except Exception:
        for v in variants:
            payload["variants"].append(
                {
                    "id": v["id"],
                    "label": v["label"],
                    "description": v["description"],
                    "selectable": False,
                    "recommended": False,
                    "reason": "torchvision is not installed.",
                }
            )
        print(json.dumps(payload))
        return

    # Device probe.
    device = "cpu"
    gpu_name = None
    gpu_memory_gb = None
    try:
        if torch.cuda.is_available():
            device = "cuda"
            props = torch.cuda.get_device_properties(0)
            gpu_name = getattr(props, "name", None)
            gpu_memory_gb = round(float(getattr(props, "total_memory", 0)) / (1024 ** 3), 2)
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
    except Exception:
        device = "cpu"

    payload["device"] = device
    payload["gpu_name"] = gpu_name
    payload["gpu_memory_gb"] = gpu_memory_gb

    out = []
    for v in variants:
        ctor_name = v["ctor"]
        has_ctor = hasattr(tv_models, ctor_name)
        selectable = bool(has_ctor)
        reason = None

        # System gating for expensive variants.
        if selectable and v["id"] == "hrnet_w32":
            if device == "cpu":
                selectable = False
                reason = "Disabled on CPU to avoid very slow training."
            elif device == "cuda" and gpu_memory_gb is not None and gpu_memory_gb < 6.0:
                selectable = False
                reason = "Requires ~6GB+ GPU memory for stable training."

        if not has_ctor:
            reason = "Not available in this torchvision build."

        out.append(
            {
                "id": v["id"],
                "label": v["label"],
                "description": v["description"],
                "selectable": selectable,
                "recommended": False,
                "reason": reason,
            }
        )

    # Required default: SimpleBaseline.
    default_variant = None
    if any(v["id"] == "simplebaseline" and v["selectable"] for v in out):
        default_variant = "simplebaseline"
    else:
        first_selectable = next((v["id"] for v in out if v["selectable"]), None)
        default_variant = first_selectable

    for v in out:
        v["recommended"] = bool(v["id"] == default_variant)

    payload["default_variant"] = default_variant
    payload["variants"] = out
    print(json.dumps(payload))


if __name__ == "__main__":
    main()

