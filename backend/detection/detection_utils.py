import math
from typing import Any, Mapping


ORIENTATION_MODES = {"directional", "bilateral", "axial", "invariant"}


def normalize_orientation_mode(mode: Any) -> str:
    mode_s = str(mode or "").strip().lower()
    if mode_s in ORIENTATION_MODES:
        return mode_s
    return "invariant"


def normalize_bilateral_class_axis(orientation_policy: Mapping[str, Any] | None = None) -> str:
    raw_value = None
    if isinstance(orientation_policy, Mapping):
        raw_value = orientation_policy.get("bilateralClassAxis")
    if raw_value not in (None, "", "vertical_obb"):
        raise ValueError(
            f'Unsupported bilateralClassAxis "{raw_value}". Bilateral inference expects vertical_obb only.'
        )
    return "vertical_obb"


def normalize_orientation_payload(raw_class_id: Any, orientation_policy: Mapping[str, Any] | None = None) -> dict[str, Any]:
    mode = normalize_orientation_mode((orientation_policy or {}).get("mode"))
    try:
      class_id = int(raw_class_id)
    except Exception:
      class_id = 0
    class_id = 0 if class_id <= 0 else 1

    if mode == "bilateral":
        orientation = "up" if class_id == 0 else "down"
    elif mode == "directional":
        orientation = "left" if class_id == 0 else "right"
    elif mode in {"axial", "invariant"}:
        orientation = None
    else:
        orientation = "left" if class_id == 0 else "right"

    payload: dict[str, Any] = {"class_id": class_id}
    if orientation is not None:
        payload["orientation_hint"] = {
            "orientation": orientation,
            "source": "obb_class_id",
        }
    return payload


def coerce_detector_angle_degrees(angle_rad: Any) -> float:
    try:
        return float(angle_rad) * 180.0 / math.pi
    except Exception:
        return 0.0
