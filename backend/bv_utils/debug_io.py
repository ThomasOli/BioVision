import json
import os
import re
import shutil
from datetime import datetime, timezone
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sanitize_name(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "default"
    text = text.replace(" ", "_")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = text.strip("._-")
    return text or "default"


def _run_id() -> str:
    # e.g. 20260221T184501_123456Z
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def _write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def write_json(path: str, data: Any) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def read_json(path: str, default: Any = None) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def append_json_array(path: str, row: Any, max_entries: int | None = None) -> str:
    data = read_json(path, default=[])
    if not isinstance(data, list):
        data = []
    data.append(row)
    if max_entries and len(data) > int(max_entries):
        data = data[-int(max_entries):]
    write_json(path, data)
    return path


def get_model_tag_dir(project_root: str, model_type: str, tag: str) -> str:
    model_key = _sanitize_name(model_type)
    tag_key = _sanitize_name(tag)
    out = os.path.join(project_root, "debug", "models", model_key, tag_key)
    os.makedirs(out, exist_ok=True)
    return out


def create_model_run_dir(project_root: str, model_type: str, tag: str) -> tuple[str, str]:
    tag_dir = get_model_tag_dir(project_root, model_type, tag)
    run_id = _run_id()
    run_dir = os.path.join(tag_dir, run_id)
    suffix = 1
    while os.path.exists(run_dir):
        run_dir = os.path.join(tag_dir, f"{run_id}_{suffix}")
        suffix += 1
    os.makedirs(run_dir, exist_ok=True)
    _write_text(os.path.join(tag_dir, "latest_run.txt"), os.path.basename(run_dir))
    return run_dir, os.path.basename(run_dir)


def write_run_json(run_dir: str, filename: str, data: Any) -> str:
    if not filename.endswith(".json"):
        filename = f"{filename}.json"
    return write_json(os.path.join(run_dir, filename), data)


def write_run_manifest(
    run_dir: str,
    *,
    model_type: str,
    tag: str,
    project_root: str,
    extra: dict[str, Any] | None = None,
) -> str:
    manifest = {
        "created_at": _utc_now_iso(),
        "model_type": model_type,
        "tag": tag,
        "project_root": os.path.abspath(project_root),
        "run_id": os.path.basename(run_dir),
    }
    if extra:
        manifest.update(extra)
    return write_run_json(run_dir, "run_manifest.json", manifest)


def copy_json_if_exists(src_path: str, dst_dir: str, dst_name: str | None = None) -> str | None:
    if not os.path.exists(src_path):
        return None
    if dst_name is None:
        dst_name = os.path.basename(src_path)
    dst_path = os.path.join(dst_dir, dst_name)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    payload = read_json(src_path, default=None)
    if payload is not None:
        write_json(dst_path, payload)
    else:
        shutil.copy2(src_path, dst_path)
    return dst_path


def append_model_prediction_log(
    project_root: str,
    model_type: str,
    tag: str,
    log_entry: Any,
    max_entries: int = 2000,
) -> str:
    tag_dir = get_model_tag_dir(project_root, model_type, tag)
    log_path = os.path.join(tag_dir, "prediction_log.json")
    append_json_array(log_path, log_entry, max_entries=max_entries)
    return log_path
