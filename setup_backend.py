#!/usr/bin/env python3
"""
BioVision Backend Setup Script
================================
Detects your hardware (NVIDIA GPU, Apple MPS, or CPU-only), creates a Python
virtual environment, and installs all backend dependencies with the correct
PyTorch build for your system.

Usage:
  Windows:      python setup_backend.py
  macOS/Linux:  python3 setup_backend.py

Tip:
  Use setup.bat / setup.sh from the project root to install both backend and
  frontend dependencies in one pass.
"""

import os
import re
import shutil
import subprocess
import sys
import platform
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

ROOT      = Path(__file__).parent.resolve()
VENV_DIR  = ROOT / "venv"
REQ_FILE  = ROOT / "backend" / "requirements.txt"
MIN_PYTHON = (3, 10)

# Supported PyTorch CUDA wheel indexes (highest version first for selection)
TORCH_CUDA_WHEELS: dict[str, str] = {
    "12.6": "https://download.pytorch.org/whl/cu126",
    "12.4": "https://download.pytorch.org/whl/cu124",
    "12.1": "https://download.pytorch.org/whl/cu121",
    "11.8": "https://download.pytorch.org/whl/cu118",
}

CLIP_URL = "git+https://github.com/openai/CLIP.git"

# ── Terminal helpers ───────────────────────────────────────────────────────────

W = platform.system() == "Windows"

def _c(code: str, text: str) -> str:
    """Wrap text in ANSI colour if the terminal supports it."""
    if W and "WT_SESSION" not in os.environ and "TERM" not in os.environ:
        return text  # plain Windows CMD — skip ANSI
    return f"\033[{code}m{text}\033[0m"

def hdr(msg: str) -> None:
    bar = "═" * 54
    print(f"\n╔{bar}╗")
    print(f"║  {msg:<52}║")
    print(f"╚{bar}╝")

def step(msg: str) -> None:
    print(f"\n{_c('1;36', '>>>')} {msg}")

def ok(msg: str) -> None:
    print(f"  {_c('32', '[OK]')}   {msg}")

def warn(msg: str) -> None:
    print(f"  {_c('33', '[WARN]')} {msg}")

def err(msg: str) -> None:
    print(f"  {_c('31', '[ERR]')} {msg}")

def info(msg: str) -> None:
    print(f"  {_c('90', '···')}   {msg}")

# ── Venv path helpers ──────────────────────────────────────────────────────────

def venv_python() -> Path:
    return VENV_DIR / ("Scripts/python.exe" if W else "bin/python")

def venv_pip() -> Path:
    return VENV_DIR / ("Scripts/pip.exe" if W else "bin/pip")

def _run(cmd: list, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, **kwargs)

def _check(cmd: list, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, **kwargs)


def _command_exists(cmd: str) -> bool:
    return shutil.which(cmd) is not None

# ── Step 1: Python version check ───────────────────────────────────────────────

def check_python() -> None:
    v = sys.version_info
    if v < MIN_PYTHON:
        err(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required — found {v.major}.{v.minor}")
        sys.exit(1)
    ok(f"Python {v.major}.{v.minor}.{v.micro}")

# ── Step 2: System info ────────────────────────────────────────────────────────

def _total_ram_gb() -> float | None:
    """Best-effort total RAM detection without psutil."""
    try:
        if W:
            r = _run(
                ["wmic", "ComputerSystem", "get", "TotalPhysicalMemory", "/value"],
                capture_output=True, text=True, timeout=5,
            )
            m = re.search(r"TotalPhysicalMemory=(\d+)", r.stdout)
            if m:
                return int(m.group(1)) / (1024 ** 3)
        elif sys.platform == "darwin":
            r = _run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5)
            return int(r.stdout.strip()) / (1024 ** 3)
        else:  # Linux
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        return kb / (1024 ** 2)
    except Exception:
        pass
    return None

def print_system_info() -> None:
    step("System information")
    info(f"OS       : {platform.system()} {platform.release()} ({platform.machine()})")
    info(f"CPU      : {platform.processor() or 'unknown'}")
    total = _total_ram_gb()
    info(f"RAM      : {f'{total:.1f} GB total' if total else 'unknown'}")

# ── Step 3: Create venv ────────────────────────────────────────────────────────

def create_venv() -> None:
    step("Virtual environment")
    if VENV_DIR.exists() and venv_python().exists():
        ok(f"venv already exists at {VENV_DIR}")
        return
    info(f"Creating venv at {VENV_DIR} …")
    _check([sys.executable, "-m", "venv", str(VENV_DIR)])
    ok("venv created")

# ── Step 4: Upgrade pip ────────────────────────────────────────────────────────

def upgrade_pip() -> None:
    step("Upgrading pip")
    _check(
        [
            str(venv_python()),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
            "setuptools",
            "wheel",
            "-q",
        ]
    )
    ok("pip/setuptools/wheel up-to-date")

# ── Step 5: Detect GPU / CUDA ──────────────────────────────────────────────────

def _find_nvidia_smi() -> str | None:
    """Return path to nvidia-smi or None."""
    found = shutil.which("nvidia-smi")
    if found:
        return found
    # Common Windows fallback location
    win_path = r"C:\Windows\System32\nvidia-smi.exe"
    if Path(win_path).exists():
        return win_path
    return None

def detect_cuda() -> str | None:
    """
    Returns:
      - CUDA version string like "12.6" if NVIDIA GPU detected
      - "mps"  if Apple Silicon MPS is available
      - None   if no accelerator found
    """
    step("Detecting GPU / accelerator")

    # ── NVIDIA ──────────────────────────────────────────────────────────────
    smi = _find_nvidia_smi()
    if smi:
        try:
            r = _run([smi], capture_output=True, text=True, timeout=10)
            m = re.search(r"CUDA Version:\s*([\d.]+)", r.stdout)
            if m:
                cuda_ver = m.group(1)
                # Also grab GPU name for display
                gm = re.search(r"^\|\s+\d+\s+(.*?)\s{2,}", r.stdout, re.MULTILINE)
                gpu_name = gm.group(1).strip() if gm else "NVIDIA GPU"
                ok(f"NVIDIA GPU detected: {gpu_name}  (driver CUDA {cuda_ver})")
                return cuda_ver
        except Exception:
            pass

    # ── Apple MPS ────────────────────────────────────────────────────────────
    if sys.platform == "darwin":
        try:
            r = _run(
                [str(venv_python()), "-c",
                 "import torch; print(torch.backends.mps.is_available())"],
                capture_output=True, text=True, timeout=15,
            )
            if r.returncode == 0 and r.stdout.strip() == "True":
                ok("Apple MPS (Metal) detected")
                return "mps"
        except Exception:
            # torch not yet installed — use sysctl as fallback
            r2 = _run(["sysctl", "-n", "hw.optional.arm64"], capture_output=True, text=True)
            if r2.returncode == 0 and r2.stdout.strip() == "1":
                ok("Apple Silicon detected — MPS will be available after torch install")
                return "mps"

    warn("No GPU/accelerator detected — PyTorch will be CPU-only")
    warn("SAM2 requires GPU + 4 GB free RAM; it will be unavailable in CPU mode")
    return None

# ── Step 6: Check available RAM ────────────────────────────────────────────────

def check_ram(cuda_ver: str | None) -> float:
    """Returns free RAM in GB. Warns if SAM2 requirements aren't met."""
    try:
        import psutil  # noqa: PLC0415
        free_gb = psutil.virtual_memory().available / (1024 ** 3)
        total_gb = psutil.virtual_memory().total / (1024 ** 3)
        ok(f"RAM: {free_gb:.1f} GB free / {total_gb:.1f} GB total")
        if cuda_ver and cuda_ver != "mps" and free_gb < 4:
            warn("Less than 4 GB free RAM — SAM2 may not load at runtime")
        return free_gb
    except ImportError:
        # psutil not yet installed in the venv (will be installed via requirements.txt)
        total = _total_ram_gb()
        if total:
            info(f"RAM: {total:.1f} GB total (psutil not yet installed — free RAM unknown)")
        else:
            info("RAM: unknown (psutil will be installed with requirements)")
        return 0.0

# ── Step 7: Install PyTorch ────────────────────────────────────────────────────

def _torch_installed() -> bool:
    r = _run(
        [str(venv_python()), "-c", "import torch"],
        capture_output=True, timeout=20,
    )
    return r.returncode == 0

def _torch_cuda_ok() -> bool:
    r = _run(
        [str(venv_python()), "-c", "import torch; print(torch.cuda.is_available())"],
        capture_output=True, text=True, timeout=20,
    )
    return r.returncode == 0 and r.stdout.strip() == "True"

def _torch_version() -> str:
    r = _run(
        [str(venv_python()), "-c", "import torch; print(torch.__version__)"],
        capture_output=True, text=True, timeout=20,
    )
    return r.stdout.strip() if r.returncode == 0 else "unknown"

def _pick_cuda_wheel(cuda_ver: str) -> str:
    """Return highest supported CUDA wheel key that is ≤ detected version."""
    detected = tuple(int(x) for x in cuda_ver.split(".")[:2])
    best = None
    for key in TORCH_CUDA_WHEELS:
        key_tup = tuple(int(x) for x in key.split("."))
        if key_tup <= detected:
            if best is None or key_tup > tuple(int(x) for x in best.split(".")):
                best = key
    return best or "12.4"

def install_torch(cuda_ver: str | None) -> None:
    step("Installing PyTorch")
    py = str(venv_python())

    needs_gpu = cuda_ver is not None and cuda_ver != "mps"

    if _torch_installed():
        cuda_ok = _torch_cuda_ok()
        ver = _torch_version()
        if needs_gpu and cuda_ok:
            ok(f"CUDA-enabled PyTorch {ver} already installed — skipping")
            return
        if not needs_gpu and not cuda_ok:
            ok(f"CPU PyTorch {ver} already installed — skipping")
            return
        if cuda_ver == "mps":
            ok(f"PyTorch {ver} already installed — skipping (MPS enabled via standard build)")
            return
        # Wrong build (e.g. CPU installed but GPU now detected) — reinstall
        warn(f"PyTorch {ver} is the wrong build — reinstalling")
        _run([py, "-m", "pip", "uninstall", "torch", "torchvision", "torchaudio", "-y"],
             capture_output=True)

    if needs_gpu:
        wheel_key  = _pick_cuda_wheel(cuda_ver)
        index_url  = TORCH_CUDA_WHEELS[wheel_key]
        info(f"Installing CUDA {wheel_key} build (index: {index_url}) …")
        _check([py, "-m", "pip", "install",
                "torch", "torchvision", "torchaudio",
                "--index-url", index_url])
        ok(f"PyTorch (CUDA {wheel_key}) installed")
    else:
        # CPU or MPS — both use the standard PyPI build
        label = "MPS (Apple Silicon)" if cuda_ver == "mps" else "CPU-only"
        info(f"Installing {label} build …")
        _check([py, "-m", "pip", "install", "torch", "torchvision", "torchaudio"])
        ok(f"PyTorch ({label}) installed")

# ── Step 8: Install requirements.txt ──────────────────────────────────────────

def install_requirements() -> None:
    step("Installing backend requirements")
    if not REQ_FILE.exists():
        warn(f"requirements.txt not found at {REQ_FILE} — skipping")
        return
    _check([str(venv_python()), "-m", "pip", "install", "-r", str(REQ_FILE)])
    ok("requirements installed")

# ── Step 9: Install CLIP ───────────────────────────────────────────────────────

def install_clip() -> None:
    step("Installing OpenAI CLIP")
    py = str(venv_python())

    # Check if already installed
    r = _run([py, "-c", "import clip"], capture_output=True, timeout=15)
    if r.returncode == 0:
        ok("CLIP already installed - skipping")
        return

    if not _command_exists("git"):
        warn("Git not found on PATH - CLIP install skipped")
        warn("Install Git, then run:")
        warn("  pip install git+https://github.com/openai/CLIP.git")
        return

    try:
        _check([py, "-m", "pip", "install", CLIP_URL])
        ok("CLIP installed")
    except subprocess.CalledProcessError:
        warn("CLIP install failed - text-prompt detection will be unavailable")
        warn("You can retry manually: pip install git+https://github.com/openai/CLIP.git")

def print_summary(cuda_ver: str | None, free_gb: float) -> None:
    py    = str(venv_python())
    torch_ver = _torch_version() if _torch_installed() else "not installed"
    clip_ok   = _run([py, "-c", "import clip"], capture_output=True, timeout=15).returncode == 0

    needs_gpu = cuda_ver is not None and cuda_ver != "mps"
    sam2_ok   = (needs_gpu or cuda_ver == "mps") and (free_gb == 0 or free_gb >= 4)

    if cuda_ver == "mps":
        gpu_line  = "Apple Silicon (MPS)"
        sam2_line = "✓ Will load (MPS + RAM OK)"
    elif needs_gpu:
        wheel_key = _pick_cuda_wheel(cuda_ver)
        gpu_line  = f"NVIDIA GPU  (CUDA {cuda_ver})"
        sam2_line = ("✓ Will load (GPU + RAM OK)" if sam2_ok
                     else "⚠ GPU OK but low free RAM (need 4 GB+)")
    else:
        gpu_line  = "None (CPU-only)"
        sam2_line = "✗ Unavailable (requires GPU)"

    torch_label = torch_ver
    cuda_ok = _torch_cuda_ok() if _torch_installed() else False
    if cuda_ok:
        torch_label += "  [CUDA ✓]"

    W_  = 52
    bar = "═" * W_
    def row(k: str, v: str) -> None:
        line = f"{k:<12}{v}"
        print(f"║  {line:<{W_}}║")

    print(f"\n╔{bar}╗")
    print(f"║{'  BioVision Setup Complete':^{W_}}║")
    print(f"╠{bar}╣")
    row("GPU",     gpu_line)
    row("PyTorch", torch_label)
    row("SAM2",    sam2_line)
    row("CLIP",    "✓ Installed" if clip_ok else "✗ Not installed (optional)")
    print(f"╠{bar}╣")
    venv_py = str(VENV_DIR / ("Scripts\\python.exe" if W else "bin/python"))
    print(f"║  {'venv python':<12}{venv_py:<{W_-12}}║")
    print(f"╚{bar}╝")
    print()
    print("  Next steps:")
    if W:
        print("    venv\\Scripts\\activate")
    else:
        print("    source venv/bin/activate")
    print("    npm install")
    print("    npm run dev")
    print()

# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    hdr("BioVision Backend Setup")

    check_python()
    print_system_info()
    create_venv()
    upgrade_pip()

    cuda_ver = detect_cuda()
    free_gb  = check_ram(cuda_ver)

    install_torch(cuda_ver)
    install_requirements()
    install_clip()

    print_summary(cuda_ver, free_gb)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup interrupted.")
        sys.exit(1)

