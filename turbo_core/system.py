"""System status and speed tips. Used by the startup banner and TURBO Manager."""

from __future__ import annotations

import os
import platform
import re
from dataclasses import dataclass, field
from typing import Any

from .llm import LLMError

try:
    import psutil
except ImportError:  # optional: resource numbers are skipped without it
    psutil = None

GB = 1024 ** 3

# Small, fast models worth suggesting on modest hardware (name, download size).
FAST_MODELS = {
    "code": [("qwen2.5-coder:3b", "1.9 GB"), ("qwen2.5-coder:1.5b", "1.0 GB")],
    "chat": [("llama3.2:3b", "2.0 GB"), ("qwen2.5:3b", "1.9 GB"), ("llama3.2:1b", "1.3 GB")],
}


@dataclass
class Snapshot:
    ollama_ok: bool
    installed: list[dict[str, Any]] = field(default_factory=list)
    loaded: list[dict[str, Any]] = field(default_factory=list)
    chat_model: str = ""
    code_model: str = ""
    context_tokens: int = 0
    ram_total_gb: float | None = None
    ram_free_gb: float | None = None
    cpu_percent: float | None = None
    cpu_cores: int | None = None
    os_name: str = ""

    @property
    def installed_names(self) -> list[str]:
        return [m["name"] for m in self.installed]

    def is_installed(self, name: str) -> bool:
        names = self.installed_names
        return name in names or (":" not in name and f"{name}:latest" in names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ollama_running": self.ollama_ok,
            "os": self.os_name,
            "cpu_cores": self.cpu_cores,
            "cpu_percent": self.cpu_percent,
            "ram_total_gb": self.ram_total_gb,
            "ram_free_gb": self.ram_free_gb,
            "active_chat_model": self.chat_model,
            "active_code_model": self.code_model,
            "context_tokens": self.context_tokens,
            "installed_models": [
                {"name": m["name"], "size_gb": round(m["size"] / GB, 1), "parameters": m["parameters"]}
                for m in self.installed
            ],
            "loaded_models": [
                {"name": m["name"], "size_gb": round(m["size"] / GB, 1),
                 "gpu_percent": m["gpu_percent"], "context": m["context"]}
                for m in self.loaded
            ],
        }


def snapshot(assistant, cpu_sample_seconds: float = 0.0) -> Snapshot:
    """Gather status. Never raises; missing pieces are left empty."""
    settings = assistant.settings
    llm = assistant.llm
    snap = Snapshot(
        ollama_ok=False,
        chat_model=settings.chat_model,
        code_model=settings.code_model,
        context_tokens=getattr(llm, "context_tokens", settings.context_tokens),
        os_name=f"{platform.system()} {platform.release()}",
        cpu_cores=os.cpu_count(),
    )
    try:
        snap.installed = llm.model_details()
        snap.ollama_ok = True
        snap.loaded = llm.loaded_models()
    except (LLMError, AttributeError):
        pass
    if psutil is not None:
        memory = psutil.virtual_memory()
        snap.ram_total_gb = round(memory.total / GB, 1)
        snap.ram_free_gb = round(memory.available / GB, 1)
        snap.cpu_percent = psutil.cpu_percent(interval=cpu_sample_seconds or None)
    return snap


def model_billions(name: str, installed: list[dict[str, Any]] | None = None) -> float | None:
    """Rough parameter count in billions, from Ollama's details or the tag."""
    for m in installed or []:
        if m["name"] == name and m.get("parameters"):
            match = re.match(r"([\d.]+)\s*([BM])", m["parameters"], re.IGNORECASE)
            if match:
                value = float(match.group(1))
                return value / 1000 if match.group(2).upper() == "M" else value
    match = re.search(r":(\d+(?:\.\d+)?)b\b", name.lower())
    return float(match.group(1)) if match else None


def speed_tips(snap: Snapshot) -> list[str]:
    """Plain-language suggestions to make TURBO faster on this machine."""
    tips: list[str] = []
    if not snap.ollama_ok:
        return ["Ollama isn't running. Start the Ollama app (or run `ollama serve`)."]

    for m in snap.loaded:
        if m["gpu_percent"] == 0:
            tips.append(
                f"{m['name']} runs 100% on the CPU (no GPU Ollama can use). Smaller models "
                "are the biggest speed win here."
            )
        elif m["gpu_percent"] < 100:
            tips.append(
                f"{m['name']} only fits {m['gpu_percent']}% on the GPU; the rest spills into "
                "CPU RAM, which is much slower. A smaller model or a smaller context "
                "(TURBO_CONTEXT_TOKENS) would fit entirely on the GPU."
            )

    on_cpu = not snap.loaded or any(m["gpu_percent"] == 0 for m in snap.loaded)
    for role, name in (("code", snap.code_model), ("chat", snap.chat_model)):
        size = model_billions(name, snap.installed)
        if on_cpu and size and size > 4:
            faster = ", ".join(n for n, _ in FAST_MODELS[role] if n != name)
            tips.append(f"Your {role} model {name} (~{size:g}B) is large for CPU. Faster: {faster} "
                        f"(try /model {role} {FAST_MODELS[role][0][0]}).")

    if snap.ram_free_gb is not None and snap.ram_free_gb < 2.5:
        tips.append(
            f"Only {snap.ram_free_gb} GB of RAM is free. Loading a model may swap to disk and "
            "stall for minutes. Close big apps (browsers, VMs) or use a 1-3B model."
        )
    if len(snap.loaded) > 1 and snap.ram_free_gb is not None and snap.ram_free_gb < 4:
        names = ", ".join(m["name"] for m in snap.loaded)
        tips.append(f"{len(snap.loaded)} models are loaded at once ({names}). Unloading unused ones "
                    "frees RAM (ollama stop <name>).")
    if snap.context_tokens > 8192:
        tips.append(f"The context window has grown to {snap.context_tokens} tokens (a big file was "
                    "shared). /clear and restarting TURBO brings it back down.")
    if snap.cpu_percent is not None and snap.cpu_percent > 85 and not snap.loaded:
        tips.append(f"CPU is at {snap.cpu_percent:.0f}% before TURBO even starts; other programs "
                    "will slow answers down.")
    return tips


def switch_model(assistant, role: str, name: str, persist: bool = False) -> str:
    """Switch the chat model, code model, or both ('both'). Returns a summary.

    persist=True also writes the choice to ~/.turbo/.env so it sticks.
    """
    from .config import save_setting

    role = role.lower()
    if role not in ("chat", "code", "both"):
        raise ValueError("role must be 'chat', 'code' or 'both'")
    name = name.strip()
    if not name or any(c.isspace() for c in name):
        raise ValueError("model name can't be empty or contain spaces")
    chat = name if role in ("chat", "both") else None
    code = name if role in ("code", "both") else None
    assistant.set_models(chat=chat, code=code)
    saved = ""
    if persist:
        path = None
        if chat:
            path = save_setting("TURBO_CHAT_MODEL", chat)
        if code:
            path = save_setting("TURBO_CODE_MODEL", code)
        saved = f" (saved to {path})"
    which = "chat and code models" if role == "both" else f"{role} model"
    return f"Switched {which} to {name}{saved}."
