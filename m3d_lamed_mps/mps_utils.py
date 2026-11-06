"""Apple-Silicon/MPS helpers: no-CPU-fallback guard, op-level device audit, memory measurement."""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import threading
import time
from collections import Counter
from dataclasses import dataclass, field

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

GiB = 1024**3


# --------------------------------------------------------------------------- device guard
def mps_fallback_enabled() -> bool:
    """True if PYTORCH_ENABLE_MPS_FALLBACK would silently run unsupported ops on the CPU."""
    return os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0").strip() not in ("", "0")


def require_mps(allow_cpu_fallback: bool = False) -> torch.device:
    """Return the MPS device or raise. By default refuses to run with the CPU fallback enabled,
    so that any operator without a Metal kernel raises instead of silently running on the CPU."""
    if not torch.backends.mps.is_built():
        raise RuntimeError("this PyTorch build has no MPS support (install the macOS arm64 wheel)")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available (needs Apple Silicon + a supported macOS)")
    if mps_fallback_enabled() and not allow_cpu_fallback:
        raise RuntimeError(
            "PYTORCH_ENABLE_MPS_FALLBACK is set: unset it (export PYTORCH_ENABLE_MPS_FALLBACK=0) so that "
            "unsupported operators fail loudly instead of silently running on the CPU"
        )
    return torch.device("mps")


def synchronize(device: torch.device | str | None = None) -> None:
    dev = torch.device(device) if device is not None else None
    if dev is None or dev.type == "mps":
        if torch.backends.mps.is_available():
            torch.mps.synchronize()


def resolve_dtype(name: str | torch.dtype) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    table = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp16": torch.float16, "float16": torch.float16,
             "half": torch.float16, "fp32": torch.float32, "float32": torch.float32}
    try:
        return table[name.lower()]
    except KeyError as e:
        raise ValueError(f"unknown dtype {name!r}; use bf16, fp16 or fp32") from e


# --------------------------------------------------------------------------- op audit
_TRANSFER_OPS = {"_to_copy", "copy_", "to", "_copy_from", "_copy_from_and_resize", "lift_fresh", "lift_fresh_copy"}


@dataclass
class DeviceAuditReport:
    mps_ops: Counter = field(default_factory=Counter)
    transfer_ops: Counter = field(default_factory=Counter)
    cpu_ops: Counter = field(default_factory=Counter)  # compute on a non-scalar CPU tensor = violation
    cpu_scalar_ops: Counter = field(default_factory=Counter)  # only CPU scalars (e.g. Python numbers)
    other_ops: Counter = field(default_factory=Counter)

    @property
    def total_mps(self) -> int:
        return sum(self.mps_ops.values())

    @property
    def ok(self) -> bool:
        return not self.cpu_ops and not self.other_ops and self.total_mps > 0

    def summary(self, top: int = 12) -> dict:
        return {
            "ok": self.ok,
            "mps_compute_ops": self.total_mps,
            "distinct_mps_ops": len(self.mps_ops),
            "top_mps_ops": dict(self.mps_ops.most_common(top)),
            "host_device_transfers": dict(self.transfer_ops),
            "cpu_compute_ops": dict(self.cpu_ops),
            "cpu_scalar_ops": dict(self.cpu_scalar_ops),
            "other_device_ops": dict(self.other_ops),
        }


class DeviceAudit(TorchDispatchMode):
    """Records every ATen op executed inside the context and the devices of its tensors.

    A compute op is counted as a *CPU violation* if any input/output tensor with more than one
    element lives on the CPU (0-dim / 1-element CPU scalars are how Python numbers are wrapped and
    are not computation). Host<->device copies are counted separately.

    Note: PyTorch's MPS->CPU fallback happens *below* this dispatch level, so this audit cannot
    see it; that is what :func:`require_mps` (fallback disabled -> missing kernels raise) is for.
    """

    def __init__(self):
        super().__init__()
        self.report = DeviceAuditReport()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        name = func.overloadpacket.__name__
        devices, cpu_big = set(), False
        for t in tree_flatten((args, kwargs, out))[0]:
            if isinstance(t, torch.Tensor):
                devices.add(t.device.type)
                if t.device.type == "cpu" and t.numel() > 1:
                    cpu_big = True
        if not devices:
            pass  # no tensors involved
        elif name in _TRANSFER_OPS and len(devices) > 1:
            self.report.transfer_ops[name] += 1
        elif cpu_big:
            self.report.cpu_ops[name] += 1
        elif devices <= {"mps", "cpu"} and "mps" in devices:
            self.report.mps_ops[name] += 1
        elif devices == {"cpu"}:
            self.report.cpu_scalar_ops[name] += 1  # only 0-dim/1-element CPU scalars: bookkeeping
        else:
            self.report.other_ops[f"{name}@{','.join(sorted(devices))}"] += 1
        return out


# --------------------------------------------------------------------------- memory
class _RUsageInfoV4(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (n, ctypes.c_uint64)
        for n in (
            "ri_user_time ri_system_time ri_pkg_idle_wkups ri_interrupt_wkups ri_pageins ri_wired_size "
            "ri_resident_size ri_phys_footprint ri_proc_start_abstime ri_proc_exit_abstime ri_child_user_time "
            "ri_child_system_time ri_child_pkg_idle_wkups ri_child_interrupt_wkups ri_child_pageins "
            "ri_child_elapsed_abstime ri_diskio_bytesread ri_diskio_byteswritten ri_cpu_time_qos_default "
            "ri_cpu_time_qos_maintenance ri_cpu_time_qos_background ri_cpu_time_qos_utility "
            "ri_cpu_time_qos_legacy ri_cpu_time_qos_user_initiated ri_cpu_time_qos_user_interactive "
            "ri_billed_system_time ri_serviced_system_time ri_logical_writes ri_lifetime_max_phys_footprint "
            "ri_instructions ri_cycles ri_billed_energy ri_serviced_energy ri_interval_max_phys_footprint "
            "ri_runnable_time"
        ).split()
    ]


_RUSAGE_INFO_V4 = 4
_libsystem = None


def process_footprint() -> dict:
    """macOS phys_footprint of this process (what Activity Monitor calls "Memory"); on Apple
    Silicon it includes Metal/GPU allocations. Returns bytes: current and lifetime peak."""
    global _libsystem
    if platform.system() != "Darwin":
        return {}
    if _libsystem is None:
        _libsystem = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        _libsystem.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        _libsystem.proc_pid_rusage.restype = ctypes.c_int
    info = _RUsageInfoV4()
    if _libsystem.proc_pid_rusage(os.getpid(), _RUSAGE_INFO_V4, ctypes.byref(info)) != 0:
        return {}
    return {
        "phys_footprint": int(info.ri_phys_footprint),
        "lifetime_max_phys_footprint": int(info.ri_lifetime_max_phys_footprint),
        "resident_size": int(info.ri_resident_size),
    }


def mps_memory() -> dict:
    if not torch.backends.mps.is_available():
        return {}
    return {
        "mps_current_allocated": int(torch.mps.current_allocated_memory()),
        "mps_driver_allocated": int(torch.mps.driver_allocated_memory()),
    }


class PeakMemorySampler:
    """Background sampler of MPS allocator memory (PyTorch has no MPS ``max_memory_allocated``).

    Use as a context manager; ``.result()`` gives peaks in bytes. Sampling interval 2 ms by
    default; brief spikes shorter than that can be missed, so the macOS lifetime peak footprint
    (``process_footprint()["lifetime_max_phys_footprint"]``) is reported alongside.
    """

    def __init__(self, interval_s: float = 0.002):
        self.interval_s = interval_s
        self.peak_allocated = 0
        self.peak_driver = 0
        self.samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        a, d = torch.mps.current_allocated_memory(), torch.mps.driver_allocated_memory()
        self.peak_allocated = max(self.peak_allocated, a)
        self.peak_driver = max(self.peak_driver, d)
        self.samples += 1

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            time.sleep(self.interval_s)

    def __enter__(self):
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        synchronize()
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()
        return False

    def result(self) -> dict:
        return {"mps_peak_allocated": self.peak_allocated, "mps_peak_driver_allocated": self.peak_driver, "samples": self.samples}


def fmt_gib(nbytes: int | float | None) -> str:
    return "n/a" if nbytes is None else f"{nbytes / GiB:.2f} GiB"


# --------------------------------------------------------------------------- system info
def _sysctl(name: str) -> str:
    try:
        return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""


def system_info() -> dict:
    import transformers

    info = {
        "macos": platform.mac_ver()[0],
        "machine": platform.machine(),
        "chip": _sysctl("machdep.cpu.brand_string"),
        "memory_bytes": int(_sysctl("hw.memsize") or 0),
        "gpu_cores": None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "mps_fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
    }
    if torch.backends.mps.is_available():
        info["mps_recommended_max_memory"] = int(torch.mps.recommended_max_memory())
    try:
        out = subprocess.run(["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True, timeout=20).stdout
        for line in out.splitlines():
            if "Total Number of Cores" in line:
                info["gpu_cores"] = int(line.split(":")[1].strip())
    except Exception:
        pass
    return info
