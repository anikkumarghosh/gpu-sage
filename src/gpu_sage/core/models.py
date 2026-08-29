"""Core data models for GPU-Sage.

Heterogeneous GPU note: hardware characteristics below are *simulation
abstractions* (performance_factor, memory_gb, compute_capability) and NOT
claims about measured vendor performance. Values are relative and
explainable for studying placement-aware scheduling (see
docs/heterogeneous.md and src/gpu_sage/core/topology.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# ---------------------------------------------------------------------------
# Heterogeneous GPU type catalogue (simulation parameters)
# ---------------------------------------------------------------------------

# Reference specs used for heterogeneous experiments. performance_factor is
# a dimensionless scalar relative to A100_80GB = 1.0, used only inside the
# topology penalty model and observation. It does NOT imply exact benchmark parity.
GPU_TYPE_SPECS: dict[str, dict[str, float | str]] = {
    "A100_80GB": {"memory_gb": 80, "performance_factor": 1.0, "compute_capability": "8.0"},
    "A100_40GB": {"memory_gb": 40, "performance_factor": 0.90, "compute_capability": "8.0"},
    "A100": {"memory_gb": 80, "performance_factor": 1.0, "compute_capability": "8.0"},
    "V100_32GB": {"memory_gb": 32, "performance_factor": 0.65, "compute_capability": "7.0"},
    "V100": {"memory_gb": 32, "performance_factor": 0.65, "compute_capability": "7.0"},
    "T4_16GB": {"memory_gb": 16, "performance_factor": 0.35, "compute_capability": "7.5"},
    "T4": {"memory_gb": 16, "performance_factor": 0.35, "compute_capability": "7.5"},
}


def gpu_type_spec(gpu_type: str) -> dict:
    """Return spec dict for a gpu_type, falling back to A100 baseline."""
    return GPU_TYPE_SPECS.get(gpu_type, GPU_TYPE_SPECS["A100_80GB"])


class JobStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    COMPLETED = "completed"
    REJECTED = "rejected"


@dataclass(slots=True)
class Job:
    """A schedulable ML workload.

    Heterogeneous extensions (all optional, backward compatible):
      - required_gpu_type / preferred_gpu_type: None = any type.
      - min_gpu_memory_gb: if None, falls back to gpu_memory_gb.
      - topology_sensitive: if True, effective_runtime = base_duration * placement_penalty.
      - base_duration: original duration before penalty (auto-filled).
    """

    job_id: int
    arrival_time: float
    gpu_count: int
    gpu_memory_gb: float
    duration: float
    priority: int = 1
    job_type: str = "training"
    preemptible: bool = False

    # Heterogeneous / topology extensions
    required_gpu_type: Optional[str] = None
    preferred_gpu_type: Optional[str] = None
    min_gpu_memory_gb: Optional[float] = None  # None => uses gpu_memory_gb
    topology_sensitive: bool = False
    base_duration: Optional[float] = None  # filled on creation if None

    # Simulator-populated
    status: JobStatus = JobStatus.WAITING
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    assigned_gpus: list[int] = field(default_factory=list)
    placement_penalty: float = 1.0
    communication_cost: float = 0.0

    def __post_init__(self):
        if self.base_duration is None:
            self.base_duration = self.duration
        if self.min_gpu_memory_gb is None:
            # default: use per-GPU memory demand
            self.min_gpu_memory_gb = self.gpu_memory_gb

    @property
    def waiting_time(self) -> Optional[float]:
        if self.start_time is None:
            return None
        return self.start_time - self.arrival_time

    @property
    def turnaround_time(self) -> Optional[float]:
        if self.completion_time is None:
            return None
        return self.completion_time - self.arrival_time

    @property
    def effective_gpu_memory(self) -> float:
        return self.min_gpu_memory_gb if self.min_gpu_memory_gb is not None else self.gpu_memory_gb

    def is_compatible(self, gpu_type: str, memory_gb: float) -> bool:
        """Check type + memory compatibility for a single GPU."""
        if self.required_gpu_type is not None and gpu_type != self.required_gpu_type:
            return False
        if memory_gb < self.effective_gpu_memory:
            return False
        return True


@dataclass(slots=True)
class GPU:
    """A single GPU resource."""

    gpu_id: int
    memory_gb: float
    gpu_type: str = "A100"
    allocated_job_id: Optional[int] = None
    # Heterogeneous performance metadata (simulation abstraction)
    performance_factor: float = 1.0
    compute_capability: str = "8.0"

    def __post_init__(self):
        # Auto-fill from catalogue if defaults left
        spec = gpu_type_spec(self.gpu_type)
        # If caller passed default performance_factor, override from spec unless explicitly set.
        # We detect "explicit" by checking if gpu_type was non-default and factor is still 1.0
        # but the type's spec says otherwise — then use spec. This keeps homogeneous A100
        # homogeneous while heterogeneous configs get correct factors automatically.
        if self.gpu_type in GPU_TYPE_SPECS:
            s_mem = float(spec["memory_gb"])
            s_pf = float(spec["performance_factor"])
            s_cc = str(spec["compute_capability"])
            # Only auto-fill if caller left default-ish values but type implies difference
            # For memory, trust explicit memory_gb; for perf factor, trust spec if not set to custom.
            if self.performance_factor == 1.0 and s_pf != 1.0 and self.gpu_type != "A100":
                object.__setattr__(self, "performance_factor", s_pf)
            if self.compute_capability == "8.0" and s_cc != "8.0":
                object.__setattr__(self, "compute_capability", s_cc)
            # Memory: if caller passes default 80 but type says 16, trust type? We trust explicit.
            # Keep as-is — caller should pass matching memory_gb.

    @property
    def is_free(self) -> bool:
        return self.allocated_job_id is None


@dataclass(slots=True)
class SimulationEvent:
    """An event in the discrete-event simulator."""

    time: float
    sequence: int
    event_type: str
    job_id: int

    def sort_key(self) -> tuple[float, int]:
        return self.time, self.sequence
