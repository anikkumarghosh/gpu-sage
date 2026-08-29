"""Core data models for GPU-Sage."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class JobStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    COMPLETED = "completed"
    REJECTED = "rejected"


@dataclass(slots=True)
class Job:
    """A schedulable ML workload."""

    job_id: int
    arrival_time: float
    gpu_count: int
    gpu_memory_gb: float
    duration: float
    priority: int = 1
    job_type: str = "training"
    preemptible: bool = False

    status: JobStatus = JobStatus.WAITING
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    assigned_gpus: list[int] = field(default_factory=list)

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


@dataclass(slots=True)
class GPU:
    """A single GPU resource."""

    gpu_id: int
    memory_gb: float
    gpu_type: str = "A100"
    allocated_job_id: Optional[int] = None

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
