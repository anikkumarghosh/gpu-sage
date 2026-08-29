"""Scheduler interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

from gpu_sage.core.models import Job


class Scheduler(ABC):
    """Base interface shared by heuristic and RL schedulers."""

    @abstractmethod
    def select(self, waiting_jobs: list[Job], feasible_jobs: list[Job]) -> Job | None:
        """Return the next job to launch, or None."""
