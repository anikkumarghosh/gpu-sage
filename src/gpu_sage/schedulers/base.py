"""Scheduler interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from gpu_sage.core.models import Job

if TYPE_CHECKING:
    from gpu_sage.core.cluster import Cluster


class Scheduler(ABC):
    """Base interface shared by heuristic and RL schedulers.

    The scheduler conceptually receives:

    * current simulator/cluster state (``cluster`` + ``current_time``)
    * waiting_jobs (all jobs currently waiting, regardless of feasibility)
    * available resources (encoded as ``feasible_jobs`` and ``cluster.free_gpu_count``)

    and returns:

    * the Job it wants to schedule next, or
    * None to indicate NOOP / no feasible scheduling decision at this time.

    The simulator remains the sole owner of:

    * advancing time
    * job arrivals/completions
    * resource allocation/release

    Schedulers must NOT duplicate simulation logic.
    """

    @abstractmethod
    def select(
        self,
        waiting_jobs: list[Job],
        feasible_jobs: list[Job],
        cluster: "Cluster | None" = None,
        current_time: float = 0.0,
    ) -> Job | None:
        """Return the next job to launch, or None for NOOP.

        Args:
            waiting_jobs: All waiting jobs (unfiltered).
            feasible_jobs: Subset of waiting_jobs that can be allocated on free GPUs.
            cluster: Current cluster state (free/total GPUs, memory, etc.) or None.
            current_time: Simulator current time.

        Returns:
            Selected Job or None if no scheduling decision is possible.
        """
