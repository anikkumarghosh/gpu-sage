"""GPU cluster resource management (homogeneous + heterogeneous + topology-aware)."""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
from typing import Optional

from .models import GPU, Job, gpu_type_spec
from .topology import Topology


# Default placement alpha / cap (mirrors topology defaults for consistency)
DEFAULT_ALPHA = 0.6
DEFAULT_CAP = 2.5


@dataclass
class Cluster:
    """Collection of GPUs (homogeneous or heterogeneous) with optional topology."""

    gpus: list[GPU]
    topology: Optional[Topology] = None
    placement_alpha: float = DEFAULT_ALPHA
    placement_cap: float = DEFAULT_CAP

    @classmethod
    def homogeneous(cls, num_gpus: int, memory_gb: float = 80.0, gpu_type: str = "A100") -> "Cluster":
        if num_gpus <= 0:
            raise ValueError("num_gpus must be positive")
        spec = gpu_type_spec(gpu_type)
        # If caller left defaults but type implies different mem, still respect explicit memory_gb
        gpus = []
        for i in range(num_gpus):
            pf = float(spec["performance_factor"])
            cc = str(spec["compute_capability"])
            gpus.append(GPU(i, memory_gb, gpu_type, None, pf, cc))
        # Homogeneous keeps topology None so placement_penalty is 1.0 and
        # communication_cost is 0, preserving exact backward compatibility.
        # Heterogeneous clusters get an explicit topology via Cluster.heterogeneous.
        return cls(gpus=gpus, topology=None)

    @classmethod
    def heterogeneous(cls, specs: list[dict]) -> "Cluster":
        """Build cluster from explicit per-GPU specs.

        Each spec dict: {gpu_type, memory_gb, performance_factor?, compute_capability?}
        Example:
          [{"gpu_type": "A100_80GB", "memory_gb": 80}, {"gpu_type": "T4_16GB", ...}]
        """
        if not specs:
            raise ValueError("heterogeneous specs must be non-empty")
        gpus: list[GPU] = []
        for idx, s in enumerate(specs):
            gtype = s.get("gpu_type", s.get("type", "A100"))
            mem = float(s.get("memory_gb", gpu_type_spec(gtype)["memory_gb"]))
            pf = float(s.get("performance_factor", gpu_type_spec(gtype)["performance_factor"]))
            cc = str(s.get("compute_capability", gpu_type_spec(gtype)["compute_capability"]))
            gpus.append(GPU(idx, mem, gtype, None, pf, cc))
        topo = Topology.default_for(len(gpus))
        return cls(gpus=gpus, topology=topo)

    @classmethod
    def from_spec_list(cls, gpu_specs: list[dict], topology: Topology | None = None) -> "Cluster":
        """Convenience: heterogeneous specs + optional explicit topology."""
        c = cls.heterogeneous(gpu_specs)
        if topology is not None:
            c.topology = topology
        return c

    @property
    def total_gpus(self) -> int:
        return len(self.gpus)

    @property
    def free_gpus(self) -> list[GPU]:
        return [gpu for gpu in self.gpus if gpu.is_free]

    @property
    def free_gpu_count(self) -> int:
        return len(self.free_gpus)

    def feasible_gpu_ids(self, job: Job) -> list[int]:
        # Heterogeneous checks: type + memory
        candidates = []
        for gpu in self.gpus:
            if not gpu.is_free:
                continue
            if not job.is_compatible(gpu.gpu_type, gpu.memory_gb):
                continue
            candidates.append(gpu.gpu_id)
        return candidates

    def can_allocate(self, job: Job) -> bool:
        return len(self.feasible_gpu_ids(job)) >= job.gpu_count

    # --- topology-aware selection ---

    def _placement_cost(self, gpu_ids: list[int]) -> float:
        if self.topology is None or len(gpu_ids) <= 1:
            return 0.0
        return self.topology.communication_cost(gpu_ids)

    def _placement_penalty(self, job: Job, gpu_ids: list[int]) -> float:
        if self.topology is None:
            return 1.0
        return self.topology.placement_penalty(
            gpu_ids, job.topology_sensitive, alpha=self.placement_alpha, cap=self.placement_cap
        )

    def best_feasible_set(
        self,
        job: Job,
        strategy: str = "compact",
    ) -> list[int] | None:
        """Pick the best feasible GPU set for a job.

        Strategies:
          compact: minimize communication_cost (prefers tightly-connected group).
          spread: maximize communication_cost (for testing spread vs compact).
          best_fit: smallest sufficient feasible set by cost then memory fragmentation.
        Returns None if not feasible.
        """
        candidates = self.feasible_gpu_ids(job)
        if len(candidates) < job.gpu_count:
            return None
        n = job.gpu_count
        if n == 1:
            # Prefer GPU with smallest sufficient memory; tie-break by ID for determinism
            # and prefer preferred type if given
            def _score(gid: int):
                g = self.gpus[gid]
                pref_bonus = 0 if (job.preferred_gpu_type and g.gpu_type == job.preferred_gpu_type) else 1
                return (pref_bonus, g.memory_gb, gid)
            return [min(candidates, key=_score)]

        # For small clusters (<=16) we can enumerate combinations; for larger would sample.
        # 8 choose 4 = 70, so enumeration is cheap.
        all_combos = list(itertools.combinations(sorted(candidates), n))
        if not all_combos:
            return None

        if strategy == "spread":
            # Max cost
            best = max(all_combos, key=lambda c: (self._placement_cost(list(c)), list(c)))
            return list(best)

        # compact / best_fit: minimize cost, tie-break by memory sum and then lexicographic
        def _key(c):
            c_list = list(c)
            cost = self._placement_cost(c_list)
            # Prefer lower memory waste + preferred type alignment
            mem_sum = sum(self.gpus[gid].memory_gb for gid in c_list)
            pref_mismatch = sum(0 if self.gpus[gid].gpu_type == job.preferred_gpu_type else 1 for gid in c_list) if job.preferred_gpu_type else 0
            # Lower pref_mismatch is better
            return (cost, pref_mismatch, mem_sum, c_list)

        best = min(all_combos, key=_key)
        return list(best)

    def allocate(self, job: Job, strategy: str = "compact") -> list[int]:
        """Allocate the best feasible set (topology-aware)."""
        assigned = self.best_feasible_set(job, strategy=strategy)
        if assigned is None:
            raise ValueError(f"Job {job.job_id} cannot be allocated: insufficient compatible GPUs")
        for gpu_id in assigned:
            self.gpus[gpu_id].allocated_job_id = job.job_id
        # Annotate job with placement cost/penalty for later metrics/penalty
        if self.topology is not None:
            cost = self._placement_cost(assigned)
            penalty = self._placement_penalty(job, assigned)
            job.communication_cost = cost
            job.placement_penalty = penalty
        else:
            job.communication_cost = 0.0
            job.placement_penalty = 1.0
        return assigned

    def release(self, job: Job) -> None:
        for gpu in self.gpus:
            if gpu.allocated_job_id == job.job_id:
                gpu.allocated_job_id = None

    def utilization(self, running_jobs: dict[int, Job]) -> float:
        """GPU-count utilization in [0, 1]."""
        if self.total_gpus == 0:
            return 0.0
        used = sum(job.gpu_count for job in running_jobs.values())
        return used / self.total_gpus

    # Helpers for observation / dashboard
    def gpu_type_list(self) -> list[str]:
        return [g.gpu_type for g in self.gpus]

    def mem_list(self) -> list[float]:
        return [g.memory_gb for g in self.gpus]

    def perf_list(self) -> list[float]:
        return [g.performance_factor for g in self.gpus]
