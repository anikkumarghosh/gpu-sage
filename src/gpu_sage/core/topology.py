"""Topology model for heterogeneous GPU cluster.

Abstraction: each GPU is a node; edges encode (bandwidth, latency, link_type).
This is NOT a claim about exact NVLink/PCIe topology of real servers. Values
are normalized and explainable for studying placement-aware scheduling.

Default: for 8 GPUs, two groups of 4 with NVLink inside group, PCIe across groups.
For other sizes, groups of 4 (e.g., 4,12,16 also work).

Formulas (documented, deterministic):

  pairwise latency in  [0, ~5] ms (normalized later)
  pairwise bandwidth in [0, 1]   (1 = NVLink, 0.2 = PCIe)

  communication_cost(S) = mean pairwise latency / bandwidth penalty
                         = mean_{i<j in S}  latency(i,j) / bandwidth(i,j)
                         normalized by dividing by MAX_LATENCY (5) so cost in ~[0,1+].
                         Single-GPU S => cost 0.

  placement_penalty(job, S) =
        1.0                                     if not job.topology_sensitive
        1.0 + alpha * communication_cost(S)     if topology_sensitive
        with alpha=0.6 by default (configurable).
        Capped at 2.5 to avoid arbitrary extreme values.

These are simulation abstractions; see README Advanced Extension for assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import math

# Normalized constants (simulation abstraction)
NVLINK_BANDWIDTH = 1.0
PCIE_BANDWIDTH = 0.22
NVLINK_LATENCY = 0.6
PCIE_LATENCY = 4.5
MAX_LATENCY = 5.0

LINK_TYPES = ("NVLINK", "PCIe")


@dataclass
class Topology:
    """Graph of GPUs with link bandwidth/latency."""

    num_gpus: int
    # adjacency indexed by (i,j) with i<j
    bandwidth: dict[tuple[int, int], float] = field(default_factory=dict)
    latency: dict[tuple[int, int], float] = field(default_factory=dict)
    link_type: dict[tuple[int, int], str] = field(default_factory=dict)

    @classmethod
    def fully_connected_nvlink(cls, num_gpus: int) -> "Topology":
        """Ideal topology: all pairs NVLink (penalty ~1)."""
        t = cls(num_gpus=num_gpus)
        for i in range(num_gpus):
            for j in range(i + 1, num_gpus):
                t.bandwidth[(i, j)] = NVLINK_BANDWIDTH
                t.latency[(i, j)] = NVLINK_LATENCY
                t.link_type[(i, j)] = "NVLINK"
        return t

    @classmethod
    def two_group(cls, num_gpus: int = 8, group_size: int = 4) -> "Topology":
        """Two-group model: NVLink inside group, PCIe across groups."""
        t = cls(num_gpus=num_gpus)
        for i in range(num_gpus):
            for j in range(i + 1, num_gpus):
                same_group = (i // group_size) == (j // group_size)
                if same_group:
                    t.bandwidth[(i, j)] = NVLINK_BANDWIDTH
                    t.latency[(i, j)] = NVLINK_LATENCY
                    t.link_type[(i, j)] = "NVLINK"
                else:
                    t.bandwidth[(i, j)] = PCIE_BANDWIDTH
                    t.latency[(i, j)] = PCIE_LATENCY
                    t.link_type[(i, j)] = "PCIe"
        return t

    @classmethod
    def default_for(cls, num_gpus: int) -> "Topology":
        """Pick a sensible default per cluster size."""
        if num_gpus <= 4:
            return cls.fully_connected_nvlink(num_gpus)
        return cls.two_group(num_gpus, group_size=4)

    def get_bandwidth(self, i: int, j: int) -> float:
        if i == j:
            return NVLINK_BANDWIDTH
        a, b = (i, j) if i < j else (j, i)
        return self.bandwidth.get((a, b), PCIE_BANDWIDTH)

    def get_latency(self, i: int, j: int) -> float:
        if i == j:
            return 0.0
        a, b = (i, j) if i < j else (j, i)
        return self.latency.get((a, b), PCIE_LATENCY)

    def get_link_type(self, i: int, j: int) -> str:
        if i == j:
            return "NVLINK"
        a, b = (i, j) if i < j else (j, i)
        return self.link_type.get((a, b), "PCIe")

    def communication_cost(self, gpu_ids: list[int]) -> float:
        """Mean pairwise latency/bandwidth cost for set S.

        For |S| <=1, cost 0. Single-GPU jobs are unaffected.
        Otherwise: mean_{i<j} latency(i,j) / bandwidth(i,j) / MAX_LATENCY
        so cost is roughly in [0, ~1.5] and explainable.
        """
        if len(gpu_ids) <= 1:
            return 0.0
        pairs = list(itertools.combinations(sorted(gpu_ids), 2))
        costs = []
        for a, b in pairs:
            bw = self.get_bandwidth(a, b)
            lat = self.get_latency(a, b)
            # bandwidth in (0,1], so lat/bw amplifies PCIe
            costs.append(lat / max(bw, 1e-9) / MAX_LATENCY)
        return float(sum(costs) / len(costs)) if costs else 0.0

    def placement_penalty(
        self, gpu_ids: list[int], topology_sensitive: bool, alpha: float = 0.6, cap: float = 2.5
    ) -> float:
        """Effective runtime multiplier.

        1.0 if not topology_sensitive else 1 + alpha * communication_cost.
        Capped at `cap` to avoid extremes.
        """
        if not topology_sensitive:
            return 1.0
        cost = self.communication_cost(gpu_ids)
        penalty = 1.0 + alpha * cost
        return float(min(penalty, cap))

    def adjacency_matrix(self, normalize: bool = True) -> list[list[float]]:
        """Return symmetric normalized bandwidth matrix for observation."""
        n = self.num_gpus
        mat = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    mat[i][j] = 1.0
                else:
                    bw = self.get_bandwidth(i, j)
                    mat[i][j] = bw  # already in [0,1]
        return mat

    def to_dict(self) -> dict:
        return {
            "num_gpus": self.num_gpus,
            "bandwidth": {f"{k[0]}-{k[1]}": v for k, v in self.bandwidth.items()},
            "latency": {f"{k[0]}-{k[1]}": v for k, v in self.latency.items()},
            "link_type": {f"{k[0]}-{k[1]}": v for k, v in self.link_type.items()},
        }
