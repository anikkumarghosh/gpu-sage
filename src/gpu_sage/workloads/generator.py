"""Synthetic workload generation with reproducible named scenarios."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterator

import numpy as np

from gpu_sage.core.models import Job


@dataclass
class WorkloadConfig:
    """Bounded workload distribution parameters.

    All random draws are performed via ``numpy.random.Generator`` seeded by the
    caller, so identical seeds produce identical workloads. Distribution
    parameters are explicit and not hard-coded inside the generator.
    """

    arrival_rate: float = 0.15
    min_gpus: int = 1
    max_gpus: int = 4
    min_memory_gb: float = 8.0
    max_memory_gb: float = 64.0
    min_duration: float = 20.0
    max_duration: float = 200.0
    min_priority: int = 1
    max_priority: int = 5
    # Optional heavy-tail / burst tailoring (used only by some scenarios)
    heavy_tail_fraction: float = 0.05
    heavy_tail_multiplier: float = 5.0


# ---------------------------------------------------------------------------
# Named scenario presets — parameters are explicit, documented, and reproducible.
# ---------------------------------------------------------------------------

SCENARIO_CONFIGS: dict[str, WorkloadConfig] = {
    "balanced": WorkloadConfig(
        arrival_rate=0.08,
        min_gpus=1,
        max_gpus=4,
        min_memory_gb=8.0,
        max_memory_gb=64.0,
        min_duration=20.0,
        max_duration=180.0,
        min_priority=1,
        max_priority=5,
    ),
    "gpu_heavy": WorkloadConfig(
        arrival_rate=0.05,
        min_gpus=4,
        max_gpus=8,
        min_memory_gb=16.0,
        max_memory_gb=64.0,
        min_duration=60.0,
        max_duration=400.0,
        min_priority=1,
        max_priority=5,
    ),
    "short_jobs": WorkloadConfig(
        arrival_rate=0.20,
        min_gpus=1,
        max_gpus=2,
        min_memory_gb=8.0,
        max_memory_gb=32.0,
        min_duration=5.0,
        max_duration=50.0,
        min_priority=1,
        max_priority=5,
    ),
    "bursty": WorkloadConfig(
        # Base rate; generator will modulate burst periods.
        arrival_rate=0.08,
        min_gpus=1,
        max_gpus=4,
        min_memory_gb=8.0,
        max_memory_gb=64.0,
        min_duration=20.0,
        max_duration=180.0,
        min_priority=1,
        max_priority=5,
    ),
    "heavy_tail": WorkloadConfig(
        arrival_rate=0.08,
        min_gpus=1,
        max_gpus=4,
        min_memory_gb=8.0,
        max_memory_gb=64.0,
        min_duration=10.0,
        max_duration=400.0,
        min_priority=1,
        max_priority=5,
        heavy_tail_fraction=0.07,
        heavy_tail_multiplier=8.0,
    ),
    "priority_skew": WorkloadConfig(
        arrival_rate=0.08,
        min_gpus=1,
        max_gpus=4,
        min_memory_gb=8.0,
        max_memory_gb=64.0,
        min_duration=20.0,
        max_duration=180.0,
        min_priority=1,
        max_priority=5,
    ),
}

SCENARIO_NAMES: list[str] = sorted(SCENARIO_CONFIGS.keys())


def get_scenario_config(scenario: str) -> WorkloadConfig:
    """Return a copy of the preset config for ``scenario``."""
    if scenario not in SCENARIO_CONFIGS:
        raise ValueError(f"Unknown scenario '{scenario}'. Available: {SCENARIO_NAMES}")
    # Return a copy so callers can mutate without affecting the preset.
    return replace(SCENARIO_CONFIGS[scenario])


def list_scenarios() -> list[str]:
    """Return sorted list of available scenario names."""
    return list(SCENARIO_NAMES)


class SyntheticWorkload:
    """Poisson arrivals + bounded job properties, with scenario-aware generation.

    The generator is fully deterministic given ``seed``. It supports
    scenario-specific distributions without hard-coded global randomness:

    * ``balanced`` — moderate rate, 1–4 GPUs, mixed runtimes
    * ``gpu_heavy`` — many 4–8 GPU, longer runtimes
    * ``short_jobs`` — many short-duration, 1–2 GPUs
    * ``bursty`` — periods of low arrivals followed by bursts
    * ``heavy_tail`` — mostly short with few extremely long jobs
    * ``priority_skew`` — mostly low/medium priority with occasional high priority

    The caller controls: seed, number of jobs, arrival-rate parameters,
    and distribution parameters via ``WorkloadConfig``.
    """

    def __init__(
        self,
        config: WorkloadConfig | None = None,
        seed: int = 0,
        scenario: str | None = None,
    ) -> None:
        if config is None and scenario is not None:
            config = get_scenario_config(scenario)
        if config is None:
            config = WorkloadConfig()
        if config.arrival_rate <= 0:
            raise ValueError("arrival_rate must be positive")
        self.config = config
        self.scenario = scenario or "custom"
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    # -- internal helpers -------------------------------------------------

    def _arrival_times(self, count: int) -> np.ndarray:
        cfg = self.config
        if self.scenario == "bursty":
            # Bursty: alternate low-rate and burst-rate periods.
            # Each chunk of ~8 jobs is either burst or quiet.
            # Quiet: 0.3 * base rate, burst: 5 * base rate.
            chunk = 8
            inter: list[float] = []
            for i in range(count):
                is_burst_chunk = self.rng.random() < 0.25 if i % chunk == 0 else inter and getattr(self, "_in_burst", False)
                # Re-decide per chunk.
                if i % chunk == 0:
                    self._in_burst = self.rng.random() < 0.25  # 25% chunks are bursts
                rate = cfg.arrival_rate * 5.0 if getattr(self, "_in_burst", False) else cfg.arrival_rate * 0.35
                inter.append(float(self.rng.exponential(1.0 / rate)))
            return np.cumsum(np.array(inter))
        else:
            inter_arrivals = self.rng.exponential(1.0 / cfg.arrival_rate, size=count)
            return np.cumsum(inter_arrivals)

    def _sample_gpu_count(self) -> int:
        cfg = self.config
        return int(self.rng.integers(cfg.min_gpus, cfg.max_gpus + 1))

    def _sample_duration(self) -> float:
        cfg = self.config
        if self.scenario == "heavy_tail":
            # Mixture: heavy_tail_fraction are extremely long.
            if self.rng.random() < cfg.heavy_tail_fraction:
                # Long tail: uniform between max_duration and max_duration * multiplier
                low = cfg.max_duration
                high = cfg.max_duration * cfg.heavy_tail_multiplier
                return float(self.rng.uniform(low, high))
            # Otherwise short: uniform between min_duration and max_duration*0.4
            # to keep median low.
            short_high = min(cfg.max_duration, cfg.min_duration + (cfg.max_duration - cfg.min_duration) * 0.4)
            return float(self.rng.uniform(cfg.min_duration, short_high))
        elif self.scenario == "short_jobs":
            # Already bounded short; uniform within config is fine.
            return float(self.rng.uniform(cfg.min_duration, cfg.max_duration))
        else:
            return float(self.rng.uniform(cfg.min_duration, cfg.max_duration))

    def _sample_priority(self) -> int:
        cfg = self.config
        if self.scenario == "priority_skew":
            # Skew: ~60% 1, 25% 2, 10% 3, 4% 4, 1% 5
            r = self.rng.random()
            if r < 0.60:
                return 1
            elif r < 0.85:
                return 2
            elif r < 0.95:
                return 3
            elif r < 0.99:
                return 4
            else:
                return 5
        else:
            return int(self.rng.integers(cfg.min_priority, cfg.max_priority + 1))

    # -- public API -------------------------------------------------------

    def generate(self, count: int) -> list[Job]:
        """Generate ``count`` jobs deterministically."""
        if count <= 0:
            return []

        arrivals = self._arrival_times(count)
        jobs: list[Job] = []

        for job_id, arrival_time in enumerate(arrivals):
            # For heavy_tail, occasionally force large gpu_count as well? Keep as per config.
            gpu_count = self._sample_gpu_count()
            # Clamp to valid range.
            gpu_count = max(1, gpu_count)
            jobs.append(
                Job(
                    job_id=job_id,
                    arrival_time=float(arrival_time),
                    gpu_count=gpu_count,
                    gpu_memory_gb=float(self.rng.uniform(self.config.min_memory_gb, self.config.max_memory_gb)),
                    duration=self._sample_duration(),
                    priority=self._sample_priority(),
                )
            )
        return jobs


def generate_workload(
    scenario: str = "balanced",
    seed: int = 0,
    count: int = 100,
    config: WorkloadConfig | None = None,
) -> list[Job]:
    """Convenience helper to generate a scenario workload deterministically.

    Args:
        scenario: One of SCENARIO_NAMES. Ignored if ``config`` is provided
            with a custom ``WorkloadConfig``.
        seed: Deterministic RNG seed.
        count: Number of jobs.
        config: Optional override config. If None, the scenario preset is used.

    Returns:
        List of Jobs with deterministic arrivals/properties.
    """
    if config is not None:
        gen = SyntheticWorkload(config=config, seed=seed, scenario=scenario if scenario in SCENARIO_CONFIGS else None)
    else:
        gen = SyntheticWorkload(scenario=scenario, seed=seed)
    return gen.generate(count)
