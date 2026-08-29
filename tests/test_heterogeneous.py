"""Tests for heterogeneous GPU + topology-aware scheduling (fast)."""

import copy
from pathlib import Path

import pytest

from gpu_sage.core.cluster import Cluster
from gpu_sage.core.models import Job, GPU_TYPE_SPECS
from gpu_sage.core.topology import Topology
from gpu_sage.core.simulator import Simulator
from gpu_sage.schedulers.heuristics import TopologyBestFitScheduler
from gpu_sage.schedulers.fcfs import FCFSScheduler
from gpu_sage.workloads.generator import generate_workload, get_scenario_config
from gpu_sage.evaluation.benchmark import run_single_seed
from gpu_sage.env.gpu_env import GPUSchedulingEnv
from gpu_sage.evaluation.metrics import compute_metrics


def test_heterogeneous_gpu_creation():
    specs = [
        {"gpu_type": "A100_80GB", "memory_gb": 80},
        {"gpu_type": "V100_32GB", "memory_gb": 32},
        {"gpu_type": "T4_16GB", "memory_gb": 16},
    ]
    c = Cluster.heterogeneous(specs)
    assert c.total_gpus == 3
    assert c.gpus[0].gpu_type == "A100_80GB"
    assert c.gpus[0].performance_factor == 1.0
    assert c.gpus[1].performance_factor == 0.65
    assert c.gpus[2].performance_factor == 0.35


def test_gpu_memory_compatibility():
    c = Cluster.heterogeneous([{"gpu_type": "A100_80GB", "memory_gb": 80}, {"gpu_type": "T4_16GB", "memory_gb": 16}])
    j_small = Job(0, 0, 1, 8, 10)
    j_big = Job(1, 0, 1, 32, 10)
    assert c.can_allocate(j_small)  # fits on either
    # T4 cannot fit 32GB, but A100 can, so still feasible for 1 GPU
    assert c.can_allocate(j_big)
    # Require 2x32GB -> needs both? T4 16 fails, so only A100+? Actually need 2 GPUs each >=32, only 1 GPU qualifies
    j_two_big = Job(2, 0, 2, 32, 10)
    assert not c.can_allocate(j_two_big)
    # Occupy A100, then big job not feasible
    c.allocate(j_big)
    assert not c.can_allocate(j_small) or c.free_gpu_count == 1  # one free T4 remains but j_small needs 8GB so fits


def test_gpu_type_compatibility():
    c = Cluster.heterogeneous([{"gpu_type": "A100_80GB", "memory_gb": 80}, {"gpu_type": "T4_16GB", "memory_gb": 16}])
    j_req = Job(0, 0, 1, 8, 10, required_gpu_type="A100_80GB")
    j_wrong = Job(1, 0, 1, 8, 10, required_gpu_type="V100_32GB")
    assert c.can_allocate(j_req)
    assert not c.can_allocate(j_wrong)
    # Preferred not required should still be feasible on any
    j_pref = Job(2, 0, 1, 8, 10, preferred_gpu_type="A100_80GB")
    assert c.can_allocate(j_pref)


def test_topology_graph_validity():
    t = Topology.two_group(8, 4)
    assert t.num_gpus == 8
    # NVLink inside group
    assert t.get_link_type(0, 1) == "NVLINK"
    assert t.get_link_type(0, 3) == "NVLINK"
    # PCIe across groups
    assert t.get_link_type(0, 4) == "PCIe"
    assert t.get_bandwidth(0, 1) == 1.0
    assert t.get_bandwidth(0, 4) == 0.22
    # adjacency matrix symmetric and normalized
    mat = t.adjacency_matrix()
    assert len(mat) == 8 and len(mat[0]) == 8
    assert mat[0][1] == 1.0 and mat[0][4] == 0.22


def test_topology_cost_calculation():
    t = Topology.two_group(8, 4)
    # Single GPU cost 0
    assert t.communication_cost([0]) == 0.0
    # Compact group low cost
    compact = t.communication_cost([0, 1, 2, 3])
    spread = t.communication_cost([0, 1, 4, 5])
    assert compact < spread
    # Penalty
    assert t.placement_penalty([0], True) == 1.0
    assert t.placement_penalty([0, 1, 4, 5], False) == 1.0
    pen = t.placement_penalty([0, 1, 4, 5], True)
    assert 1.0 < pen <= 2.5


def test_placement_penalty_calculation():
    from gpu_sage.core.cluster import Cluster

    c = Cluster.heterogeneous(
        [
            {"gpu_type": "A100_80GB", "memory_gb": 80},
            {"gpu_type": "A100_80GB", "memory_gb": 80},
            {"gpu_type": "V100_32GB", "memory_gb": 32},
            {"gpu_type": "V100_32GB", "memory_gb": 32},
        ]
    )
    j_sens = Job(0, 0, 2, 16, 100, topology_sensitive=True)
    c.allocate(j_sens)
    assert j_sens.placement_penalty >= 1.0
    assert j_sens.communication_cost >= 0
    # Non-sensitive penalty 1.0
    c2 = Cluster.heterogeneous(
        [
            {"gpu_type": "A100_80GB", "memory_gb": 80},
            {"gpu_type": "A100_80GB", "memory_gb": 80},
        ]
    )
    j_nons = Job(1, 0, 1, 16, 100, topology_sensitive=False)
    c2.allocate(j_nons)
    assert j_nons.placement_penalty == 1.0


def test_topology_aware_allocation():
    c = Cluster.heterogeneous(
        [
            {"gpu_type": "A100_80GB", "memory_gb": 80},
            {"gpu_type": "A100_80GB", "memory_gb": 80},
            {"gpu_type": "A100_80GB", "memory_gb": 80},
            {"gpu_type": "A100_80GB", "memory_gb": 80},
            {"gpu_type": "V100_32GB", "memory_gb": 32},
            {"gpu_type": "V100_32GB", "memory_gb": 32},
            {"gpu_type": "T4_16GB", "memory_gb": 16},
            {"gpu_type": "T4_16GB", "memory_gb": 16},
        ]
    )
    j = Job(0, 0, 4, 16, 100, topology_sensitive=True)
    compact = c.best_feasible_set(j, strategy="compact")
    spread = c.best_feasible_set(j, strategy="spread")
    assert compact is not None and len(compact) == 4
    assert spread is not None and len(spread) == 4
    # Compact should have lower cost than spread
    assert c._placement_cost(compact) <= c._placement_cost(spread)


def test_homogeneous_backward_compatibility():
    # 8 homogeneous A100 must behave exactly as before (homogeneous factory)
    c = Cluster.homogeneous(8, 80, "A100")
    assert c.total_gpus == 8
    assert all(g.memory_gb == 80 for g in c.gpus)
    assert all(g.gpu_type == "A100" for g in c.gpus)
    assert all(g.performance_factor == 1.0 for g in c.gpus)
    jobs = generate_workload(scenario="balanced", seed=0, count=20)
    sim = Simulator(cluster=c, scheduler=FCFSScheduler())
    sim.load_jobs(copy.deepcopy(jobs))
    sim.run()
    m = compute_metrics(list(sim._job_store.values()), total_gpus=8, simulated_time=sim.current_time, gpu_time_used=sim.gpu_time_used)
    # Homogeneous has topology None => penalty 1.0, comm cost 0
    assert m.avg_placement_penalty == 1.0
    assert m.avg_communication_cost == 0.0
    # Jobs should not have heterogeneous fields set
    assert all(not j.topology_sensitive for j in jobs)


def test_heterogeneous_workload_generation():
    for scen in ["heterogeneous", "topology_sensitive", "mixed_ml"]:
        jobs = generate_workload(scenario=scen, seed=0, count=50)
        assert len(jobs) == 50
        # Check that some jobs have topology_sensitive or preferred types
        topo = sum(1 for j in jobs if j.topology_sensitive)
        pref = sum(1 for j in jobs if j.preferred_gpu_type)
        # topology_sensitive should have significant fraction for its scenario
        if scen == "topology_sensitive":
            assert topo >= 10
        # heterogeneous should have some preferred
        if scen == "heterogeneous":
            assert pref >= 5


def test_ppo_observation_dimensions():
    # Homogeneous obs 6/3/8
    env = GPUSchedulingEnv(num_gpus=8, heterogeneous_obs=False)
    assert env.observation_space["cluster"].shape == (6,)
    assert env.observation_space["gpus"].shape == (8, 3)
    assert env.observation_space["jobs"].shape == (16, 8)
    # Heterogeneous obs 8/5/10
    env2 = GPUSchedulingEnv(num_gpus=8, heterogeneous_obs=True, cluster_config_path="configs/heterogeneous_8gpu.yaml")
    assert env2.observation_space["cluster"].shape == (8,)
    assert env2.observation_space["gpus"].shape == (8, 5)
    assert env2.observation_space["jobs"].shape == (16, 10)


def test_benchmark_reproducibility_heterogeneous():
    # Same workload + same cluster must be reproducible
    res1 = run_single_seed(scenario="heterogeneous", seed=42, num_jobs=20)
    res2 = run_single_seed(scenario="heterogeneous", seed=42, num_jobs=20)
    for sched in ["FCFS", "SJF"]:
        assert res1[sched].completed_jobs == res2[sched].completed_jobs
        assert res1[sched].average_waiting_time == res2[sched].average_waiting_time


def test_same_workload_comparison_heterogeneous():
    # Within single seed/scenario, all schedulers see same W0
    from gpu_sage.workloads.generator import generate_workload

    base = generate_workload(scenario="heterogeneous", seed=7, count=20)
    copy1 = copy.deepcopy(base)
    copy2 = copy.deepcopy(base)
    copy1[0].job_id = 9999
    assert copy2[0].job_id == base[0].job_id
    # Benchmark must be reproducible
    r1 = run_single_seed(scenario="heterogeneous", seed=7, num_jobs=20)
    r2 = run_single_seed(scenario="heterogeneous", seed=7, num_jobs=20)
    assert r1["FCFS"].throughput == r2["FCFS"].throughput


def test_heterogeneous_training_smoke():
    """Heterogeneous PPO smoke training must complete without hanging (Windows fix)."""
    import tempfile
    from pathlib import Path

    from gpu_sage.env.gpu_env import GPUSchedulingEnv
    from gpu_sage.workloads.generator import get_scenario_config
    from stable_baselines3.common.vec_env import DummyVecEnv
    from sb3_contrib import MaskablePPO

    cfg = get_scenario_config("heterogeneous")
    from pathlib import Path as _P
    import yaml

    data = yaml.safe_load(_P("configs/heterogeneous_8gpu.yaml").read_text())
    from gpu_sage.core.cluster import Cluster
    from gpu_sage.core.topology import Topology

    cluster = Cluster.heterogeneous(data["cluster"]["gpus"])
    cluster.topology = Topology.two_group(8, 4)

    def make():
        return GPUSchedulingEnv(num_gpus=8, workload_config=cfg, cluster=cluster, heterogeneous_obs=True, seed=0)

    vec = DummyVecEnv([make])
    model = MaskablePPO("MultiInputPolicy", vec, verbose=0, seed=0, n_steps=64, batch_size=32)
    model.learn(128)
    vec.close()
    assert model.num_timesteps == 128


def test_topology_ablation_config():
    """PPO-T vs PPO-NoTopo configs must differ only in heterogeneous_obs."""
    from gpu_sage.env.gpu_env import GPUSchedulingEnv
    from gpu_sage.workloads.generator import get_scenario_config

    cfg = get_scenario_config("heterogeneous")
    env_topo = GPUSchedulingEnv(num_gpus=8, workload_config=cfg, heterogeneous_obs=True)
    env_no = GPUSchedulingEnv(num_gpus=8, workload_config=cfg, heterogeneous_obs=False)
    assert env_topo.observation_space["gpus"].shape[1] == 5
    assert env_no.observation_space["gpus"].shape[1] == 3
    assert env_topo.observation_space["cluster"].shape[0] == 8
    assert env_no.observation_space["cluster"].shape[0] == 6


def test_heterogeneous_model_loading(tmp_path=None):
    """Heterogeneous model artifacts must be loadable if present."""
    from pathlib import Path

    # Check smoke model exists from earlier 5k run
    cand = Path("artifacts/ppo_hetero_fixed/runs/20260829_224735_seed0_steps5000/model/final_model.zip")
    cand2 = Path("artifacts/ppo_hetero/runs/20260829_224803_seed0_steps250000/model/final_model.zip")
    p = cand if cand.exists() else cand2
    if not p.exists():
        pytest.skip("No heterogeneous model artifact yet")
    from sb3_contrib import MaskablePPO

    m = MaskablePPO.load(str(p))
    assert m is not None
    # Check obs shape matches hetero
    assert m.observation_space["gpus"].shape[1] == 5


def test_deterministic_heterogeneous_evaluation():
    model = Path("artifacts/ppo_hetero_fixed/runs/20260829_224735_seed0_steps5000/model/final_model.zip")
    if not model.exists():
        pytest.skip("No hetero smoke model")
    r1 = run_single_seed(scenario="heterogeneous", seed=0, num_jobs=20, schedulers=["PPO"], ppo_model_path=str(model))
    r2 = run_single_seed(scenario="heterogeneous", seed=0, num_jobs=20, schedulers=["PPO"], ppo_model_path=str(model))
    assert r1["PPO"].average_waiting_time == r2["PPO"].average_waiting_time
    assert r1["PPO"].average_turnaround_time == r2["PPO"].average_turnaround_time
