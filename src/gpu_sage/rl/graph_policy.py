"""Graph-aware PPO policy for topology-aware scheduling.

Design:
  GPU nodes (8) with features [is_free, is_busy, mem_norm, perf_factor, type_id]
  Edges weighted by normalized bandwidth (NVLink 1.0, PCIe 0.22) from Topology.two_group.
  Lightweight 2-layer GCN:
    h0 = Linear(F_in -> H)
    h1 = ReLU( Linear(h0) + Linear( agg(W * h0) ) )
    h2 = ReLU( Linear(h1) + Linear( agg(W * h1) ) )
    pooled = mean(h2)  -> cluster graph embedding
  This is concatenated with:
    - cluster vector (8) -> Linear 8->32
    - jobs masked mean (10 -> 32)
  Total ~ 64+32+32 =128 features for policy/value heads.

Action space unchanged: select waiting job or NOOP (Discrete 17). Simulator places GPUs.

Comparison is fair: same workload/cluster/reward/timesteps as flat MultiInputPolicy.
"""

from __future__ import annotations

from typing import Dict

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy


class GraphFeaturesExtractor(BaseFeaturesExtractor):
    """Dict observation -> latent vector with GNN on GPUs."""

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        # Infer dims from spaces
        gpus_space = observation_space["gpus"]
        cluster_space = observation_space["cluster"]
        jobs_space = observation_space["jobs"]
        # gpus: (N, F) ; cluster: (C,) ; jobs: (max_jobs, J)
        self.num_gpus = int(gpus_space.shape[0])
        self.gpu_feat = int(gpus_space.shape[1])  # 5 for hetero
        self.cluster_dim = int(cluster_space.shape[0])  # 8
        self.job_feat = int(jobs_space.shape[1])  # 10
        self.max_jobs = int(jobs_space.shape[0])

        hidden = 32
        # GNN layers for GPUs
        self.gpu_in = nn.Linear(self.gpu_feat, hidden)
        self.gnn_lin1 = nn.Linear(hidden, hidden)
        self.gnn_agg1 = nn.Linear(hidden, hidden)
        self.gnn_lin2 = nn.Linear(hidden, hidden)
        self.gnn_agg2 = nn.Linear(hidden, hidden)
        self.relu = nn.ReLU()
        # Cluster linear
        self.cluster_lin = nn.Sequential(nn.Linear(self.cluster_dim, 32), nn.ReLU())
        # Jobs: per-job linear then masked mean
        self.job_lin = nn.Sequential(nn.Linear(self.job_feat, 32), nn.ReLU())
        # Final projection to features_dim
        concat_dim = hidden + 32 + 32  # pooled graph + cluster + jobs
        self.proj = nn.Sequential(nn.Linear(concat_dim, features_dim), nn.ReLU())

        # Precompute static adjacency (bandwidth) for 8 GPUs two_group
        # This is the training topology; for generalization we keep same adjacency
        # but the GNN will still operate (topology generalization experiment uses same N).
        self.register_buffer("adj", self._build_adj(self.num_gpus))

        # Parameter count logging
        self._param_count = sum(p.numel() for p in self.parameters())

    def _build_adj(self, n: int) -> torch.Tensor:
        # Build normalized bandwidth adjacency (N,N) as in Topology.two_group
        # NVLink inside group of 4: 1.0, PCIe across: 0.22
        adj = torch.zeros(n, n)
        for i in range(n):
            for j in range(n):
                if i == j:
                    adj[i, j] = 1.0  # self-loop
                else:
                    same_group = (i // 4) == (j // 4)
                    adj[i, j] = 1.0 if same_group else 0.22
        # Row-normalize for stable aggregation
        row_sum = adj.sum(dim=1, keepdim=True).clamp(min=1e-6)
        adj = adj / row_sum
        return adj

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # observations are dict of tensors with batch dim B
        gpus = observations["gpus"]  # (B, N, F)
        cluster = observations["cluster"]  # (B, C)
        jobs = observations["jobs"]  # (B, M, J)
        job_mask = observations["job_mask"]  # (B, M) int

        B = gpus.shape[0]
        N = self.num_gpus

        # --- GNN on GPUs ---
        # gpus: (B,N,F) -> (B,N,hidden)
        h = self.gpu_in(gpus)  # (B,N,hidden)
        # Layer 1: agg = adj @ h  (B,N,hidden)
        # adj is (N,N) -> expand to (B,N,N)
        adj = self.adj.unsqueeze(0).expand(B, -1, -1)  # (B,N,N)
        agg = torch.bmm(adj, h)  # (B,N,hidden)
        h = self.relu(self.gnn_lin1(h) + self.gnn_agg1(agg))
        # Layer 2
        agg2 = torch.bmm(adj, h)
        h = self.relu(self.gnn_lin2(h) + self.gnn_agg2(agg2))
        # Global mean pool over N nodes -> (B,hidden)
        pooled = h.mean(dim=1)  # (B,hidden)

        # --- Cluster ---
        c_emb = self.cluster_lin(cluster)  # (B,32)

        # --- Jobs masked mean ---
        # job_mask: (B,M) int -> float
        mask = job_mask.float().unsqueeze(-1)  # (B,M,1)
        j_emb = self.job_lin(jobs)  # (B,M,32)
        # Masked mean
        j_sum = (j_emb * mask).sum(dim=1)  # (B,32)
        denom = mask.sum(dim=1).clamp(min=1)  # (B,1)
        j_pooled = j_sum / denom  # (B,32)

        concat = torch.cat([pooled, c_emb, j_pooled], dim=1)  # (B, hidden+64)
        out = self.proj(concat)  # (B, features_dim)
        return out

    def param_count(self) -> int:
        return self._param_count


class GraphMaskablePolicy(MaskableActorCriticPolicy):
    """MaskableActorCriticPolicy that uses GraphFeaturesExtractor."""

    def __init__(self, *args, **kwargs):
        # Default to our graph extractor if not supplied
        if "features_extractor_class" not in kwargs:
            kwargs["features_extractor_class"] = GraphFeaturesExtractor
            kwargs["features_extractor_kwargs"] = dict(features_dim=128)
        if "net_arch" not in kwargs:
            kwargs["net_arch"] = dict(pi=[64, 32], vf=[64, 32])
        super().__init__(*args, **kwargs)
