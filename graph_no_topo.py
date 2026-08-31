"""Graph features extractor without topology (for PPO-NoTopo ablation)."""

from __future__ import annotations

from typing import Dict

import gymnasium as gym
import torch
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy


class GraphFeaturesExtractorNoTopo(BaseFeaturesExtractor):
    """Graph encoder WITHOUT topology - treats GPUs as independent (diagonal adjacency).
    
    This is the "PPO-NoTopo" variant: the GNN has no inter-GPU communication,
    effectively equivalent to flattening the graph observation.
    """

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
        # GNN layers - but with identity adjacency (no aggregation)
        # Instead of adj @ h, we just use h directly (each GPU independent)
        self.gpu_in = nn.Linear(self.gpu_feat, hidden)
        # No gnn_agg - we just use the individual GPU features
        # Cluster linear
        self.cluster_lin = nn.Sequential(nn.Linear(self.cluster_dim, 32), nn.ReLU())
        # Jobs: per-job linear then masked mean
        self.job_lin = nn.Sequential(nn.Linear(self.job_feat, 32), nn.ReLU())
        # Final projection to features_dim
        # Without GNN: just cluster + jobs pooled = 32 + 32 = 64
        # With projection to 128
        concat_dim = hidden + 32 + 32  # pooled graph + cluster + jobs
        self.proj = nn.Sequential(nn.Linear(concat_dim, features_dim), nn.ReLU())

        # Parameter count logging
        self._param_count = sum(p.numel() for p in self.parameters())

        # Register buffer for identity adjacency (but not used in forward)
        self.register_buffer("adj", torch.eye(self.num_gpus))

        # Parameter count logging
        self._param_count = sum(p.numel() for p in self.parameters())

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # observations are dict of tensors with batch dim B
        gpus = observations["gpus"]  # (B, N, F)
        cluster = observations["cluster"]  # (B, C)
        jobs = observations["jobs"]  # (B, M, J)
        job_mask = observations["job_mask"]  # (B, M) int

        B = gpus.shape[0]
        N = self.num_gpus

        # --- GNN on GPUs WITHOUT topology ---
        # Instead of adj @ h, just use h directly (each GPU independent)
        h = self.gpu_in(gpus)  # (B,N,hidden)

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

        concat = torch.cat([h.mean(dim=1), c_emb, j_pooled], dim=1)  # (B, hidden+64)
        out = self.proj(concat)  # (B, features_dim)
        return out

    def param_count(self) -> int:
        return self._param_count


class GraphMaskablePolicyNoTopo(MaskableActorCriticPolicy):
    """MaskableActorCriticPolicy that uses GraphFeaturesExtractorNoTopo."""

    def __init__(self, *args, **kwargs):
        # Default to our no-topo graph extractor if not supplied
        if "features_extractor_class" not in kwargs:
            kwargs["features_extractor_class"] = GraphFeaturesExtractorNoTopo
            kwargs["features_extractor_kwargs"] = dict(features_dim=128)
        if "net_arch" not in kwargs:
            kwargs["net_arch"] = dict(pi=[64, 32], vf=[64, 32])
        super().__init__(*args, **kwargs)