"""
RLPolicyNetwork: Actor-Critic network with BC warm-start.

Architecture:
  Shared ResNet18 backbone (from BC checkpoint) → 512-d feature
  Actor Head (reused from BC FC Head): 512→256→128→2 (steering, throttle)
  Critic Head (new): 512→256→1 (state value)
"""

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from model.bc_model import BehavioralCloningModel
from model.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


class RLPolicyNetwork(nn.Module):
    """
    Actor-Critic policy network for PPO with BC warm-start.

    Shares ResNet18 backbone between Actor and Critic heads.
    Actor Head is initialized from BC FC Head weights.
    Critic Head is newly initialized.

    Args:
        pretrained: Whether to use ImageNet pretrained backbone
                    (only used when not loading from BC checkpoint).
    """

    def __init__(self, pretrained: bool = False):
        super().__init__()

        bc_model = BehavioralCloningModel(pretrained=pretrained)
        self.backbone = bc_model.backbone
        self._feature_dim = 512

        # Actor Head (same structure as BC FC Head)
        self.actor_head = nn.Sequential(
            nn.Linear(self._feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

        # Critic Head (new)
        self.critic_head = nn.Sequential(
            nn.Linear(self._feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        # Log std for stochastic action sampling
        self.log_std = nn.Parameter(torch.zeros(2))

    @classmethod
    def from_bc_checkpoint(cls, bc_checkpoint_path: str, device: str = "cpu") -> "RLPolicyNetwork":
        """
        Create RL policy from BC checkpoint (warm-start).

        Copies backbone and actor head weights from BC model.
        Critic head is randomly initialized.

        Args:
            bc_checkpoint_path: Path to BC .pth checkpoint.
            device: Device to load weights to.

        Returns:
            RLPolicyNetwork with BC weights loaded.
        """
        policy = cls(pretrained=False)

        # Load BC model
        bc_model = BehavioralCloningModel(pretrained=False)
        ckpt_mgr = CheckpointManager()
        ckpt_mgr.load(bc_checkpoint_path, bc_model, device=device)

        # Copy backbone weights
        policy.backbone.load_state_dict(bc_model.backbone.state_dict())

        # Copy FC Head → Actor Head
        policy.actor_head.load_state_dict(bc_model.fc_head.state_dict())

        logger.info("RL policy warm-started from BC checkpoint: %s", bc_checkpoint_path)
        return policy.to(device)

    def forward(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through both heads.

        Args:
            state: (batch, 3, 224, 224) normalized tensor.

        Returns:
            steering: (batch, 1) in [-1, 1] via tanh.
            throttle: (batch, 1) in [0, 1] via sigmoid.
            value: (batch, 1) state value estimate.
        """
        features = self.backbone(state)
        features = features.view(features.size(0), -1)

        raw_action = self.actor_head(features)
        steering = torch.tanh(raw_action[:, 0:1])
        throttle = torch.sigmoid(raw_action[:, 1:2])

        value = self.critic_head(features)

        return steering, throttle, value

    def get_action(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Args:
            state: (1, 3, 224, 224) tensor.
            deterministic: If True, return mean action.

        Returns:
            (action_np, log_prob_np, value, entropy)
            action_np: [steering, throttle] numpy array
        """
        features = self.backbone(state)
        features = features.view(features.size(0), -1)

        raw_action = self.actor_head(features)
        value = self.critic_head(features)

        std = torch.exp(self.log_std).clamp(min=1e-6)
        dist = Normal(raw_action, std)

        if deterministic:
            sampled = raw_action
        else:
            sampled = dist.rsample()

        log_prob = dist.log_prob(sampled).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)

        # Apply activation functions
        steering = torch.tanh(sampled[:, 0:1])
        throttle = torch.sigmoid(sampled[:, 1:2])

        action = torch.cat([steering, throttle], dim=-1)
        action_np = action.detach().cpu().numpy().flatten()

        return action_np, log_prob, value, entropy

    def evaluate_actions(
        self, states: torch.Tensor, actions_raw: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.

        Args:
            states: (batch, 3, 224, 224)
            actions_raw: (batch, 2) raw (pre-activation) actions

        Returns:
            (log_probs, values, entropy)
        """
        features = self.backbone(states)
        features = features.view(features.size(0), -1)

        raw_action = self.actor_head(features)
        value = self.critic_head(features)

        std = torch.exp(self.log_std).clamp(min=1e-6)
        dist = Normal(raw_action, std)

        log_prob = dist.log_prob(actions_raw).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)

        return log_prob, value, entropy

    def freeze_backbone(self):
        """Freeze backbone (first 100 episodes)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
