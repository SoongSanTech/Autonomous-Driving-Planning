"""
Tests for RLPolicyNetwork.

Property 12: BC→RL warm-start weight preservation.
"""

import pytest
import torch

from model.bc_model import BehavioralCloningModel
from model.checkpoint import CheckpointManager
from model.rl_policy import RLPolicyNetwork


@pytest.fixture
def bc_checkpoint(tmp_path):
    """Create a BC checkpoint for warm-start testing."""
    model = BehavioralCloningModel(pretrained=False)
    optimizer = torch.optim.Adam(model.parameters())
    mgr = CheckpointManager(str(tmp_path))
    return mgr.save(model, optimizer, epoch=10, metrics={"val_loss": 0.05}, model_type="bc")


class TestRLPolicyBasic:
    """Basic RL policy tests."""

    def test_forward_shapes(self):
        policy = RLPolicyNetwork(pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        steering, throttle, value = policy(x)
        assert steering.shape == (2, 1)
        assert throttle.shape == (2, 1)
        assert value.shape == (2, 1)

    def test_output_ranges(self):
        policy = RLPolicyNetwork(pretrained=False)
        policy.eval()
        x = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            steering, throttle, value = policy(x)
        assert steering.min() >= -1.0
        assert steering.max() <= 1.0
        assert throttle.min() >= 0.0
        assert throttle.max() <= 1.0

    def test_get_action_deterministic(self):
        policy = RLPolicyNetwork(pretrained=False)
        policy.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            action, log_prob, value, entropy = policy.get_action(x, deterministic=True)
        assert action.shape == (2,)
        assert -1.0 <= action[0] <= 1.0
        assert 0.0 <= action[1] <= 1.0

    def test_get_action_stochastic(self):
        policy = RLPolicyNetwork(pretrained=False)
        x = torch.randn(1, 3, 224, 224)
        action, log_prob, value, entropy = policy.get_action(x, deterministic=False)
        assert action.shape == (2,)


class TestRLPolicyWarmStart:
    """BC→RL warm-start tests."""

    def test_from_bc_checkpoint(self, bc_checkpoint):
        policy = RLPolicyNetwork.from_bc_checkpoint(bc_checkpoint)
        x = torch.randn(1, 3, 224, 224)
        steering, throttle, value = policy(x)
        assert steering.shape == (1, 1)

    def test_backbone_weights_match(self, bc_checkpoint):
        """Property 12: Backbone weights must match BC checkpoint."""
        bc_model = BehavioralCloningModel(pretrained=False)
        mgr = CheckpointManager()
        mgr.load(bc_checkpoint, bc_model)

        policy = RLPolicyNetwork.from_bc_checkpoint(bc_checkpoint)

        for (name_bc, param_bc), (name_rl, param_rl) in zip(
            bc_model.backbone.named_parameters(),
            policy.backbone.named_parameters(),
        ):
            assert torch.equal(param_bc, param_rl), f"Mismatch in backbone: {name_bc}"

    def test_actor_head_weights_match(self, bc_checkpoint):
        """Property 12: Actor head weights must match BC FC head."""
        bc_model = BehavioralCloningModel(pretrained=False)
        mgr = CheckpointManager()
        mgr.load(bc_checkpoint, bc_model)

        policy = RLPolicyNetwork.from_bc_checkpoint(bc_checkpoint)

        for (name_bc, param_bc), (name_rl, param_rl) in zip(
            bc_model.fc_head.named_parameters(),
            policy.actor_head.named_parameters(),
        ):
            assert torch.equal(param_bc, param_rl), f"Mismatch in actor head: {name_bc}"


class TestRLPolicyFreeze:
    """Backbone freeze/unfreeze tests."""

    def test_freeze(self):
        policy = RLPolicyNetwork(pretrained=False)
        policy.freeze_backbone()
        for p in policy.backbone.parameters():
            assert not p.requires_grad
        # Actor and critic should still be trainable
        for p in policy.actor_head.parameters():
            assert p.requires_grad
        for p in policy.critic_head.parameters():
            assert p.requires_grad

    def test_unfreeze(self):
        policy = RLPolicyNetwork(pretrained=False)
        policy.freeze_backbone()
        policy.unfreeze_backbone()
        for p in policy.backbone.parameters():
            assert p.requires_grad
