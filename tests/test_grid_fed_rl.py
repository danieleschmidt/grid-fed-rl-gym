"""
Tests for grid-fed-rl-gym core components.
"""

import pytest
import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grid_fed_rl import GridEnv, PolicyNetwork, LocalRLAgent, FederatedGridCoordinator, GridSimulator


# ── GridEnv tests ─────────────────────────────────────────────────────────────

class TestGridEnv:
    def test_reset_returns_correct_shape(self):
        env = GridEnv(n_buses=5, seed=0)
        obs = env.reset()
        assert obs.shape == (15,), f"Expected (15,), got {obs.shape}"

    def test_step_returns_valid_types(self):
        env = GridEnv(n_buses=4, seed=1)
        env.reset()
        action = np.ones(4) * 0.5
        obs, reward, done, info = env.step(action)
        assert obs.shape == (12,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "grid_loss" in info
        assert "voltage_violations" in info

    def test_reward_is_negative(self):
        """Reward should be negative (cost = losses + penalties)."""
        env = GridEnv(n_buses=5, seed=2)
        env.reset()
        rewards = []
        for _ in range(20):
            action = env.rng.uniform(0, 1, size=5)
            _, r, done, _ = env.step(action)
            rewards.append(r)
            if done:
                break
        assert all(r <= 0 for r in rewards), "Rewards should be non-positive"

    def test_episode_terminates(self):
        env = GridEnv(n_buses=3, seed=3)
        obs = env.reset()
        done = False
        steps = 0
        while not done:
            action = np.ones(3) * 0.5
            obs, _, done, _ = env.step(action)
            steps += 1
            if steps > 300:
                break
        assert done, "Episode should terminate"
        assert steps == env.max_steps

    def test_stability_score_range(self):
        env = GridEnv(n_buses=5, seed=4)
        env.reset()
        score = env.grid_stability_score()
        assert 0.0 <= score <= 1.0

    def test_different_seeds_give_different_grids(self):
        env1 = GridEnv(n_buses=5, seed=10)
        env2 = GridEnv(n_buses=5, seed=20)
        obs1 = env1.reset()
        obs2 = env2.reset()
        assert not np.allclose(obs1, obs2), "Different seeds should yield different initial states"

    def test_action_clipping(self):
        """Out-of-range actions should be clipped, not crash."""
        env = GridEnv(n_buses=3, seed=5)
        env.reset()
        obs, r, done, _ = env.step(np.array([2.0, -1.0, 0.5]))
        assert obs is not None
        assert not np.isnan(r)


# ── PolicyNetwork tests ───────────────────────────────────────────────────────

class TestPolicyNetwork:
    def test_output_shape(self):
        policy = PolicyNetwork(obs_dim=15, act_dim=5)
        obs = torch.zeros(15)
        action, log_prob = policy.get_action(obs)
        assert action.shape == (5,)
        assert log_prob.shape == ()

    def test_action_in_range(self):
        policy = PolicyNetwork(obs_dim=15, act_dim=5)
        for _ in range(10):
            obs = torch.randn(15)
            action, _ = policy.get_action(obs)
            assert torch.all(action >= 0) and torch.all(action <= 1), \
                "Actions should be in [0, 1]"

    def test_deterministic_action_shape(self):
        policy = PolicyNetwork(obs_dim=12, act_dim=4)
        obs = torch.randn(12)
        action = policy.deterministic_action(obs)
        assert action.shape == (4,)

    def test_flat_weights_roundtrip(self):
        policy = PolicyNetwork(obs_dim=10, act_dim=3)
        flat = policy.flat_weights()
        original_flat = flat.clone()
        # Perturb
        policy.load_flat_weights(flat * 2)
        policy.load_flat_weights(original_flat)
        recovered = policy.flat_weights()
        assert torch.allclose(original_flat, recovered), "Weight roundtrip should be exact"

    def test_batch_input(self):
        policy = PolicyNetwork(obs_dim=15, act_dim=5)
        obs_batch = torch.randn(8, 15)
        mean, std = policy(obs_batch)
        assert mean.shape == (8, 5)
        assert std.shape == (5,)  # log_std is per-dim, not per-sample


# ── LocalRLAgent tests ────────────────────────────────────────────────────────

class TestLocalRLAgent:
    def test_train_returns_rewards(self):
        env = GridEnv(n_buses=4, seed=0)
        agent = LocalRLAgent(env, operator_id="test_agent")
        rewards = agent.train(n_episodes=3)
        assert len(rewards) == 3
        assert all(isinstance(r, float) for r in rewards)

    def test_evaluate_returns_valid_scores(self):
        env = GridEnv(n_buses=4, seed=1)
        agent = LocalRLAgent(env, operator_id="eval_test")
        agent.train(n_episodes=5)
        mean_reward, mean_stability = agent.evaluate(n_episodes=2)
        assert isinstance(mean_reward, float)
        assert 0.0 <= mean_stability <= 1.0

    def test_get_set_weights(self):
        env = GridEnv(n_buses=4, seed=2)
        agent = LocalRLAgent(env)
        w1 = agent.get_policy_weights()
        agent.train(n_episodes=3)
        w2 = agent.get_policy_weights()
        # Weights should change after training
        assert not torch.allclose(w1, w2), "Weights should update after training"

    def test_set_policy_weights_changes_behavior(self):
        env = GridEnv(n_buses=4, seed=3)
        agent1 = LocalRLAgent(env, operator_id="a1")
        agent2 = LocalRLAgent(GridEnv(n_buses=4, seed=4), operator_id="a2")

        agent1.train(n_episodes=5)
        w1 = agent1.get_policy_weights()

        # Set agent2's weights to agent1's weights
        agent2.set_policy_weights(w1)
        w2 = agent2.get_policy_weights()

        assert torch.allclose(w1, w2), "Weights should match after set"


# ── FederatedGridCoordinator tests ────────────────────────────────────────────

class TestFederatedGridCoordinator:
    def _make_agents(self, n=3, n_buses=4):
        agents = []
        for i in range(n):
            env = GridEnv(n_buses=n_buses, seed=i * 7)
            agents.append(LocalRLAgent(env, operator_id=f"op_{i}"))
        return agents

    def test_run_round_returns_result(self):
        agents = self._make_agents(n=2, n_buses=4)
        coord = FederatedGridCoordinator(agents, local_episodes_per_round=3)
        result = coord.run_round(round_num=1)
        assert result.round_num == 1
        assert isinstance(result.mean_reward, float)
        assert 0.0 <= result.mean_stability <= 1.0
        assert len(result.operator_rewards) == 2

    def test_run_federation_length(self):
        agents = self._make_agents(n=2, n_buses=3)
        coord = FederatedGridCoordinator(agents, local_episodes_per_round=2)
        results = coord.run_federation(n_rounds=3, verbose=False)
        assert len(results) == 3

    def test_weights_converge(self):
        """After FedAvg, all operators should have same weights as global."""
        agents = self._make_agents(n=3, n_buses=4)
        coord = FederatedGridCoordinator(agents, local_episodes_per_round=2)
        coord.run_round(round_num=1)
        global_w = coord.get_global_policy_weights()
        for agent in agents:
            w = agent.get_policy_weights()
            assert torch.allclose(w, global_w), "Post-round weights should match global"

    def test_fedavg_uniform(self):
        """FedAvg of N identical tensors should equal that tensor."""
        agents = self._make_agents(n=3, n_buses=3)
        coord = FederatedGridCoordinator(agents, local_episodes_per_round=1)
        t = torch.randn(100)
        avg = coord._fedavg([t.clone(), t.clone(), t.clone()])
        assert torch.allclose(avg, t, atol=1e-6)


# ── GridSimulator tests ───────────────────────────────────────────────────────

class TestGridSimulator:
    def test_simulation_runs_and_returns_result(self):
        sim = GridSimulator(
            n_operators=2,
            n_buses_per_operator=4,
            n_local_pretrain_episodes=5,
            seed=0,
        )
        result = sim.run(n_federation_rounds=2, local_episodes_per_round=3, verbose=False)
        assert result.n_operators == 2
        assert result.n_rounds == 2
        assert len(result.local_stability_scores) == 2
        assert len(result.federated_stability_scores) == 2
        assert len(result.per_round_stability) == 2

    def test_stability_scores_in_range(self):
        sim = GridSimulator(n_operators=2, n_buses_per_operator=4, seed=1)
        result = sim.run(n_federation_rounds=2, local_episodes_per_round=3, verbose=False)
        for s in result.local_stability_scores + result.federated_stability_scores:
            assert 0.0 <= s <= 1.0, f"Stability {s} out of [0,1]"

    def test_properties(self):
        sim = GridSimulator(n_operators=2, n_buses_per_operator=4, seed=2)
        result = sim.run(n_federation_rounds=2, local_episodes_per_round=3, verbose=False)
        assert isinstance(result.mean_local_stability, float)
        assert isinstance(result.mean_federated_stability, float)
        assert isinstance(result.stability_improvement, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
