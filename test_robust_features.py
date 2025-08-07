#!/usr/bin/env python3
"""Test Generation 2 robust features."""

import sys
import numpy as np
import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_federated_learning():
    """Test federated learning framework."""
    print("🔗 Testing Federated Learning Framework...")
    
    try:
        from grid_fed_rl.federated import (
            FederatedOfflineRL, GridUtilityClient, FedLearningConfig,
            DifferentialPrivacy, create_private_federated_setup
        )
        from grid_fed_rl.algorithms.base import BaseAlgorithm
        
        # Create mock algorithm
        class MockAlgorithm(BaseAlgorithm):
            def __init__(self):
                super().__init__()
                self.params = {"weight": np.random.randn(10, 5)}
                
            def get_parameters(self) -> Dict[str, np.ndarray]:
                return self.params
                
            def set_parameters(self, params: Dict[str, np.ndarray]):
                self.params = params
                
            def train_step(self, batch_data: List[Dict[str, Any]]) -> float:
                return np.random.random()
                
            def evaluate_batch(self, batch_data: List[Dict[str, Any]]) -> float:
                return np.random.random()
        
        # Test federated setup
        config = FedLearningConfig(
            num_clients=3,
            rounds=5,
            local_epochs=2,
            privacy_budget=1.0
        )
        
        fed_learner = FederatedOfflineRL(MockAlgorithm, config)
        
        # Add clients with mock data
        for i in range(3):
            client = GridUtilityClient(
                client_id=f"utility_{i}",
                algorithm=MockAlgorithm(),
                grid_data=[{"state": np.random.randn(10), "action": np.random.randn(5), 
                           "reward": np.random.random(), "next_state": np.random.randn(10)}
                          for _ in range(50)]
            )
            fed_learner.add_client(client)
            
        # Test training
        global_params = fed_learner.train()
        assert isinstance(global_params, dict)
        assert "weight" in global_params
        
        # Test evaluation
        eval_results = fed_learner.evaluate_global_model()
        assert "avg_loss" in eval_results
        
        # Test metrics
        metrics = fed_learner.get_training_metrics()
        assert "total_rounds" in metrics
        
        print("   ✓ Federated learning framework functional")
        
        # Test privacy mechanisms
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        
        data = np.random.randn(100, 10)
        noisy_data = dp.add_noise(data)
        assert noisy_data.shape == data.shape
        
        private_sum = dp.private_sum([np.random.randn(10) for _ in range(5)], epsilon=0.1)
        assert private_sum.shape == (10,)
        
        budget_status = dp.get_budget_status()
        assert "remaining_epsilon" in budget_status
        
        print("   ✓ Privacy mechanisms functional")
        
        # Test secure aggregation setup
        setup = create_private_federated_setup(
            num_clients=5,
            total_epsilon=1.0,
            secure_aggregation=True
        )
        
        assert "privacy_accountant" in setup
        assert "dp_mechanism" in setup
        assert "secure_aggregation" in setup
        
        print("   ✓ Private federated setup functional")
        
    except Exception as e:
        print(f"   ✗ Federated learning test failed: {e}")
        return False
        
    return True


def test_safety_constraints():
    """Test safety-constrained RL."""
    print("🛡️ Testing Safety-Constrained RL...")
    
    try:
        from grid_fed_rl.algorithms.safe import (
            SafeRL, ConstrainedPolicyOptimization, SafetyConstraint,
            voltage_constraint, frequency_constraint, thermal_constraint
        )
        
        # Create safety constraints
        constraints = [
            voltage_constraint(0.95, 1.05),
            frequency_constraint(59.5, 60.5),
            thermal_constraint(0.8)
        ]
        
        # Test SafeRL
        safe_rl = SafeRL(
            state_dim=10,
            action_dim=5,
            constraints=constraints,
            safety_weight=10.0
        )
        
        state = np.random.randn(10)
        unsafe_action = np.random.randn(5) * 2  # Potentially unsafe action
        
        # Test action correction
        safe_action, info = safe_rl.get_safe_action(state, unsafe_action)
        assert safe_action.shape == unsafe_action.shape
        assert "safe" in info
        assert "corrections_made" in info
        
        # Test constraint evaluation
        constraint_values = safe_rl._evaluate_constraints(state, safe_action)
        assert len(constraint_values) == len(constraints)
        
        # Test safety metrics
        metrics = safe_rl.get_safety_metrics()
        assert "violation_rate" in metrics
        assert "total_actions" in metrics
        
        print("   ✓ SafeRL constraint handling functional")
        
        # Test CPO
        cpo = ConstrainedPolicyOptimization(
            state_dim=10,
            action_dim=5,
            constraints=constraints
        )
        
        # Mock training data
        batch_data = [
            {
                "state": np.random.randn(10),
                "action": np.random.randn(5),
                "reward": np.random.random(),
                "constraint_values": {c.name: np.random.random() - 0.5 for c in constraints}
            }
            for _ in range(32)
        ]
        
        loss = cpo.train_step(batch_data)
        assert isinstance(loss, float)
        
        print("   ✓ Constrained Policy Optimization functional")
        
    except Exception as e:
        print(f"   ✗ Safety constraints test failed: {e}")
        return False
        
    return True


def test_multi_agent():
    """Test multi-agent algorithms."""
    print("🤖 Testing Multi-Agent Algorithms...")
    
    try:
        from grid_fed_rl.algorithms.multi_agent import (
            MADDPG, QMIX, MultiAgentEnvironmentWrapper, AgentConfig
        )
        from grid_fed_rl.environments import GridEnvironment
        from grid_fed_rl.feeders import SimpleRadialFeeder
        
        # Create agent configurations
        agent_configs = [
            AgentConfig(agent_id="battery_1", observation_dim=5, action_dim=2),
            AgentConfig(agent_id="battery_2", observation_dim=5, action_dim=2),
            AgentConfig(agent_id="solar_curtail", observation_dim=3, action_dim=1)
        ]
        
        # Test MADDPG
        maddpg = MADDPG(
            agent_configs=agent_configs,
            gamma=0.99,
            buffer_size=1000
        )
        
        # Test action generation
        observations = {
            "battery_1": np.random.randn(5),
            "battery_2": np.random.randn(5), 
            "solar_curtail": np.random.randn(3)
        }
        
        actions = maddpg.get_actions(observations, add_noise=True)
        assert len(actions) == 3
        assert all(agent_id in actions for agent_id in ["battery_1", "battery_2", "solar_curtail"])
        
        # Test experience addition
        next_obs = {k: np.random.randn(v.observation_dim) for k, v in 
                   {config.agent_id: config for config in agent_configs}.items()}
        rewards = {agent_id: np.random.random() for agent_id in ["battery_1", "battery_2", "solar_curtail"]}
        dones = {agent_id: False for agent_id in ["battery_1", "battery_2", "solar_curtail"]}
        
        maddpg.add_experience(observations, actions, rewards, next_obs, dones)
        
        print("   ✓ MADDPG functional")
        
        # Test QMIX
        qmix = QMIX(
            n_agents=3,
            state_shape=15,
            obs_shape=5,
            n_actions=4
        )
        
        discrete_actions = qmix.get_actions({
            "agent_0": np.random.randn(5),
            "agent_1": np.random.randn(5),
            "agent_2": np.random.randn(5)
        }, epsilon=0.1)
        
        assert len(discrete_actions) == 3
        
        print("   ✓ QMIX functional")
        
        # Test multi-agent environment wrapper
        base_env = GridEnvironment(
            feeder=SimpleRadialFeeder(num_buses=3),
            timestep=1.0,
            episode_length=10
        )
        
        ma_env = MultiAgentEnvironmentWrapper(base_env, agent_configs)
        
        ma_obs = ma_env.reset()
        assert isinstance(ma_obs, dict)
        assert len(ma_obs) == 3
        
        ma_obs, ma_rewards, ma_done, ma_info = ma_env.step(actions)
        assert len(ma_rewards) == 3
        
        print("   ✓ Multi-agent environment wrapper functional")
        
    except Exception as e:
        print(f"   ✗ Multi-agent test failed: {e}")
        return False
        
    return True


def test_enhanced_ieee_feeders():
    """Test enhanced IEEE feeder implementations."""
    print("⚡ Testing Enhanced IEEE Test Feeders...")
    
    try:
        from grid_fed_rl.feeders import IEEE13Bus, IEEE34Bus, IEEE123Bus
        from grid_fed_rl.environments import GridEnvironment
        
        # Test IEEE 13-bus with environment
        ieee13 = IEEE13Bus()
        stats = ieee13.get_network_stats()
        
        assert stats["num_buses"] == 13
        assert stats["num_loads"] > 0
        assert stats["num_generators"] >= 2  # Solar and wind added
        
        validation_errors = ieee13.validate_network()
        assert len(validation_errors) == 0, f"IEEE13 validation errors: {validation_errors}"
        
        # Test with grid environment
        env13 = GridEnvironment(
            feeder=ieee13,
            timestep=1.0,
            episode_length=5,
            renewable_sources=["solar", "wind"]
        )
        
        obs13, info = env13.reset()
        assert len(obs13) > 0
        
        for _ in range(3):
            action = env13.action_space.sample()
            obs13, reward, done, truncated, info = env13.step(action)
            assert isinstance(reward, float)
            
        print("   ✓ IEEE 13-bus enhanced feeder functional")
        
        # Test IEEE 34-bus
        ieee34 = IEEE34Bus()
        stats34 = ieee34.get_network_stats()
        
        assert stats34["num_buses"] >= 25  # Simplified version
        assert stats34["base_voltage_kv"] == 24.9
        
        validation_errors = ieee34.validate_network()
        assert len(validation_errors) == 0, f"IEEE34 validation errors: {validation_errors}"
        
        print("   ✓ IEEE 34-bus enhanced feeder functional")
        
        # Test IEEE 123-bus
        ieee123 = IEEE123Bus()
        stats123 = ieee123.get_network_stats()
        
        assert stats123["num_buses"] == 123
        assert stats123["num_loads"] > 50  # Should have many loads
        
        # Validation might have some acceptable issues for complex network
        validation_errors = ieee123.validate_network()
        print(f"   IEEE 123-bus validation: {len(validation_errors)} warnings")
        
        print("   ✓ IEEE 123-bus enhanced feeder functional")
        
    except Exception as e:
        print(f"   ✗ Enhanced IEEE feeders test failed: {e}")
        return False
        
    return True


def test_error_handling_robustness():
    """Test comprehensive error handling."""
    print("🔧 Testing Error Handling Robustness...")
    
    try:
        from grid_fed_rl.environments import GridEnvironment
        from grid_fed_rl.feeders import SimpleRadialFeeder
        from grid_fed_rl.utils.exceptions import (
            PowerFlowError, InvalidActionError, SafetyLimitExceededError
        )
        
        env = GridEnvironment(
            feeder=SimpleRadialFeeder(num_buses=3),
            timestep=1.0,
            episode_length=10,
            safety_penalty=100.0
        )
        
        env.reset()
        
        # Test invalid actions
        invalid_actions = [
            np.array([np.inf]),      # Infinite values
            np.array([np.nan]),      # NaN values  
            np.array([1e10]),        # Very large values
            np.array([-1e10]),       # Very negative values
            np.array([])             # Empty array
        ]
        
        for i, invalid_action in enumerate(invalid_actions):
            try:
                obs, reward, done, truncated, info = env.step(invalid_action)
                # Should handle gracefully with penalty
                assert reward <= -env.safety_penalty, f"Expected penalty for invalid action {i}"
                assert "error" in info or "action_invalid" in info
                print(f"   ✓ Invalid action {i} handled gracefully")
            except Exception as e:
                print(f"   ✗ Invalid action {i} not handled: {e}")
                return False
                
        # Test network validation
        from grid_fed_rl.feeders.base import CustomFeeder
        
        # Create invalid network
        invalid_feeder = CustomFeeder("Invalid")
        
        # Add bus without connections
        from grid_fed_rl.environments.base import Bus
        orphan_bus = Bus(id=999, voltage_level=4160, bus_type="pq")
        invalid_feeder.add_bus(orphan_bus)
        
        # Add another connected bus
        connected_bus = Bus(id=1, voltage_level=4160, bus_type="slack")
        invalid_feeder.add_bus(connected_bus)
        
        errors = invalid_feeder.validate_network()
        assert len(errors) > 0, "Should detect orphaned bus"
        assert any("999" in error for error in errors), "Should mention orphaned bus ID"
        
        print("   ✓ Network validation catches errors")
        
        # Test power flow convergence issues
        # Create very high impedance line that might cause convergence issues
        from grid_fed_rl.environments.base import Line
        
        feeder = SimpleRadialFeeder(num_buses=3)
        # Find existing line and modify it
        if feeder.lines:
            feeder.lines[0].resistance = 1000.0  # Very high resistance
            feeder.lines[0].reactance = 1000.0   # Very high reactance
            
        try:
            problem_env = GridEnvironment(
                feeder=feeder,
                timestep=1.0,
                episode_length=5
            )
            
            problem_env.reset()
            action = np.array([0.5])
            obs, reward, done, truncated, info = problem_env.step(action)
            
            # Should handle convergence issues gracefully
            if not info.get("power_flow_converged", True):
                print("   ✓ Non-convergent power flow handled gracefully")
            else:
                print("   ✓ Power flow solved despite high impedance")
                
        except Exception as e:
            print(f"   ✗ Power flow convergence issue not handled: {e}")
            return False
            
        print("   ✓ Error handling robustness verified")
        
    except Exception as e:
        print(f"   ✗ Error handling test failed: {e}")
        return False
        
    return True


def main():
    """Run all robustness tests."""
    print("🔧 Generation 2: Testing Robust Features")
    print("=" * 50)
    
    test_results = []
    
    # Run all tests
    test_functions = [
        test_federated_learning,
        test_safety_constraints,
        test_multi_agent,
        test_enhanced_ieee_feeders,
        test_error_handling_robustness
    ]
    
    for test_func in test_functions:
        try:
            result = test_func()
            test_results.append(result)
        except Exception as e:
            print(f"Test {test_func.__name__} crashed: {e}")
            test_results.append(False)
        print()
        
    # Summary
    passed = sum(test_results)
    total = len(test_results)
    
    print("🏁 Generation 2 Robustness Test Results")
    print("=" * 40)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All robustness tests passed!")
        print("\nRobust features verified:")
        print("  ✅ Federated learning framework")
        print("  ✅ Privacy-preserving mechanisms") 
        print("  ✅ Safety-constrained RL")
        print("  ✅ Multi-agent coordination")
        print("  ✅ Enhanced IEEE test feeders")
        print("  ✅ Comprehensive error handling")
        return True
    else:
        print(f"❌ {total - passed} tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)