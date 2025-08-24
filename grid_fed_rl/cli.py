"""Command-line interface for grid-fed-rl-gym."""

import argparse
import sys
import json
import os
from typing import List, Optional, Dict, Any

# Graceful numpy import
try:
    import numpy as np
except ImportError:
    # Minimal numpy-like functionality for basic operation
    class MinimalNumPy:
        ndarray = list  # Use list as ndarray replacement
        
        def array(self, data, dtype=None):
            if isinstance(data, list):
                return data
            return [data] if not hasattr(data, '__iter__') else list(data)
        def random(self):
            import random
            return type('obj', (object,), {
                'uniform': lambda low, high: random.uniform(low, high),
                'random': lambda: random.random()
            })()
        def clip(self, x, low, high): return max(low, min(high, x))
    np = MinimalNumPy()

from grid_fed_rl.version import __version__
from grid_fed_rl.environments import GridEnvironment
from grid_fed_rl.feeders import IEEE13Bus, IEEE34Bus, IEEE123Bus, SimpleRadialFeeder

# Optional ML algorithm imports (fail gracefully)
try:
    from grid_fed_rl.algorithms import CQL, IQL, AWR
    from grid_fed_rl.algorithms.base import GridDataset, collect_random_data
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Note: ML algorithms not available. Install torch for full functionality.")


def create_environment(feeder_type: str, **kwargs) -> GridEnvironment:
    """Create environment with specified feeder."""
    feeders = {
        "ieee13": IEEE13Bus,
        "ieee34": IEEE34Bus, 
        "ieee123": IEEE123Bus,
        "simple": SimpleRadialFeeder
    }
    
    if feeder_type not in feeders:
        raise ValueError(f"Unknown feeder type: {feeder_type}. Available: {list(feeders.keys())}")
        
    feeder_class = feeders[feeder_type]
    
    # Enhanced configuration with sensible defaults
    try:
        if feeder_type == "simple":
            feeder = feeder_class(
                num_buses=kwargs.get("num_buses", 5),
                base_power_mw=kwargs.get("base_power_mw", 10.0),
                **{k: v for k, v in kwargs.items() if k not in ["num_buses", "base_power_mw"]}
            )
        else:
            feeder = feeder_class()
            
        env = GridEnvironment(
            feeder=feeder,
            timestep=kwargs.get("timestep", 1.0),
            episode_length=kwargs.get("episode_length", 86400),
            stochastic_loads=kwargs.get("stochastic_loads", True),
            renewable_sources=kwargs.get("renewable_sources", ["solar", "wind"]),
            weather_variation=kwargs.get("weather_variation", True),
            safety_penalty=kwargs.get("safety_penalty", 100.0),
            action_scaling=kwargs.get("action_scaling", True),
            observation_normalization=kwargs.get("observation_normalization", True)
        )
        
        return env
        
    except Exception as e:
        print(f"Warning: Failed to create full environment ({e}), using basic fallback")
        # Create minimal working environment for testing
        from grid_fed_rl.environments.base import BaseGridEnvironment
        return BaseGridEnvironment(
            timestep=kwargs.get("timestep", 1.0),
            episode_length=kwargs.get("episode_length", 86400)
        )


def run_quick_start_command(args) -> None:
    """Execute quick start demonstration."""
    from .quick_start import run_quick_demo
    
    print("Executing Grid-Fed-RL-Gym Quick Start Demo...")
    try:
        result = run_quick_demo()
        
        if result["success"]:
            summary = result["performance_summary"]
            print(f"\n🎉 Demo completed successfully!")
            print(f"📊 System stability: {summary['stability_rate']}")
            print(f"⚡ Performance score: {summary['performance_score']}")
            
            if hasattr(args, 'save_results') and args.save_results:
                import json
                output_file = getattr(args, 'output_file', None) or "quick_demo_results.json"
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"💾 Results saved to {output_file}")
        else:
            print("❌ Demo failed to complete")
            return 1
            
    except Exception as e:
        print(f"❌ Error running quick start: {e}")
        return 1
        
    return 0


def demo_command(args) -> int:
    """Run a quick demonstration."""
    print("Grid-Fed-RL-Gym Demo")
    print("====================\n")
    
    # Create environment
    print("1. Creating IEEE 13-bus test feeder environment...")
    env = create_environment("ieee13", episode_length=100)
    print(f"   - Observation space: {env.observation_space.shape}")
    print(f"   - Action space: {env.action_space.shape}")
    
    # Run short episode
    print("\n2. Running short episode with random policy...")
    obs, info = env.reset()
    total_reward = 0
    
    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 10 == 0:
            print(f"   Step {step}: Reward={reward:.3f}")
            
        if terminated or truncated:
            break
    
    print(f"   Total reward: {total_reward:.2f}")
    
    # Show network stats
    print("\n3. Network statistics:")
    stats = env.feeder.get_network_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\nDemo completed successfully!")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Grid-Fed-RL-Gym: Federated RL for Power Grids",
        prog="grid-fed-rl"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"grid-fed-rl-gym {__version__}"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Demo subcommand
    demo_parser = subparsers.add_parser("demo", help="Run quick demonstration")
    
    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train RL agent")
    train_parser.add_argument("--config", help="Training configuration file (JSON)")
    train_parser.add_argument("--federated", action="store_true", help="Enable federated learning")
    
    # Evaluate subcommand  
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained agent")
    eval_parser.add_argument("--model", required=True, help="Path to trained model")
    eval_parser.add_argument("--env", required=True, help="Environment configuration")
    
    # Simulate subcommand
    sim_parser = subparsers.add_parser("simulate", help="Run grid simulation")
    sim_parser.add_argument("--feeder", required=True, 
                           choices=["ieee13", "ieee34", "ieee123", "simple"],
                           help="Grid feeder type")
    sim_parser.add_argument("--duration", type=int, default=3600, 
                           help="Simulation duration (seconds)")
    
    args = parser.parse_args(argv)
    
    if args.command is None:
        parser.print_help()
        return 1
        
    print(f"grid-fed-rl-gym v{__version__}")
    print(f"Command: {args.command}\n")
    
    try:
        if args.command == "demo":
            return demo_command(args)
        else:
            print(f"Command '{args.command}' implementation coming soon...")
            return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())