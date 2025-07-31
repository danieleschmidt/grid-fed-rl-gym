"""Command-line interface for grid-fed-rl-gym."""

import argparse
import sys
from typing import List, Optional

from grid_fed_rl.version import __version__


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
    
    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train RL agent")
    train_parser.add_argument("--config", required=True, help="Training configuration file")
    train_parser.add_argument("--federated", action="store_true", help="Enable federated learning")
    
    # Evaluate subcommand  
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained agent")
    eval_parser.add_argument("--model", required=True, help="Path to trained model")
    eval_parser.add_argument("--env", required=True, help="Environment configuration")
    
    # Simulate subcommand
    sim_parser = subparsers.add_parser("simulate", help="Run grid simulation")
    sim_parser.add_argument("--feeder", required=True, help="Grid feeder configuration")
    sim_parser.add_argument("--duration", type=int, default=86400, help="Simulation duration (seconds)")
    
    args = parser.parse_args(argv)
    
    if args.command is None:
        parser.print_help()
        return 1
        
    print(f"grid-fed-rl-gym v{__version__}")
    print(f"Command: {args.command}")
    print("Implementation coming soon...")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())