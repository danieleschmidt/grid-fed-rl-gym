"""Quick start module for immediate functionality demonstration."""

from typing import Dict, Any, List, Optional
import time
import json

from .environments.base import BaseGridEnvironment
from .feeders.base import Bus, Line, Load

class QuickStartDemo:
    """Simplified demonstration of core functionality."""
    
    def __init__(self):
        self.env = None
        self.results = []
        
    def create_minimal_grid(self) -> Dict[str, Any]:
        """Create a minimal 3-bus system for demonstration."""
        
        # Create buses
        buses = [
            Bus(id=1, voltage_level=12.47e3, bus_type="slack"),
            Bus(id=2, voltage_level=12.47e3, bus_type="pq"), 
            Bus(id=3, voltage_level=12.47e3, bus_type="pq")
        ]
        
        # Create lines
        lines = [
            Line(id="line_1_2", from_bus=1, to_bus=2, resistance=0.01, reactance=0.02, rating=5e6),
            Line(id="line_2_3", from_bus=2, to_bus=3, resistance=0.015, reactance=0.025, rating=5e6)
        ]
        
        # Create loads
        loads = [
            Load(id="load_2", bus=2, base_power=1.0e6, power_factor=0.95, active_power=1.0e6),
            Load(id="load_3", bus=3, base_power=1.5e6, power_factor=0.92, active_power=1.5e6)
        ]
        
        # Manually create data dictionary since to_dict might not be implemented
        grid_data = {
            "name": "QuickStart3Bus",
            "buses": [{"id": b.id, "voltage_level": b.voltage_level, "bus_type": b.bus_type} for b in buses],
            "lines": [{"id": l.id, "from_bus": l.from_bus, "to_bus": l.to_bus, "resistance": l.resistance, "reactance": l.reactance, "rating": l.rating} for l in lines], 
            "loads": [{"id": load.id, "bus": load.bus, "base_power": load.base_power, "active_power": load.active_power, "power_factor": load.power_factor} for load in loads],
            "total_load_mw": sum(load.active_power for load in loads) / 1e6,
            "base_voltage_kv": 12.47
        }
        
        return grid_data
        
    def simulate_basic_operation(self, steps: int = 10) -> List[Dict[str, Any]]:
        """Simulate basic grid operation."""
        
        grid = self.create_minimal_grid()
        simulation_results = []
        
        print(f"Starting {steps}-step simulation of {grid['name']}...")
        print(f"Total Load: {grid['total_load_mw']:.1f} MW")
        
        for step in range(steps):
            # Simple power flow simulation
            import random
            
            # Simulate voltage variations
            voltages = {}
            for bus_data in grid["buses"]:
                bus_id = bus_data["id"]
                if bus_data["bus_type"] == "slack":
                    voltages[bus_id] = 1.0  # Reference voltage
                else:
                    # Simple voltage drop model
                    base_voltage = 0.98 - (bus_id - 1) * 0.01
                    voltages[bus_id] = base_voltage + 0.02 * (random.random() - 0.5)
            
            # Simulate line flows
            line_flows = {}
            for line_data in grid["lines"]:
                line_id = f"{line_data['from_bus']}-{line_data['to_bus']}"
                base_flow = grid['total_load_mw'] * 0.5
                line_flows[line_id] = base_flow + 0.1 * base_flow * (random.random() - 0.5)
            
            # Check constraints
            violations = []
            for bus_id, voltage in voltages.items():
                if voltage < 0.95 or voltage > 1.05:
                    violations.append(f"Bus {bus_id} voltage: {voltage:.3f} pu")
            
            step_result = {
                "step": step,
                "time": time.time(),
                "voltages": voltages,
                "line_flows": line_flows,
                "violations": violations,
                "system_stable": len(violations) == 0
            }
            
            simulation_results.append(step_result)
            
            if step % 5 == 0:
                print(f"Step {step}: {'✓ Stable' if step_result['system_stable'] else '⚠ Violations'}")
                
        self.results = simulation_results
        return simulation_results
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary from simulation results."""
        
        if not self.results:
            return {"error": "No simulation results available"}
            
        # Calculate statistics
        total_steps = len(self.results)
        stable_steps = sum(1 for r in self.results if r["system_stable"])
        stability_rate = stable_steps / total_steps
        
        # Voltage statistics
        all_voltages = []
        for result in self.results:
            all_voltages.extend(result["voltages"].values())
            
        min_voltage = min(all_voltages)
        max_voltage = max(all_voltages)
        avg_voltage = sum(all_voltages) / len(all_voltages)
        
        # Line flow statistics  
        all_flows = []
        for result in self.results:
            all_flows.extend(result["line_flows"].values())
            
        avg_flow = sum(all_flows) / len(all_flows)
        max_flow = max(all_flows)
        
        summary = {
            "simulation_steps": total_steps,
            "stability_rate": f"{stability_rate:.1%}",
            "voltage_stats": {
                "min_pu": f"{min_voltage:.3f}",
                "max_pu": f"{max_voltage:.3f}", 
                "avg_pu": f"{avg_voltage:.3f}"
            },
            "power_flow_stats": {
                "avg_mw": f"{avg_flow:.2f}",
                "peak_mw": f"{max_flow:.2f}"
            },
            "constraint_violations": total_steps - stable_steps,
            "performance_score": f"{stability_rate * 100:.1f}/100"
        }
        
        return summary
        
    def run_complete_demo(self) -> Dict[str, Any]:
        """Run complete demonstration workflow."""
        
        print("=" * 50)
        print("GRID-FED-RL-GYM QUICK START DEMO")
        print("=" * 50)
        
        start_time = time.time()
        
        # 1. Create grid
        print("\n1. Creating minimal power grid...")
        grid = self.create_minimal_grid()
        print(f"✓ Created {grid['name']} with {len(grid['buses'])} buses")
        
        # 2. Run simulation
        print("\n2. Running power flow simulation...")
        results = self.simulate_basic_operation(20)
        
        # 3. Generate summary
        print("\n3. Analyzing results...")
        summary = self.get_performance_summary()
        
        execution_time = time.time() - start_time
        
        print("\n" + "=" * 50)
        print("SIMULATION COMPLETE")
        print("=" * 50)
        print(f"Total execution time: {execution_time:.2f}s")
        print(f"System stability: {summary['stability_rate']}")
        print(f"Performance score: {summary['performance_score']}")
        print(f"Voltage range: {summary['voltage_stats']['min_pu']} - {summary['voltage_stats']['max_pu']} pu")
        
        complete_results = {
            "demo_metadata": {
                "execution_time_seconds": execution_time,
                "timestamp": time.time(),
                "grid_config": grid
            },
            "simulation_results": results,
            "performance_summary": summary,
            "success": True
        }
        
        return complete_results

def run_quick_demo() -> Dict[str, Any]:
    """Standalone function to run quick demonstration."""
    demo = QuickStartDemo()
    return demo.run_complete_demo()

if __name__ == "__main__":
    # Allow direct execution
    results = run_quick_demo()
    print(f"\nDemo completed successfully: {results['success']}")