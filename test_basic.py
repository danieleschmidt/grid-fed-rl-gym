#!/usr/bin/env python3
"""Basic test of grid-fed-rl-gym core functionality."""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

def test_imports():
    """Test basic imports."""
    print("Testing imports...")
    
    try:
        from grid_fed_rl.version import __version__
        print(f"✓ Version: {__version__}")
        
        from grid_fed_rl.feeders.base import BaseFeeder, FeederParameters
        print("✓ Base feeder classes imported")
        
        from grid_fed_rl.environments.base import BaseGridEnvironment, Bus, Line, Load
        print("✓ Base environment classes imported")
        
        from grid_fed_rl.algorithms.base import BaseAlgorithm
        print("✓ Base algorithm classes imported")
        
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def test_basic_components():
    """Test basic component creation."""
    print("\nTesting basic components...")
    
    try:
        from grid_fed_rl.environments.base import Bus, Line, Load
        
        # Create a bus
        bus = Bus(id=1, voltage_level=12470, bus_type="slack")
        print(f"✓ Bus created: {bus.get_state()}")
        
        # Create a line
        line = Line(id="line_1_2", from_bus=1, to_bus=2, 
                   resistance=0.01, reactance=0.02, rating=5e6)
        print(f"✓ Line created: {line.get_state()}")
        
        # Create a load
        load = Load(id="load_2", bus=2, base_power=1e6, power_factor=0.95)
        print(f"✓ Load created: {load.get_state()}")
        
        return True
    except Exception as e:
        print(f"✗ Component error: {e}")
        return False

def test_feeder():
    """Test feeder creation."""
    print("\nTesting feeder creation...")
    
    try:
        from grid_fed_rl.feeders.base import SimpleRadialFeeder
        
        # Create simple feeder
        feeder = SimpleRadialFeeder(num_buses=3, name="TestFeeder")
        print(f"✓ Simple feeder created")
        
        # Check network stats
        stats = feeder.get_network_stats()
        print(f"✓ Network stats: {stats}")
        
        # Validate network
        errors = feeder.validate_network()
        if errors:
            print(f"⚠ Validation issues: {errors}")
        else:
            print("✓ Network validation passed")
        
        return True
    except Exception as e:
        print(f"✗ Feeder error: {e}")
        return False

def main():
    """Run basic tests."""
    print("Grid-Fed-RL-Gym Basic Test")
    print("==========================")
    
    tests = [
        test_imports,
        test_basic_components,
        test_feeder
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("✓ All basic tests passed! Core functionality working.")
        return 0
    else:
        print("✗ Some tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())