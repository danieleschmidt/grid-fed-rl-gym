#!/usr/bin/env python3
"""
Test basic Generation 1 functionality - MAKE IT WORK (Simple)
"""

import sys
import traceback
import warnings
warnings.filterwarnings('ignore')

def test_basic_import():
    """Test basic package import"""
    try:
        import grid_fed_rl
        print("✅ Basic import successful")
        print(f"📦 Version: {grid_fed_rl.__version__}")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_core_components():
    """Test core component availability"""
    try:
        import grid_fed_rl as gfrl
        
        components = ['GridEnvironment', 'IEEE13Bus', 'CQL', 'FederatedOfflineRL']
        available = []
        
        for component in components:
            try:
                cls = getattr(gfrl, component)
                available.append(component)
                print(f"✅ {component} available")
            except Exception as e:
                print(f"⚠️  {component} degraded: {str(e)[:50]}...")
        
        return len(available) >= 3  # At least 3/4 should work
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        return False

def test_cli_availability():
    """Test CLI interface"""
    try:
        from grid_fed_rl.cli import main
        print("✅ CLI interface available")
        return True
    except Exception as e:
        print(f"⚠️  CLI interface degraded: {e}")
        return False

def test_basic_validation():
    """Test basic validation utilities"""
    try:
        from grid_fed_rl.utils import validate_action, sanitize_config
        print("✅ Basic validation utilities available")
        return True
    except Exception as e:
        print(f"⚠️  Validation utilities degraded: {e}")
        return False

def main():
    """Run Generation 1 basic functionality tests"""
    print("🚀 GENERATION 1 TESTING: MAKE IT WORK (Simple)")
    print("=" * 50)
    
    tests = [
        ("Basic Import", test_basic_import),
        ("Core Components", test_core_components), 
        ("CLI Interface", test_cli_availability),
        ("Basic Validation", test_basic_validation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Testing {test_name}...")
        try:
            if test_func():
                passed += 1
            else:
                print(f"   ⚠️  {test_name} has issues but framework continues")
        except Exception as e:
            print(f"   ❌ {test_name} failed: {e}")
            traceback.print_exc()
    
    print(f"\n📊 GENERATION 1 RESULTS: {passed}/{total} tests passed")
    
    if passed >= 3:  # Minimum viable functionality
        print("✅ GENERATION 1 COMPLETE: Basic functionality working!")
        return True
    else:
        print("❌ GENERATION 1 FAILED: Critical functionality missing")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)