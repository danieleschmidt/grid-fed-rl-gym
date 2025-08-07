#!/usr/bin/env python3
"""Simple CLI test."""

def test_basic_cli():
    try:
        from grid_fed_rl.cli import create_environment, demo_command
        
        # Test environment creation
        env = create_environment("ieee13", timestep=1.0, episode_length=10)
        
        print("✓ Environment creation works")
        
        # Test demo command (just creation, not full execution)
        from io import StringIO
        import sys
        
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            # Create dummy args object
            class DummyArgs:
                pass
            args = DummyArgs()
            
            demo_command(args)
            result = sys.stdout.getvalue()
            if "Demo completed successfully!" in result:
                print("✓ Demo command works")
                return True
            else:
                print("✗ Demo command output unexpected")
                print("Captured output:", result)
                return False
        finally:
            sys.stdout = old_stdout
        
    except Exception as e:
        print(f"✗ CLI test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_cli()
    print(f"CLI Test Result: {'PASS' if success else 'FAIL'}")