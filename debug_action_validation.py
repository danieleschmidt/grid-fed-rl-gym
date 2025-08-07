#!/usr/bin/env python3
"""Debug action validation."""

import numpy as np

def debug_validation():
    from grid_fed_rl.environments.base import Box
    from grid_fed_rl.utils.validation import validate_action
    from grid_fed_rl.utils.exceptions import InvalidActionError
    
    action_space = Box(low=-1.0, high=1.0, shape=(1,))
    
    # Test infinite action
    try:
        invalid_action = np.array([np.inf])
        print(f"Testing action: {invalid_action}")
        validated = validate_action(invalid_action, action_space)
        print(f"Validation passed unexpectedly: {validated}")
        return False
    except InvalidActionError as e:
        print(f"Validation correctly caught error: {e}")
        return True
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = debug_validation()
    print(f"Debug result: {success}")