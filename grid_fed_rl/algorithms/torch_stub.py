"""Minimal torch stub for testing without torch dependency."""

import numpy as np

class Tensor:
    def __init__(self, data):
        self.data = np.array(data)
        
    def detach(self):
        return self
        
    def cpu(self):
        return self
        
    def numpy(self):
        return self.data
        
    def backward(self):
        pass
        
    def item(self):
        return float(self.data)

def tensor(data):
    return Tensor(data)

def FloatTensor(data):
    return Tensor(data)

def zeros(*args, **kwargs):
    return Tensor(np.zeros(*args))

def device(name):
    return name

def load(*args, **kwargs):
    return {}

def save(*args, **kwargs):
    pass

class nn:
    class Module:
        def __init__(self):
            pass
        def parameters(self):
            return []
        def state_dict(self):
            return {}
        def load_state_dict(self, state):
            pass
        def to(self, device):
            return self
    
    class Linear(Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            
    class Sequential(Module):
        def __init__(self, *layers):
            super().__init__()
            
    class ReLU(Module):
        pass
        
    class Tanh(Module):
        pass
        
    class ELU(Module):
        pass

class optim:
    class Adam:
        def __init__(self, params, lr=0.001):
            self.lr = lr
        def zero_grad(self):
            pass
        def step(self):
            pass
        def state_dict(self):
            return {}
        def load_state_dict(self, state):
            pass