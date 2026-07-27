import math
import torch
import numpy as np
from copy import deepcopy

# Utility file to seed rngs
def seed_rng(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)


# A convenience class for sampling from distributions without corrupting global RNG state.
# Subclasses torch.Tensor based on https://discuss.pytorch.org/t/subclassing-torch-tensor/23754/2
class Distribution(torch.Tensor):
    # Init the params of the distribution
    def init_distribution(self, dist_type, **kwargs):
        self.dist_type = dist_type
        self.dist_kwargs = kwargs
        # Use a LOCAL generator so we never corrupt the global RNG state.
        # The global seeds (set by seed_everything in train.py) stay intact.
        seed = kwargs.get('seed')
        if seed is not None:
            self.generator = torch.Generator(device=self.device)
            self.generator.manual_seed(seed)
            self.np_rng = np.random.RandomState(seed)
        else:
            self.generator = None
            self.np_rng = None
        if self.dist_type == 'normal':
            self.mean = kwargs.get('mean', 0.0)
            self.var = kwargs.get('var', 1.0)
            self.std = kwargs.get('std', math.sqrt(self.var) if self.var > 0 else 1.0)
        elif self.dist_type == 'uniform':
            self.low, self.high = kwargs['low'], kwargs['high']
        elif self.dist_type == 'categorical':
            self.num_categories = kwargs['num_categories']
        elif self.dist_type == 'poisson':
            self.lam = kwargs['var']
        elif self.dist_type == 'gamma':
            self.scale = kwargs['var']

    def sample_(self):
        if self.dist_type == 'normal':
            self.normal_(self.mean, self.std, generator=self.generator)
        elif self.dist_type == 'uniform':
            self.uniform_(self.low, self.high, generator=self.generator)
        elif self.dist_type == 'categorical':
            self.random_(0, self.num_categories, generator=self.generator)
        elif self.dist_type == 'poisson':
            type = self.type()
            device = self.device
            rng = self.np_rng if self.np_rng is not None else np.random
            data = rng.poisson(self.lam, self.size())
            self.data = torch.from_numpy(data).type(type).to(device)
        elif self.dist_type == 'gamma':
            type = self.type()
            device = self.device
            rng = self.np_rng if self.np_rng is not None else np.random
            data = rng.gamma(shape=1, scale=self.scale, size=self.size())
            self.data = torch.from_numpy(data).type(type).to(device)
            # return self.variable
        return self

    def get_state(self):
        state = {}
        if hasattr(self, 'generator') and self.generator is not None:
            state['generator'] = self.generator.get_state()
        if hasattr(self, 'np_rng') and self.np_rng is not None:
            state['np_rng'] = self.np_rng.get_state()
        return state

    def set_state(self, state):
        if not state or not isinstance(state, dict):
            return
        if hasattr(self, 'generator') and self.generator is not None and 'generator' in state and state['generator'] is not None:
            gen_state = state['generator']
            if isinstance(gen_state, torch.Tensor):
                gen_state = gen_state.cpu().to(torch.uint8)
            self.generator.set_state(gen_state)
        if hasattr(self, 'np_rng') and self.np_rng is not None and 'np_rng' in state and state['np_rng'] is not None:
            self.np_rng.set_state(state['np_rng'])


    # Overwrite to() method to preserve distribution attributes and Generator device state
    def to(self, *args, **kwargs):
        device_tensor = super().to(*args, **kwargs)
        new_obj = device_tensor.as_subclass(Distribution)
        dist_type = getattr(self, 'dist_type', 'normal')
        dist_kwargs = getattr(self, 'dist_kwargs', {})
        new_obj.dist_type = dist_type
        new_obj.dist_kwargs = dist_kwargs
        # Migrate the local torch Generator to the target device
        if hasattr(self, 'generator') and self.generator is not None:
            target_device = device_tensor.device
            if self.generator.device != target_device:
                new_obj.generator = torch.Generator(device=target_device)
                seed = dist_kwargs.get('seed')
                if seed is not None:
                    new_obj.generator.manual_seed(seed)
            else:
                new_obj.generator = self.generator
        else:
            new_obj.generator = None
        # Share the numpy RNG instance (device-independent)
        new_obj.np_rng = getattr(self, 'np_rng', None)
        if dist_type == 'normal':
            new_obj.mean = getattr(self, 'mean', 0)
            new_obj.var = getattr(self, 'var', 1.0)
            new_obj.std = getattr(self, 'std', math.sqrt(new_obj.var) if new_obj.var > 0 else 1.0)
        elif dist_type == 'uniform':
            new_obj.low = getattr(self, 'low', 0)
            new_obj.high = getattr(self, 'high', 1)
        elif dist_type == 'categorical':
            new_obj.num_categories = getattr(self, 'num_categories', 1)
        elif dist_type == 'poisson':
            new_obj.lam = getattr(self, 'lam', 1)
        elif dist_type == 'gamma':
            new_obj.scale = getattr(self, 'scale', 1)
        return new_obj

# Convenience function to prepare a z vector
def prepare_z_dist(G_batch_size, dim_z, device='cuda', seed=0, num_tokens=32):
    z_ = Distribution(torch.randn(G_batch_size, num_tokens, dim_z, requires_grad=False))
    z_.init_distribution('normal', mean=0, var=1.0, seed=seed)
    z_ = z_.to(device)
    return z_

# Convenience function to prepare a z vector
def prepare_y_dist(G_batch_size, nclasses, device='cuda', seed=0):
    y_ = Distribution(torch.zeros(G_batch_size, requires_grad=False))
    y_.init_distribution('categorical', num_categories=nclasses, seed=seed)
    y_ = y_.to(device, torch.int64)
    return y_