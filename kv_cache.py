import torch

class KVCache:
    def __init__(self, n_layer):
        self.n_layer = n_layer
        self.k_cache = [None for _ in range(n_layer)]
        self.v_cache = [None for _ in range(n_layer)]

    def append(self, layer_idx, new_k, new_v):
        """
        Appends new key and value tensors to the cache for a specific layer.
        """
        if self.k_cache[layer_idx] is None:
            self.k_cache[layer_idx] = new_k
            self.v_cache[layer_idx] = new_v
        else:
            self.k_cache[layer_idx] = torch.cat([self.k_cache[layer_idx], new_k], dim=2)
            self.v_cache[layer_idx] = torch.cat([self.v_cache[layer_idx], new_v], dim=2)

    def get(self, layer_idx):
        """
        Returns the full cached key and value tensors for a specific layer.
        """
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def clear(self):
        """
        Clears the cache, freeing memory.
        """
        self.k_cache = [None for _ in range(self.n_layer)]
        self.v_cache = [None for _ in range(self.n_layer)]
