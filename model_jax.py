import math
import jax
import jax.numpy as jnp
import flax.linen as nn
from dataclasses import dataclass
from typing import Optional, Callable

# -----------------------------------------------------------------------------
# JAX & XLA Concepts (Why JAX is different from PyTorch Eager):
#
# 1. PRNG Key Handling:
# PyTorch uses a global stateful random number generator (e.g., torch.manual_seed).
# JAX requires explicit, functional PRNG state. Every function that needs randomness
# must be passed a PRNG key, and when used, the key must be split to generate new ones.
# This ensures reproducibility in a pure functional paradigm and across distributed devices.
#
# 2. JAX Tracing vs. PyTorch Eager Execution:
# PyTorch eagerly evaluates operations line-by-line as Python code runs.
# JAX uses a tracing mechanism when a function is wrapped with @jax.jit.
# On the FIRST call, JAX passes abstract tracer objects through the Python code,
# recording all operations to build a computational graph (jaxpr). 
# On SUBSEQUENT calls, the Python code is completely bypassed, and the compiled
# graph is executed directly on the accelerator/CPU.
#
# 3. What XLA Does:
# XLA (Accelerated Linear Algebra) takes the JAX computation graph and optimizes it.
# Its biggest win is "kernel fusion" — e.g., instead of doing Matrix Multiply, 
# then writing to memory, reading it back, and doing GELU, XLA fuses them into a 
# single low-level kernel, drastically reducing memory bandwidth bottlenecks.
#
# 4. Static Shapes Matter for XLA:
# XLA compilation is specialized for the exact shape of the input tensors. 
# If you pass a sequence of length 32, it compiles a graph. If you then pass 
# a sequence of length 64, it triggers a RECOMPILATION (because the shape changed).
# To avoid infinite recompilations, inputs to jitted functions must have static,
# bounded shapes during actual inference runs.
# -----------------------------------------------------------------------------

@dataclass
class GPTConfig:
    block_size: int = 64
    vocab_size: int = 50257
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128
    d_ff: int = 512
    dropout: float = 0.0

class CausalSelfAttention(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, x):
        B, T, C = x.shape
        n_head = self.config.n_head
        d_model = self.config.d_model
        
        # Dense layers for Q, K, V
        qkv = nn.Dense(3 * d_model, use_bias=True, name='c_attn')(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        # Reshape to (B, T, n_head, hs) and transpose to (B, n_head, T, hs)
        hs = d_model // n_head
        q = q.reshape(B, T, n_head, hs).transpose((0, 2, 1, 3))
        k = k.reshape(B, T, n_head, hs).transpose((0, 2, 1, 3))
        v = v.reshape(B, T, n_head, hs).transpose((0, 2, 1, 3))

        # Attention scores
        att = jnp.matmul(q, k.transpose((0, 1, 3, 2))) * (1.0 / math.sqrt(hs))

        # Causal mask
        # JAX uses jnp.where for conditional masking.
        # We generate a static boolean mask based on the sequence length T.
        mask = jnp.tril(jnp.ones((T, T)))
        mask = mask.reshape((1, 1, T, T))
        
        # Fill masked positions with negative infinity
        att = jnp.where(mask == 0, -1e9, att)
        
        att = nn.softmax(att, axis=-1)
        
        # Multiply attention weights by V
        y = jnp.matmul(att, v)
        
        # Reshape back to (B, T, C)
        y = y.transpose((0, 2, 1, 3)).reshape(B, T, C)
        
        # Output projection
        y = nn.Dense(d_model, use_bias=True, name='c_proj')(y)
        
        return y

class MLP(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.config.d_ff, use_bias=True, name='c_fc')(x)
        x = nn.gelu(x, approximate=False) # standard gelu
        x = nn.Dense(self.config.d_model, use_bias=True, name='c_proj')(x)
        return x

class TransformerBlock(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, x):
        # Pre-norm architecture
        residual = x
        x = nn.LayerNorm(name='ln_1')(x)
        x = CausalSelfAttention(self.config, name='attn')(x)
        x = residual + x
        
        residual = x
        x = nn.LayerNorm(name='ln_2')(x)
        x = MLP(self.config, name='mlp')(x)
        x = residual + x
        return x

class GPT(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, idx):
        B, T = idx.shape
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is {self.config.block_size}"
        
        # Token and positional embeddings
        tok_emb = nn.Embed(num_embeddings=self.config.vocab_size, features=self.config.d_model, name='wte')(idx)
        
        # JAX positional embeddings: create static array for positions
        pos = jnp.arange(T)[None, :]
        pos_emb = nn.Embed(num_embeddings=self.config.block_size, features=self.config.d_model, name='wpe')(pos)
        
        x = tok_emb + pos_emb
        
        for i in range(self.config.n_layer):
            x = TransformerBlock(self.config, name=f'h_{i}')(x)
            
        x = nn.LayerNorm(name='ln_f')(x)
        
        # LM Head (predicting next token logits)
        # Note: We omit weight tying for simplicity in this specific JAX benchmark implementation
        logits = nn.Dense(self.config.vocab_size, use_bias=False, name='lm_head')(x)
        
        return logits

# -----------------------------------------------------------------------------
# Jitted Apply Function:
# We wrap the Flax model's apply function with jax.jit.
# This tells JAX to trace this specific function and compile it with XLA.
# -----------------------------------------------------------------------------
@jax.jit
def jit_forward(params, idx):
    # This function takes the pure state (params) and inputs (idx)
    # and returns the result, conforming to JAX's pure functional requirement.
    # When jitted, XLA fuses the massive chain of transformer operations.
    model = GPT(GPTConfig())
    return model.apply({'params': params}, idx)
