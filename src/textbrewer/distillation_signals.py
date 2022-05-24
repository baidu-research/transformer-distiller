"""
    Implement distillation signals other than the off-the-shelf
    hidden states and attention matrices. For example:
    KK, QQ, VV matrix used in minilm2

    These signals are usually functions of some very "intermediate" variables.
    Thus cannot be simply read out by calling the default forward pass of
    huggingface's transformers.

    We achieve this by registering forward hook functions for certain modules
    in huggingface's transformers.
"""
import torch
import torch.nn as nn
from typing import Callable, Optional


class extractor:
    def __init__(self, module: nn.Module):
        # register the forward hook function to this module
        self.module = module
        # Cache of hook's readout
        self.cache = None
        def hook_function(module, input, output):
            self.cache = output
        self.hook = self.module.register_forward_hook(hook_function)

    def compute(self, **kwargs):
        # A child class can manipulate self.cache
        pass

    def empty_cache(self):
        del self.cache
        self.cache = None

    def release(self):
        self.hook.remove()
        self.empty_cache()


class token_pairwise_similarity(extractor):
    """
        Suppose a module outputs a tensor of size:
            batch x seq_length x dim1 x dim2 x ...
        This hook does the following:
        1. reshape into batch x seq_length x (D = dim1*dim2*...)
        2. slice D into `segment` chunks, assume D is divisible by `segment`
        3. for each chunk, compute the batch x seq_length x seq_length gram
            matrix, and normalize as specified
        4. store the result as batch x segments x seq_length x seq_length
    """
    def __init__(self,
                 module: nn.Module,
                 segments: int,
                 norm_func: Optional[Callable]=None):
        super(token_pairwise_similarity, self).__init__(module)
        self.segments = segments
        self.norm_func = norm_func

    def compute(self):
        self.cache = self.cache.view(self.cache.shape[:2] + (-1,))
        B, L, d = self.cache.shape
        assert d % self.segments == 0
        dr = d // self.segments
        self.cache = self.cache.view((B, L, self.segments, dr))
        self.cache = self.cache.permute((0, 2, 1, 3))
        # B x segments x L x L
        self.cache = torch.matmul(self.cache, self.cache.transpose(-1, -2))
        if self.norm_func is not None:
            self.cache = self.norm_func(self.cache)
