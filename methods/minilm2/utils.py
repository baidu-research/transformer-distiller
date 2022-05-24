import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union


# Choices of normalizing the KK, QQ, VV matrices
normalizer_types = ['sqrt_dim', 'unit_length', 'unit_std']

# Choices of losses when matching KK, QQ, VV matrices
minilm2_loss_types = ['kl_div', 'mse']

def get_module(model: nn.Module, q_k_v: str, layer_index: int):
    """
        Locate the query/key/value module in a huggingface transformer model

        model: a huggingface transformer model
        q_k_v: one of {'query', 'key', 'value'}
        layer_index: index of the transformer layer to look into, starting from 1
    """
    # locate the stacked transformer layers
    located = False
    for module in model.modules():
        if isinstance(module, nn.ModuleList):
            located = True
            break
    assert located, 'cannot locate stack of transformer layers'
    layer = module[layer_index-1]

    located = False
    for name, module in layer.named_modules():
        if name.endswith(f'.{q_k_v}'):
            located = True
            break
    if located:
        return module
    else:
        raise NotImplementedError(f'{q_k_v} module cannot be found. '
                                  'minilm2 is not implemented this model yet!')

def norm_function(sim_matrix: torch.Tensor,
                  normalizer: Optional[Union[int, float, str]]=None,
                  log_softmax: Optional[bool]=False):
    B, segs, seq_len = sim_matrix.shape[:3]
    assert sim_matrix.shape[3] == seq_len

    if isinstance(normalizer, float) or isinstance(normalizer, int):
        sim_matrix /= normalizer
    elif normalizer == 'unit_length':
        sim_matrix /= torch.norm(sim_matrix, dim=-1, keepdim=True)
    elif normalizer == 'unit_std':
        sim_matrix /= torch.std(sim_matrix, dim=-1, unbiased=False, keepdim=True)
    elif normalizer is not None:
        raise NotImplementedError('Unknown normalizer type for pairwise similarity matrix')

    if log_softmax:
        sim_matrix = F.log_softmax(sim_matrix, -1)
    return sim_matrix


def minilm2_loss(sim_T: torch.tensor, sim_S: torch.tensor, loss_type='kl_div'):
    assert sim_T.shape == sim_S.shape and len(sim_T.shape) == 4
    if loss_type == 'kl_div':
        return F.kl_div(sim_S, sim_T, reduction='none',
                        log_target=True).sum(-1).mean()
    elif loss_type == 'mse':
        return ((sim_T - sim_S)**2).sum(-1).mean()
