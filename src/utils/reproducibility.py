import numpy
import random
import torch
from typing import Optional, Union, List, Dict
from .slurm_utils import global_rank, world_size

def set_rng(seed=0, rng_state_dict: Optional[Union[Dict, List[Dict]]]=None):
    """
        seed: seed for creating global random number generator
            Ignored if `rng_state_dict` is not None
        rng_state_dict: dictonary or a list of dictionaries.
            The dictionary has the following keys:
                `torch_cpu_rng`, `torch_gpu_rng`, `numpy_rng`, `random_rng`,
            to set the torch's cpu rng, gpu rng, numpy's rng and python
            random's rng.
            In case of list of dictionaries, the list length equals the number
            of ranks in a DDP program. And the i-th dictionary are the states
            for the i-th rank.
    """
    if rng_state_dict is None:
        torch.manual_seed(seed)
        numpy.random.seed(seed)
        random.seed(seed)
    elif type(rng_state_dict) is list:
        assert len(rng_state_dict) == world_size
        torch.set_rng_state(rng_state_dict[global_rank]['torch_cpu_rng'])
        torch.cuda.set_rng_state(rng_state_dict[global_rank]['torch_gpu_rng'])
        numpy.random.set_state(rng_state_dict[global_rank]['numpy_rng'])
        random.setstate(rng_state_dict[global_rank]['random_rng'])
    else:
        torch.set_rng_state(rng_state_dict['torch_cpu_rng'])
        torch.cuda.set_rng_state(rng_state_dict['torch_gpu_rng'])
        numpy.random.set_state(rng_state_dict['numpy_rng'])
        random.setstate(rng_state_dict['random_rng'])

def get_rng():
    torch_cpu_state = torch.get_rng_state()
    torch_gpu_state = torch.cuda.get_rng_state()
    numpy_state = numpy.random.get_state()
    random_state = random.getstate()
    random_states = {
        'torch_cpu_rng': torch_cpu_state,
        'torch_gpu_rng': torch_gpu_state,
        'numpy_rng': numpy_state,
        'random_rng': random_state
    }
    return random_states
