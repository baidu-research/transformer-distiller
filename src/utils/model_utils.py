import copy
import os
import torch
import torch.nn as nn
from typing import Callable
from transformers.models.auto import *


def load_model(name_or_path, auto_type=AutoModel):
    """
        Load a huggingface named model, e.g., bert-large-cased.
        Or
        1) If `name_or_path` is a dir containing config and binary checkpoint,
        load it
        2) If `name_or_path` is a config file, load it and initialize the
        corresponding model from scratch
        3) If `name_or_path` is a dir containing only a config.json, go to 2)
    """
    try:
        model = auto_type.from_pretrained(name_or_path)
    except Exception as e:
        if os.path.isfile(name_or_path):
            model_cfg = AutoConfig.from_pretrained(name_or_path)
        else:
            cfg_file = os.path.join(name_or_path, 'config.json')
            assert os.path.exists(cfg_file)
            model_cfg = AutoConfig.from_pretrained(cfg_file)
        model = auto_type.from_config(model_cfg)
    return model


def freeze_unused_params(model: nn.Module, fwd_func: Callable, **args):
    """
        Create a dummy batch as input, run forward and backward through
        the model to detect parameters that are not involved in back-prop.
        Then freeze these parameters by setting their .require_grad=False

        Inputs:
        model: The nn.Module to be analyzed
        fwd_func: forward function of the following form
            fwd_func(model, dummy_batch, **args): -> loss
        dummy_batch: a minibatch of data to detect parameters not used during
            back-prop
        args: additional args for the fwd_function

        Returns:
        unused_params: parameters not involved in back-prop

        Note this function is not repeating DDP's find_unused_paramters=True.
        The later only detects parameters not involved in the forward pass.
        In fact, for training student models, this funcitonality is not very
        useful. As most likely, a student model is one instance of huggingface
        transformer models. And a call of its foward function will most
        likely involve all of its parameters. Note that we don't encourage
        hacking the defined forward function in huggingface transformers.
        The package should be imported intactly.

        On the other hand, it is common that some parts of the student model
        are not trained. For example, the student has 6 layers but we only ask
        its 5th layer output to match teacher's output. In this case, all
        parameters in the 6th layer are kept unchanged since initialization.
        This function returns these untrained parameters.
    """
    # cache grad for restore
    grad_dict = dict()
    for name, param in model.named_parameters():
        grad_dict[name] = copy.deepcopy(param.grad)
    dummy_len = 512
    dummy_batch = (
            torch.arange(dummy_len, dtype=int)[None, :],
            torch.tensor([dummy_len], dtype=int)
            )
    loss = fwd_func(model, dummy_batch, **args)
    loss.backward()
    frozen_params = []
    for name, param in model.named_parameters():
        if param.grad is None:
            param.requires_grad = False
            frozen_params.append(name)
        else:
            param.grad = grad_dict[name]
    del grad_dict
    return frozen_params
