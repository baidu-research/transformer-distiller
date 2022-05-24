import json
import os
import pathlib
from typing import Union, List, Optional, Dict, Callable
from .distillation_signals import extractor
from .presets import *

class Config:
    """Base class for TrainingConfig and DistillationConfig."""
    def __init__(self,**kwargs):
        pass

    @classmethod
    def from_json_file(cls, json_filename):
        """Construct configurations from a json file."""
        with open(json_filename,'r') as f:
            json_data = json.load(f)
        return cls.from_dict(json_data)

    @classmethod
    def from_dict(cls, dict_object):
        """Construct configurations from a dict."""
        config = cls(**dict_object)
        return config

    def __str__(self):
        str = ""
        for k,v in self.__dict__.items():
            str += f"{k} : {v}\n"
        return str

    def __repr__(self):
        classname = self.__class__.__name__
        return classname +":\n"+self.__str__()


class TrainingConfig(Config):
    """
    Configurations related to general model training.
    """
    def __init__(self,
                 num_steps, # max num training steps
                 output_dir, # dir to save ckpts, logs, etc.
                 max_grad_norm=-1, # if >1, clip gradient norm
                 gradient_accumulation_steps = 1,
                 ckpt_steps = None, # checkpoint and run callbacks every ckpt_steps
                 log_steps = None, # print training losses and log tensorboard event
                 save_ini = False, # save checkpoint and run callbacks at 0-th step
                 fp16 = False,
                 fp16_opt_level = 'O1',
                 data_parallel = False, # suggest False, use DDP instead
                 local_rank = 0, # rank in the local node
                 rank = 0, # global rank over all nodes
                 world_size = 1
                 ):
        super(TrainingConfig, self).__init__()

        self.num_steps = num_steps
        self.output_dir = output_dir
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.ckpt_steps = ckpt_steps
        self.log_steps = log_steps
        self.save_ini = save_ini
        self.fp16 = fp16
        self.fp16_opt_level = fp16_opt_level
        self.data_parallel = data_parallel
        self.local_rank = local_rank
        self.rank = rank
        self.world_size = world_size

        # Create log_dir for writing tensorboard files
        # Note: in distributed case, we suggest one log dir for each rank.
        self.log_dir = os.path.join(output_dir, 'rank-{}'.format(rank))
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)


class IntermediateMatch:
    def __init__(self,layer_T: Union[int,List[int]], layer_S: Union[int,List[int]],
                 weight: float, loss: str, feature: str, proj: Optional[List] = None):
        self.layer_T = layer_T
        self.layer_S = layer_S
        self.feature = feature
        self.weight = weight
        self.loss = loss
        self.proj = proj
        assert feature in FEATURES
        if proj:
            assert proj[0] in PROJ_MAP.keys()
            assert type(proj[1]) is int and type(proj[2]) is int
            if len(proj)==3:
                self.proj.append(dict())   # ['linear', dim_S, dim_T, {...}]
            else:
                assert type(proj[3]) is dict

    def __str__(self):
        str = ""
        for k,v in self.__dict__.items():
            str += f"{k} : {v}, "
        return str[:-2]

    def __repr__(self):
        classname = self.__class__.__name__
        return '\n'+classname +": "+self.__str__()

    @classmethod
    def from_dict(cls,dict_object):
        if dict_object is None:
            return None
        else:
            return cls(**dict_object)


class SignalMatch:
    """
        For matches that use `non-trivial` distillation signals. These signals
        cannot be directly read out from a forward pass of huggingface
        transformer
    """
    def __init__(self, signal_T: extractor, signal_S: extractor,
                 weight: float, loss: Callable, name_tag: Optional[str]=None):
        self.signal_T = signal_T
        self.signal_S = signal_S
        self.loss = loss
        self.weight = weight
        self.name_tag = name_tag

    def empty_cache(self):
        self.signal_T.empty_cache()
        self.signal_S.empty_cache()

    def release(self):
        self.signal_T.release()
        self.signal_S.release()


class DistillationConfig(Config):
    """
    Configurations related to distillation methods. It defines the total loss to be optimized:
    
    .. math::

        \mathcal{L}_{total}=  \mathcal{L}_{KD} * w_{KD} + \mathcal{L}_{hl} * w_{hl} + sum(\\textrm{intermediate_losses}) + sum(\\textrm{signal_match_losses})
    
    where

        * :math:`\mathcal{L}_{KD}` is the KD loss on logits, :math:`w_{KD}` is its weight;
        * :math:`\mathcal{L}_{hl}` is the sum of ``losses`` returned by the adaptor and :math:`w_{hl}` is its weight;
        * intermediate_losses is defined via `intermediate_matches`.
        * signal_match_losses is defined via `signal_matches`.

    Args:
        temperature (float) :temperature for the distillation. The teacher and student models' logits will be divided by the temperature in computing the KD loss. The temperature typicially ranges from 1 to 10. We found that temperature higher than 1 usually leads to better performance.
        temperature_scheduler: dynamically adjusts temperature. See :data:`~textbrewer.presets.TEMPERATURE_SCHEDULER` for all available options.
        kd_loss_type (str): KD loss function for the ``logits`` term returned by the adaptor, can be ``'ce'`` or ``'mse'``. See :data:`~textbrewer.presets.KD_LOSS_MAP`.
        kd_loss_weight (float): the weight for the KD loss.
        hard_label_weight (float): the weight for the sum of ``losses`` term returned by the adaptor. ``losses`` may include the losses on the ground-truth labels and other user-defined losses. 
        kd_loss_weight_scheduler: Dynamically adjusts KD loss weight. See :data:`~textbrewer.presets.WEIGHT_SCHEDULER` for all available options.
        hard_label_weight_scheduler: Dynamically adjusts the weight of the sum of ``losses``. See :data:`~textbrewer.presets.WEIGHT_SCHEDULER` for all available options.
        probability_shift (bool): if ``True``, switch the ground-truth label's logit and the largest logit predicted by the teacher, to make the ground-truth label's logit largest. Requires ``labels`` term returned by the adaptor.
        is_caching_logits (bool): if ``True``, caches the batches and the output logits of the teacher model in memory, so that those logits will only be calcuated once. It will speed up the distillation process. This feature is **only available** for :class:`~textbrewer.BasicDistiller` and :class:`~textbrewer.MultiTeacherDistiller`, and only when distillers' ``train()`` method is called with ``num_steps=None``. It is suitable for small and medium datasets.
        intermediate_matches (`List[Dict]`) : Configuration for intermediate feature matching. Each element in the list is a dict, representing a pair of matching config. 
        signal_matches (`List[src.textbrewer.extractor]`)
    
    The dict in `intermediate_matches` contains the following keys:

        * '**layer_T**': `layer_T` (*int*): selects the layer_T-th layer of teacher model.
        * '**layer_S**': `layer_S` (*int*): selects the layer_S-th layer of student model.

        .. Note::
        
            1. `layer_T` and `layer_S` indicate layers in ``attention`` or ``hidden`` list in the returned dict of the adaptor, rather than the actual layers in the model. 
            2. If the loss is :func:`fst <textbrewer.losses.fsp_loss>` or :func:`nst <textbrewer.losses.mmd_loss>`, two layers have to be chosen from the teacher and the student respectively. In this case, `layer_T` and `layer_S` are lists of two ints. See the example below.

        * '**feature**': `feature` (*str*): features of intermediate layers. It can be:

            * '**attention**' : attention matrix, of the shape (*batch_size*, *num_heads*, *length*, *length*) or (*batch_size*, *length*, *length*)
            * '**hidden**'：hidden states, of the shape (*batch_size*, *length*, *hidden_dim*).

        * '**loss**' : `loss` (*str*) : loss function. See :data:`~textbrewer.presets.MATCH_LOSS_MAP` for available losses. Currently includes: ``'attention_mse'``, ``'attention_ce'``, ``'hidden_mse'``, ``'nst'``, etc.
        * '**weight**': `weight` (float) : weight for the loss.
        * '**proj**' : `proj` (*List*, optional) : if the teacher and the student have the same feature dimension, it is optional; otherwise it is required. It is the mapping function to match teacher and student intermediate feature dimension. It is a list, with these elements:

            * **proj[0]** (*str*): mapping function, can be ``'linear'``, ``'relu'``, ``'tanh'``. See :data:`~textbrewer.presets.PROJ_MAP`.
            * **proj[1]** (*int*): feature dimension of student model.
            * **proj[2]** (*int*): feature dimension of teacher model.
            * **proj[3]** (*dict*): optional, provides configurations such as learning rate. If not provided, the learning rate and optimizer configurations will follow the default config of the optimizer, otherwise it will use the ones specified here.

    Example::

        from src.textbrewer import DistillationConfig

        # simple configuration: use default values, or try different temperatures 
        distill_config = DistillationConfig(temperature=8)

        # adding intermediate feature matching
        # under this setting, the returned dict results_T/S of adaptor_T/S should contain 'hidden' key.
        # The mse loss between teacher's results_T['hidden'][10] and student's results_S['hidden'][3] will be computed
        distill_config = DistillationConfig(
            temperature=8,
            intermediate_matches = [{'layer_T':10, 'layer_S':3, 'feature':'hidden', 'loss':'hidden_mse', 'weight':1}]
        )

        # multiple inatermediate feature matching. The teacher and the student have a  hidden_dim of 768 and 384 respectively. 
        distill_config = DistillationConfig(
            temperature = 8, 
            intermediate_matches = [ \\
            {'layer_T':0,  'layer_S':0, 'feature':'hidden','loss': 'hidden_mse', 'weight' : 1,'proj':['linear',384,768]},
            {'layer_T':4,  'layer_S':1, 'feature':'hidden','loss': 'hidden_mse', 'weight' : 1,'proj':['linear',384,768]},
            {'layer_T':8,  'layer_S':2, 'feature':'hidden','loss': 'hidden_mse', 'weight' : 1,'proj':['linear',384,768]},
            {'layer_T':12, 'layer_S':3, 'feature':'hidden','loss': 'hidden_mse', 'weight' : 1,'proj':['linear',384,768]}]
        )

    """
    def __init__(self,temperature=4,
                      temperature_scheduler = 'none',
                      hard_label_weight=0,
                      hard_label_weight_scheduler = 'none',
                      kd_loss_type='ce',
                      kd_loss_weight=1,
                      kd_loss_weight_scheduler = 'none',
                      probability_shift = False,
                      intermediate_matches:Optional[Union[List[Dict], pathlib.Path]]=None,
                      signal_matches: Optional[List[extractor]]=None,
                      is_caching_logits = False):
        super(DistillationConfig, self).__init__()

        self.temperature = temperature
        self.temperature_scheduler = None
        if temperature_scheduler != 'none':
            assert temperature_scheduler in TEMPERATURE_SCHEDULER, \
                    f"Invalid temperature_scheduler {temperature_scheduler}"
            self.temperature_scheduler = TEMPERATURE_SCHEDULER[temperature_scheduler]

        self.hard_label_weight = hard_label_weight
        self.hard_label_weight_scheduler = None
        if hard_label_weight_scheduler != 'none':
            assert hard_label_weight_scheduler in WEIGHT_SCHEDULER, \
                    "Invalid hard_label_weight_scheduler"
            self.hard_label_weight_scheduler = WEIGHT_SCHEDULER[hard_label_weight_scheduler]

        self.kd_loss_type = kd_loss_type
        self.kd_loss_weight = kd_loss_weight
        self.kd_loss_weight_scheduler = None
        if kd_loss_weight_scheduler != 'none':
            assert kd_loss_weight_scheduler in WEIGHT_SCHEDULER, \
                    "Invalid kd_loss_weight_scheduler"
            self.kd_loss_weight_scheduler = WEIGHT_SCHEDULER[kd_loss_weight_scheduler]

        self.probability_shift = probability_shift

        self.intermediate_matches:[List[IntermediateMatch]] = []
        if intermediate_matches:
            if type(intermediate_matches) == str:
                with open(intermediate_matches, 'r') as f:
                    intermediate_matches = json.loads(f.read())
            self.intermediate_matches = [IntermediateMatch.from_dict(im) for im in intermediate_matches]

        self.signal_matches = signal_matches
        self.is_caching_logits = is_caching_logits
