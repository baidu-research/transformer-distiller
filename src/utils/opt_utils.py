import logging
import torch
from functools import partial
from apex.optimizers import FusedAdam
from apex.optimizers import FusedLAMB
import transformers

log = logging.getLogger()


OPT_CLASS_DICT = {
    'adam_fused': partial(FusedAdam, betas=(0.9, 0.999), ),
    'lamb_fused': partial(FusedLAMB, betas=(0.9, 0.999), ),
    'adam':       partial(torch.optim.Adam, betas=(0.9, 0.999), ),
    'adamw':      partial(torch.optim.AdamW, betas=(0.9, 0.999), )
}

try:
    from fairscale.optim.oss import OSS
    OPT_CLASS_DICT['adafactor_oss'] = lambda params, **kwargs: OSS(params=params, optim=Adafactor, scale_parameter=False, relative_step=False, **kwargs)
    OPT_CLASS_DICT['adam_oss'] = lambda params, **kwargs: OSS(params=params, optim=T.optim.AdamW, **kwargs)
except ImportError :
    pass
try:
    from fairscale.optim import Adam
    OPT_CLASS_DICT['fairscale_adam_oss'] = lambda params, **kwargs: OSS(params=params, optim=Adam, scale_parameter=False, relative_step=False, **kwargs),
except ImportError :
    pass
try: 
    from radam import RAdam
    OPT_CLASS_DICT['radam'] = RAdam
except ImportError:
    pass

LR_SCHEDULER_DICT = {
    'constant': transformers.get_constant_schedule_with_warmup,
    'linear':   transformers.get_linear_schedule_with_warmup,
    'cosine':   transformers.get_cosine_schedule_with_warmup
}


def create_optimizer_and_scheduler(model, args):
    optimizer = OPT_CLASS_DICT[args.opt_type](
                    model.parameters(),
                    lr=args.lr,
                    weight_decay=args.weight_decay
                )
    if args.lr_scheduler == 'constant':
        scheduler = LR_SCHEDULER_DICT[args.lr_scheduler](
                        optimizer,
                        num_warmup_steps=args.warmup_steps
                    )
    else:
        scheduler = LR_SCHEDULER_DICT[args.lr_scheduler](
                        optimizer,
                        num_warmup_steps=args.warmup_steps,
                        num_training_steps=args.max_steps
                    )
    return optimizer, scheduler
