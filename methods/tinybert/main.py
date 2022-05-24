"""
    Tinybert: https://arxiv.org/pdf/1909.10351.pdf 
    
    We implement the general distillation step of tinyBERT.
    Task-specfic distillation step is skipped here.

    Training loss is a combination of

    MSE(teacher_unnormalized_attention, student_unnormalized_attention)
    MSE(teacher_hidden_states, student_hidden_states * W),
    where W is a weight matrix of appropriate size that can match student's 
    hidden size to that of the teacher's.

    Note: student is assumed to have the same number of per-layer
    attention heads as teacher
"""
import argparse
import os
import sys
import logging
from transformers import AutoModel, AutoModelForMaskedLM, AutoModelForCausalLM
root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(root)
from src import *
from transformers import logging as tl
tl.set_verbosity_error()

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[distillation_parser])
    parser.set_defaults(
        lm_type='mlm', # not for clm type yet, passing in 'clm' is futile 
        hard_label_weight=0., # no token prediction loss, value>0 is futile
        kd_loss_weight=0., # no kd loss, value>0 is futile
        mlm_pred_rate=0., # default to not no mask at input
        mlm_mask_rate=1., # if mlm_pred_rate>0, randomly place a [MASK] at input, value < 1 is futile
        mlm_rand_rate=0., # value>0 is futile
        mlm_keep_rate=0. # value>0 is futile
    )
    args = parser.parse_args()

    # random seed
    set_rng(args.seed)

    # Load teacher and student locally
    T = load_model(args.teacher, AutoModel)
    T.cuda()
    S = load_model(args.student, AutoModel)
    S.cuda()

    # create optimizer locally
    optimizer, scheduler = create_optimizer_and_scheduler(S, args)

    # misc training configs
    train_config = TrainingConfig(
                        args.max_steps,
                        args.output_dir,
                        max_grad_norm=args.max_grad_norm,
                        gradient_accumulation_steps=args.grad_acc_step,
                        ckpt_steps=args.ckpt_freq_iter,
                        log_steps=args.log_freq_iter,
                        save_ini=args.save_ini,
                        fp16=args.fp16,
                        fp16_opt_level=args.fp16_opt_level,
                        local_rank=local_rank,
                        rank=global_rank,
                        world_size=world_size
                   )
    # misc distillation configs
    distill_config = DistillationConfig(
                        temperature=args.temperature,
                        hard_label_weight=args.hard_label_weight,
                        kd_loss_weight=args.kd_loss_weight,
                        intermediate_matches=args.intermediate_matches
                     )
    # fields of forward outputs to be used in distillation
    def adaptor(model_input, model_output):
        adaptor_dict['hidden'] = model_output['hidden_states']
        adaptor_dict['attention'] = model_output['attentions']
        return adaptor_dict

    # Define the distiller
    distiller = GeneralDistiller(
                    train_config,
                    distill_config,
                    T, S, adaptor, adaptor
                )

    # Training DataLoader
    train_data = LmSeqsDataset(
                        args.data,
                        args.teacher, # teacher model and tokenizer share the same name
                        args.token_counts,
                        args.lm_type=='mlm'
                 )
    train_dataloader = create_dataLoader(train_data, args, is_training=True)

    # batch postprocessor: masking
    batch_postprocessor = lambda batch: \
        train_data.prepare_batch_mlm(batch,
            pred_rate=args.mlm_pred_rate,
            mask_rate=args.mlm_mask_rate,
            rand_rate=0.,
            keep_rate=0.
        )

    # Additional args for forward pass of a transformer model
    # We need this hidden states for distilbert
    additional_args = {'output_hidden_states': True, 'output_attention': True}

    # Validation dataloader
    if args.eval_data:
        eval_data = LmSeqsDataset(
                        args.eval_data,
                        args.teacher, # teacher model and tokenizer share the same name
                        args.token_counts,
                        args.lm_type=='mlm'
                 )
        eval_dataloader = create_dataLoader(eval_data, args)
        if global_rank == 0:
           logger.info('training size: {} | validation size: {} '.format(
                       len(train_data), len(eval_data)))
    else:
        eval_dataloader = None
        if global_rank == 0:
            logger.info('training size: {} | validation: N/A '.format(
                       len(train_data)))

    # callbacks
    callback = callbacks(distiller,
                         args.student,
                         args.teacher,
                         args.callback,
                         eval_dataloader=eval_dataloader,
                         batch_postprocessor=batch_postprocessor,
                         glue_tasks=args.glue_task)

    # run
    with distiller:
        distiller.train(train_dataloader, optimizer, scheduler,
                        resume=args.resume_from,
                        batch_postprocessor=batch_postprocessor,
                        callback=callback,
                        eval_dataloader=eval_dataloader,
                        **additional_args
                    )
