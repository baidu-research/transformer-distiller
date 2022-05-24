"""
    Distilbert: https://arxiv.org/pdf/1910.01108.pdf

    Training loss is a combination of

    KL(teacher_logits || student_logits)
    cross-entropy of token prediction (the standard masked/causal language modeling loss)
    cosine_distance(teacher_hidden_states, student_hidden_states), default to last layer

    [Future] The training loss can be applied to distilling causal language models too.
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
    parser.set_defaults(lm_type='mlm')
    parser.set_defaults(hard_label_weight=0.)
    parser.set_defaults(kd_loss_weight=1.)
    parser.set_defaults(kd_loss_type='ce')
    args = parser.parse_args()

    # check some corner cases
    if args.lm_type == 'clm':
        raise NotImplementedError('Not implemented for causal LM yet!')

    if not args.hard_label_weight and not args.kd_loss_weight:
        # If asked to not work with logits, then at least we need an
        # intermediate loss to train on
        assert args.intermediate_matches
        args.mlm_pred_rate = 0
        args.mlm_mask_rate = 0
        args.mlm_rand_rate = 0
        args.mlm_keep_rate = 0
    else:
        # If asked to work with logits, then mlm prediction rate > 0
        assert args.mlm_pred_rate > 0

    if args.mlm_pred_rate == 0:
        # If zero mlm prediction rate. Then we cannot work on any loss
        # that involves logits
        assert args.hard_label_weight == 0
        assert args.kd_loss_weight == 0

    # random seed
    set_rng(args.seed)

    # Load teacher and student locally
    if args.mlm_pred_rate:
        auto_model_type = AutoModelForMaskedLM
    else:
        auto_model_type = AutoModel

    T = load_model(args.teacher, auto_model_type)
    T.cuda()
    S = load_model(args.student, auto_model_type)
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
        adaptor_dict = {}
        if args.hard_label_weight:
            adaptor_dict['losses'] = model_output['loss'] # (1,)
        if args.kd_loss_weight:
            adaptor_dict['logits'] = model_output['logits'] # (B, L, V)
            # ignore context tokens
            adaptor_dict['logits_mask'] = model_input['labels'] != -100
        if args.intermediate_matches:
            adaptor_dict['hidden'] = model_output['hidden_states']
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
            args.mlm_pred_rate,
            args.mlm_mask_rate,
            args.mlm_rand_rate,
            args.mlm_keep_rate
        )

    # Additional args for forward pass of a transformer model
    # We need this hidden states for distilbert
    additional_args = {'output_hidden_states': True}

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
