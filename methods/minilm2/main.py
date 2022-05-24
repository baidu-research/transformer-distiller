"""
    minilm-v2: https://arxiv.org/pdf/2012.15828.pdf

    Concatenate key, query, value vectors from all heads, then split the
    vectors into R segment, where R is the number of so called relation heads.
    Each segment has dimention d_R. For example, in BERT-base, the concatenated
    key/query/value vector has dimension 768. If we use R=24 relation heads,
    then d_R = 768/24=32.

    Here we denote the r-th key vector for the i-th token as key[r,i], and
    likewise for query and value vectors.

    Define the following pairwise token relationships for the r-th relation
    head, between token i and j in an input sentence.
    kk[r,i,j] = <key[r,i], key[r,j]>
    qq[r,i,j] = <query[r,i], query[r,j]>
    vv[r,i,j] = <value[r,i], value[r,j]>

    The r-th KK matrix is normalized by softmax operator
    KK[r] = softmax(normalizer(kk[r, i, j])) over j's
    Note that the normalizer() function has multiple choices. A simpliest one
    may be dividing sqrt(d_R), suggested by the paper.

    Likewise, QQ[r] and VV[r] are obtained in the same way.

    Training loss is

    \sum_{r=1}^R KL(KK_teacher[r] || KK_student[r])
                    +KL(QQ_teacher[r] || QQ_student[r])
                    + KL(VV_teacher[r]|| VV_student[r])

    Obviously, the loss can be applied between different teacher and student
    layers. It is an open question how to select the layer alignment.
"""
import argparse
import logging
import math
import os
import sys
root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(root)
from functools import partial
from transformers import AutoModel, AutoModelForMaskedLM, AutoModelForCausalLM
from src import *
from utils import *
from transformers import logging as tl
tl.set_verbosity_error()

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    minilm2_parser = argparse.ArgumentParser(parents=[distillation_parser])
    # The seminal minilm2 paper suggests no hard label loss, logit KD loss
    # or intermediate feature matching loss. So the following input args
    # defaults to 0
    minilm2_parser.set_defaults(mlm_pred_rate=0.,
                                mlm_mask_rate=0.,
                                mlm_rand_rate=0.,
                                mlm_keep_rate=0.,
                                kd_loss_weight=0.,
                                hard_label_weight=0.
                               )
    minilm2_group = minilm2_parser.add_argument_group('minilm2 special parameters')
    minilm2_group.add_argument('--teacher-layers', type=int, nargs='+',
                               help='indices of teacher transformer layers '
                               'to be aligned with student, starting from 1. '
                               'Can be a single or a list of integer(s)')
    minilm2_group.add_argument('--student-layers', type=int, nargs='+',
                               help='same format as teacher-layer. '
                               'If a list of integers, student-layers[i] is '
                               'aligned with teacher-layers[i]. This the two '
                               'lists must have the same length')
    minilm2_group.add_argument('--weights', type=float,
                               nargs='+', default=[1.0],
                               help='weight applied to kk-qq-vv matching loss '
                               'for every pair of aligned teacher-student '
                               'layers. If a single float number, '
                               'all aligned layer pairs share this weight')
    minilm2_group.add_argument('--num-relation-heads', type=int,
                               help='number of relation heads')
    minilm2_group.add_argument('--normalizer', choices=normalizer_types,
                               default='sqrt_dim',
                               help='specify ways to normalize the qq/kk/vv '
                               'matrix. Default to "sqrt_dim", simply divide '
                               'the matrix by sqrt(d_R), as suggested by the '
                               'seminal minilm2 paper')
    minilm2_group.add_argument('--minilm2-loss', choices=minilm2_loss_types,
                               default='kl_div',
                               help='loss of matching the normalized '
                               'KK, QQ, VV matrices')
    args = minilm2_parser.parse_args()

    assert len(args.teacher_layers) == len(args.student_layers)
    if len(args.weights) == 1:  # all aligned layer pairs share the same weight
        args.weights = args.weights * len(args.teacher_layers)
    else:
        len(args.weights) == len(args.teacher_layers)

    # random seed
    set_rng(args.seed)

    # Load teacher and student locally
    if args.kd_loss_weight or args.hard_label_weight:
        if args.lm_type == 'mlm':
            auto_model_type =  AutoModelForMaskedLM
        else:
            auto_model_type = AutoModelForCausalLM
    else: # the bare model without lm head
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

    # The really interesting part: minilm2's kk, qq, vv matches
    minilm2_matches = []
    if args.normalizer == 'sqrt_dim':
        try:
            teacher_dim = T.config.hidden_size
            student_dim = S.config.hidden_size
        except:
            raise NotImplementedError('Cannot figure out Q/K/V dimensions')
        assert teacher_dim % args.num_relation_heads == 0
        assert student_dim % args.num_relation_heads == 0
        teacher_normalizer = math.sqrt(teacher_dim / args.num_relation_heads)
        student_normalizer = math.sqrt(student_dim / args.num_relation_heads)
    else:
        teacher_normalizer = student_normalizer = args.normalizer

    for layer_t, layer_s, weight in zip(args.teacher_layers,
                                        args.student_layers,
                                        args.weights):
        for field in ['query', 'key', 'value']:
            signal_t = token_pairwise_similarity(
                            get_module(T, field, layer_t),
                            args.num_relation_heads,
                            partial(norm_function,
                                    normalizer=teacher_normalizer,
                                    log_softmax=True)
                            )
            signal_s = token_pairwise_similarity(
                            get_module(S, field, layer_s),
                            args.num_relation_heads,
                            partial(norm_function,
                                    normalizer=student_normalizer,
                                    log_softmax=True)
                            )
            minilm2_matches.append(
                SignalMatch(
                            signal_t, signal_s, weight,
                            partial(minilm2_loss, loss_type=args.minilm2_loss),
                            f'{field[0]}{field[0]}_{layer_t}-{layer_s}')
                )

    distill_config = DistillationConfig(
                        temperature=args.temperature,
                        hard_label_weight=args.hard_label_weight,
                        kd_loss_weight=args.kd_loss_weight,
                        intermediate_matches=args.intermediate_matches,
                        signal_matches=minilm2_matches
                        )

    # fields of forward outputs to be used in distillation
    def adaptor(model_input, model_output):
        """
        The seminal minilm_v2 paper suggests no logit kd loss or hard
        label loss. So the adaptor_dict will be empty most likely.
        """
        adaptor_dict = {}
        if args.kd_loss_weight:
            adaptor_dict['logits'] = model_output['logits']
            adaptor_dict['logits_mask'] = model_input['labels'] != -100
        if args.hard_label_weight:
            adaptor_dict['losses'] = model_output['loss']
        if args.intermediate_matches:
            required_features = [im.feature for im in distill_config.intermediate_matches]
            if 'hidden' in required_features:
                adaptor_dict['hidden'] = model_output['hidden_states']
            if 'attention' in required_features:
                adaptor_dict['attention'] = model_output['attentions']
        return adaptor_dict

    # Define the distiller
    distiller = GeneralDistiller(
                    train_config,
                    distill_config,
                    T, S, adaptor, adaptor,
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

    additional_args = {}
    if args.intermediate_matches:
        required_features = [im.feature for im in distill_config.intermediate_matches]
        if 'hidden' in required_features:
            additional_args['output_hidden_states'] = True
        if 'attention' in required_features:
            additional_args['output_attentions'] = True

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
