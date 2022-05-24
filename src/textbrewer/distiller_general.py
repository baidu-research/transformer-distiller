from .distiller_utils import *
from .distiller_basic import BasicDistiller

class GeneralDistiller(BasicDistiller):
    """
    Supports intermediate features matching. **Recommended for single-teacher single-task distillation**.

    Args:
        train_config (:class:`TrainingConfig`): training configuration.
        distill_config (:class:`DistillationConfig`): distillation configuration.
        model_T (:class:`torch.nn.Module`): teacher model.
        model_S (:class:`torch.nn.Module`): student model.
        adaptor_T (Callable): teacher model's adaptor.
        adaptor_S (Callable): student model's adaptor.

    The roles of `adaptor_T` and `adaptor_S` are explained in :py:func:`adaptor`.

    """
    def __init__(self, train_config,
                 distill_config,
                 model_T,
                 model_S,
                 adaptor_T,
                 adaptor_S):
        super(GeneralDistiller, self).__init__(train_config, distill_config, model_T, model_S, adaptor_T, adaptor_S)

        self.projs = []
        self.projs_group = []
        for im in self.d_config.intermediate_matches:
            if im.proj is not None:
                projection = im.proj[0]
                dim_in = im.proj[1]
                dim_out = im.proj[2]
                self.projs_group.append(im.proj[3])
                self.projs.append(PROJ_MAP[projection](dim_in,dim_out))
                self.projs[-1].to(self.t_config.local_rank)
            else:
                self.projs.append(None)
                self.projs_group.append(None)

        self.d_config.is_caching_logits = False

    def train_on_batch(self, batch, args):

        (teacher_batch, results_T), (student_batch, results_S) = get_outputs_from_batch(batch, self.t_config.local_rank, self.model_T, self.model_S, args)

        results_T = post_adaptor(self.adaptor_T(teacher_batch,results_T))
        results_S = post_adaptor(self.adaptor_S(student_batch,results_S))

        total_loss, losses_dict = self.compute_loss(results_S, results_T)

        return total_loss, losses_dict


    def compute_loss(self,results_S,results_T):
        losses_dict = dict()
        total_loss  = 0

        # logit KD loss
        if 'logits' in results_T and 'logits' in results_S:
            logits_list_T = results_T['logits']  # list of tensor
            logits_list_S = results_S['logits']  # list of tensor
            total_kd_loss = 0
            if 'logits_mask' in results_S:
                masks_list_S = results_S['logits_mask']
                logits_list_S = select_logits_with_mask(logits_list_S,masks_list_S)  #(mask_sum, num_of_class)
            if 'logits_mask' in results_T:
                masks_list_T = results_T['logits_mask']
                logits_list_T = select_logits_with_mask(logits_list_T,masks_list_T)  #(mask_sum, num_of_class)

            if self.d_config.probability_shift is True:
                labels_list = results_S['labels']
                for l_T, l_S, labels in zip(logits_list_T, logits_list_S, labels_list):
                    l_T = probability_shift_(l_T, labels)
                    if self.d_config.temperature_scheduler is not None:
                        temperature = self.d_config.temperature_scheduler(l_S, l_T, self.d_config.temperature)
                    else:
                        temperature = self.d_config.temperature
                    total_kd_loss += self.kd_loss(l_S, l_T, temperature)
            else:
                for l_T,l_S in zip(logits_list_T,logits_list_S):
                    if self.d_config.temperature_scheduler is not None:
                        temperature = self.d_config.temperature_scheduler(l_S, l_T, self.d_config.temperature)
                    else:
                        temperature = self.d_config.temperature
                    total_kd_loss += self.kd_loss(l_S, l_T, temperature)
            total_loss += total_kd_loss * self.d_config.kd_loss_weight
            losses_dict['kd_loss'] = total_kd_loss

        # intermediate loss on hidden features or attention maps
        inters_T = {feature: results_T.get(feature,[]) for feature in FEATURES}
        inters_S = {feature: results_S.get(feature,[]) for feature in FEATURES}
        inputs_mask_T = results_T.get('inputs_mask',None)
        inputs_mask_S = results_S.get('inputs_mask',None)
        for ith,inter_match in enumerate(self.d_config.intermediate_matches):
            layer_T = inter_match.layer_T
            layer_S = inter_match.layer_S
            feature = inter_match.feature
            loss_type = inter_match.loss
            match_weight = inter_match.weight
            match_loss = MATCH_LOSS_MAP[loss_type]

            if type(layer_S) is list and type(layer_T) is list:
                inter_S = [inters_S[feature][s] for s in layer_S]
                inter_T = [inters_T[feature][t] for t in layer_T]
                name_S = '-'.join(map(str,layer_S))
                name_T = '-'.join(map(str,layer_T))
                if self.projs[ith]:
                    #inter_T = [self.projs[ith](t) for t in inter_T]
                    inter_S = [self.projs[ith](s) for s in inter_S]
            else:
                inter_S = inters_S[feature][layer_S]
                inter_T = inters_T[feature][layer_T]
                name_S = str(layer_S)
                name_T = str(layer_T)
                if self.projs[ith]:
                    #inter_T = self.projs[ith](inter_T)
                    inter_S = self.projs[ith](inter_S)
            intermediate_loss = match_loss(inter_S, inter_T, mask=inputs_mask_S)
            total_loss += intermediate_loss * match_weight
            losses_dict[f'{feature}_{loss_type}_{name_S}_{name_T}'] = intermediate_loss

        # "non-trivial" signal match losses
        if self.d_config.signal_matches is not None:
            for match in self.d_config.signal_matches:
                with torch.no_grad():
                    match.signal_T.compute()
                match.signal_S.compute()
                match_loss = match.loss(match.signal_T.cache.detach(),
                                        match.signal_S.cache)
                total_loss += match.weight * match_loss
                losses_dict[f'{match.name_tag}'] = match_loss

        # hard label loss
        if 'losses' in results_S:
            total_hl_loss = 0
            for loss in results_S['losses']:
                # in case of multi-GPU
                total_hl_loss += loss.mean() 
            total_loss += total_hl_loss * self.d_config.hard_label_weight
            losses_dict['hard_label_loss'] = total_hl_loss
        return total_loss, losses_dict
