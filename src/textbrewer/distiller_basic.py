import os
import signal
import subprocess
import time
import torch
from itertools import tee
from torch.nn.parallel import DistributedDataParallel
from .distiller_utils import *
from ..callbacks import callbacks
from ..utils import get_rng, set_rng, freeze_unused_params

class BasicDistiller(AbstractDistiller):
    """
    Performs **single-teacher single-task** distillation, provides basic distillation strategies.

    Args:
        train_config (:class:`TrainingConfig`): training configuration.
        distill_config (:class:`DistillationConfig`): distillation configuration.
        model_T (:class:`torch.nn.Module`): teacher model.
        model_S (:class:`torch.nn.Module`): student model.
        adaptor_T (Callable): teacher model's adaptor.
        adaptor_S (Callable): student model's adaptor.

    The roles of `adaptor_T` and `adaptor_S` are explained in :py:func:`adaptor`.

    """
    def __init__(self,
                 train_config,
                 distill_config,
                 model_T,
                 model_S,
                 adaptor_T,
                 adaptor_S):
        super(BasicDistiller, self).__init__(train_config, distill_config, model_T, model_S, adaptor_T, adaptor_S)

    def save_and_callback(self, global_step, callback,
                          batch_postprocessor, eval_dataloader, **args):
        if self.rank != 0:
            torch.distributed.barrier()    # save with single process
        else:
            logger.info(f"Saving at global step {global_step}")
            coreModel = self.model_S.module if hasattr(self.model_S, "module") else self.model_S
            state_dict = coreModel.state_dict()
            save_at = os.path.join(self.t_config.output_dir, f"{global_step}.pkl")
            torch.save(state_dict, save_at)
            if self.world_size > 1:  # rank0 reaches barrier if DDP
                torch.distributed.barrier()
        if eval_dataloader is not None:
            # evaluate the losses on eval data in the same way as in training data
            if self.rank == 0:
                logger.info("Computing losses on evaluation set...")
            eval_loss, eval_losses_dict = self.basic_eval(
                    eval_dataloader, self.model_S, batch_postprocessor, **args)
            self.write_loss(eval_loss, global_step, eval_losses_dict, 'eval')
            if self.rank == 0:
                loss_str = []
                for k, v in eval_losses_dict.items():
                    loss_str.append(' {}: {:.4f}'.format(k, v))
                loss_str = ' |'.join(loss_str)
                logger.info("Evaluation | loss: {:.4f} |".format(eval_loss) + loss_str)
            if self.d_config.signal_matches:
                for sm in self.d_config.signal_matches:
                    sm.empty_cache()

        if callback is not None:
            if self.rank == 0:
                logger.info("Running callbacks...")
            callback.set_gs(global_step)
            callback.apply()
            if self.d_config.signal_matches:
                for sm in self.d_config.signal_matches:
                    sm.empty_cache()
        self.model_S.train()


    def write_loss(self, total_loss, global_step, losses_dict=None, split='train'):
        cpu_total_loss = total_loss.cpu().item()
        self.tb_writer.add_scalars('total_loss', {split: cpu_total_loss}, global_step)
        if losses_dict is not None:
            for name, loss in losses_dict.items():
                cpu_loss = loss.cpu().item()
                self.tb_writer.add_scalars(f"{name}", {split: cpu_loss}, global_step)


    def initialize_training(self, dataloader, optimizer, scheduler, resume,
                            batch_postprocessor, **args):
        """
        Prepare for training:

        1) Update optimizer to include additional learnable parameters, e.g.,
        projections that maps hidden features to the same dimension.

        2) Wrap optimizer with amp and DDP if required

        3) If asked to, load states from a checkpoint
        """
        if hasattr(self,'projs'):
            for proj,proj_group in zip(self.projs, self.projs_group):
                if proj is not None:
                    assert isinstance(proj,nn.Module)
                    optimizer.add_param_group({**{'params':proj.parameters()},**proj_group})

        logger.debug("Optimizer param group: ")
        logger.debug(f"{[[s.shape for s in g['params']] for g in optimizer.param_groups]}")
        self.freeze_student_unused_params(batch_postprocessor, **args)

        if self.t_config.fp16:
            if not has_apex:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            if isinstance(self.model_T,(list,tuple)):
                models = [self.model_S] + list(self.model_T)
                models, optimizer = amp.initialize(models, optimizer,
                                        opt_level=self.t_config.fp16_opt_level)
                self.model_S = models[0]
                self.model_T = models[1:]
            elif isinstance(self.model_T,dict):
                tasknames, model_Ts = zip(*self.model_T.items())
                models = [self.model_S] + list(model_Ts)
                models, optimizer = amp.initialize(models, optimizer,
                                        opt_level=self.t_config.fp16_opt_level)
                self.model_S = models[0]
                self.model_T = dict(zip(tasknames,models[1:]))
            else:
                (self.model_S, self.model_T), optimizer = amp.initialize(
                        [self.model_S, self.model_T],
                        optimizer,
                        opt_level=self.t_config.fp16_opt_level
                        )
            if self.rank == 0:
                logger.info("amp optimizer created!")

        if resume is None:
            progress = progress_meter()
        else:
            # resume training from where we left
            resume_dict = torch.load(resume, map_location='cpu')
            self.model_S.load_state_dict(resume_dict['model'])
            optimizer.load_state_dict(resume_dict['optimizer'])
            scheduler.load_state_dict(resume_dict['scheduler'])
            dataloader.load_state_dict(resume_dict['dataloader'])
            progress = progress_meter()
            progress.load_state_dict(resume_dict['progress'])

            if hasattr(self, 'projs'):
                for i, proj in enumerate(self.projs):
                    if proj is not None:
                        self.projs[i].load_state_dict(resume_dict['projs'][i])

            if self.t_config.fp16:
                amp.load_state_dict(resume_dict['amp'])

        if self.world_size > 1:
            self.model_S = DistributedDataParallel(
                                    self.model_S,
                                    device_ids = [self.local_rank],
                                    output_device = self.local_rank,
                           )
            if hasattr(self,'projs'):
                for i,proj in enumerate(self.projs):
                    if proj is not None:
                        assert isinstance(proj,nn.Module)
                        self.projs[i] = DistributedDataParallel(
                                            proj,
                                            device_ids = [self.local_rank],
                                            output_device = self.local_rank
                                        )
            if self.rank == 0:
                logger.info("DDP optimizer created!")

        elif self.t_config.data_parallel:
            self.model_S = torch.nn.DataParallel(self.model_S)
            if isinstance(self.model_T,(list,tuple)):
                self.model_T = [torch.nn.DataParallel(model_t) for model_t in self.model_T]
            elif isinstance(self.model_T,dict):
                self.model_T = {k:torch.nn.DataParallel(v) for k,v in self.model_T.items()}
            else:
                self.model_T = torch.nn.DataParallel(self.model_T)

        # reload rng's at last, avoid any consumption of rng states before
        # entering training
        if resume is not None:
            set_rng(rng_state_dict=resume_dict['rng_states'])
        return dataloader, optimizer, scheduler, progress


    def train_with_num_steps(self, dataloader, optimizer, scheduler, progress,
                             callback, batch_postprocessor, eval_dataloader,
                             **args):
        if self.rank == 0:
            logger.info(f"Total training steps: {self.t_config.num_steps}")

        accumulate_loss = accumulated_loss() # report average loss due to accumulated grad steps
        
        # At time limit, set the indicator. Refer to
        # https://stackoverflow.com/a/53688983/1363671
        # https://stackoverflow.com/a/32923097/1363671
        # But note we need to set indicator as nonlocal instead of global var
        # Refer to https://stackoverflow.com/questions/2609518/unboundlocalerror-with-nested-function-scopes
        def handler(signum, frame):
            nonlocal time_limit_hits
            time_limit_hits = True
            if self.rank == 0:
                logger.info("Received SIGUSR1. Prepare to resubmit job ...")
        time_limit_hits = False
        signal.signal(signal.SIGUSR1, handler)

        t0 = time.time()
        for i, batch in enumerate(dataloader):
            if batch_postprocessor is not None:
                batch = batch_postprocessor(batch)

            total_loss, losses_dict = self.train_on_batch(batch, args)
            accumulate_loss.add(total_loss, losses_dict)
            total_loss /= self.t_config.gradient_accumulation_steps

            if self.t_config.fp16:
                with amp.scale_loss(total_loss,optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                total_loss.backward()
            if self.d_config.signal_matches:
                for sm in self.d_config.signal_matches:
                    sm.empty_cache()

            if (i+1)%self.t_config.gradient_accumulation_steps == 0:
                if self.t_config.max_grad_norm > 0:
                    if self.t_config.fp16:
                        torch.nn.utils.clip_grad_norm_(
                            amp.master_params(optimizer),
                            self.t_config.max_grad_norm
                        )
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model_S.parameters(),
                            self.t_config.max_grad_norm
                        )
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

                # write to tensorboard
                this_loss, this_loss_dict = accumulate_loss.report()
                self.write_loss(this_loss, progress.global_step, this_loss_dict)
                accumulate_loss.reset()
            
                # logging
                if progress.global_step % self.t_config.log_steps == 0:
                    loss_str = []
                    for k, v in this_loss_dict.items():
                        loss_str.append(' {}: {:.4f}'.format(k, v))
                    loss_str = ' |'.join(loss_str)
                    logger.info("rank-{} - step: {} | "
                                "speed: {:.2f} step/s | "
                                "peak_cuda_mem: {:.2f}G | loss: {:.4f} |"
                                .format(self.rank, progress.global_step,
                                (1 if progress.global_step==0 else \
                                    self.t_config.log_steps)/(time.time()-t0),
                                torch.cuda.max_memory_allocated()/1024**3,
                                this_loss) + loss_str
                               )
                    torch.cuda.reset_peak_memory_stats()
                    t0 = time.time()
                progress.step()
                # loss weight scheduler
                if self.d_config.kd_loss_weight_scheduler is not None:
                    self.d_config.kd_loss_weight = \
                        self.d_config.kd_loss_weight_scheduler(
                            progress.global_step/self.t_config.num_steps)
                if self.d_config.hard_label_weight_scheduler is not None:
                    self.d_config.hard_label_weight = \
                        self.d_config.hard_label_weight_scheduler(
                            progress.global_step/self.t_config.num_steps)

                if progress.global_step % self.t_config.ckpt_steps == 0 or \
                        progress.global_step == self.t_config.num_steps:
                    self.save_and_callback(
                        progress.global_step, callback,
                        batch_postprocessor, eval_dataloader, **args
                    )
                if time_limit_hits:
                    self.save_and_resubmit(dataloader, optimizer, scheduler,
                                           progress, amp)
            if progress.global_step == self.t_config.num_steps:
                break
        if self.rank == 0:
            logger.info("Training finished")


    def train(self, dataloader, optimizer, scheduler, resume=None,
              callback=None, batch_postprocessor=None, eval_dataloader=None, **args):
        """
        trains the student model.

        Args:
            dataloader: dataset iterator.
            optimizer: optimizer.
            scheduler: learning rate scheduler.
            resume: path to a checkpoint to reload from where training stopped
            callback (Callable): function called after each epoch, can be None.
            It is called as `callback(model=self.model_S, step = global_step)`
            batch_postprocessor (Callable): a function for post-processing batches. It should take a batch and return a batch. Its output is fed to the models and adaptors.
            **args: additional arguments fed to the model.
        Note:
            * If the batch is a list or tuple, model is called as: ``model(*batch, **args)``. Make sure the order of elements in the batch matches their order in ``model.forward``.
            * If the batch is a dict, model is called as: ``model(**batch,**args)``. Make sure the keys of the batch match the arguments of the ``model.forward``.
        """
        dataloader, optimizer, scheduler, progress = self.initialize_training(
            dataloader, optimizer, scheduler, resume,
            batch_postprocessor, **args)
        if self.rank == 0:
            logger.info("{:.2f}G GPU memory allocated per rank. Ready to train!"
                        .format(torch.cuda.memory_allocated()/1024**3)
                       )
        if self.t_config.save_ini and progress.global_step == 0:
            self.save_and_callback(0, callback,
                        batch_postprocessor, eval_dataloader, **args
                    )
        self.train_with_num_steps(dataloader, optimizer, scheduler, progress,
                                  callback, batch_postprocessor,
                                  eval_dataloader, **args)


    def train_on_batch(self, batch, args):
        if self.d_config.is_caching_logits is False:
            (teacher_batch, results_T), (student_batch, results_S) = get_outputs_from_batch(batch, self.local_rank, self.model_T, self.model_S, args)
            results_T = post_adaptor(self.adaptor_T(teacher_batch,results_T))
            results_S = post_adaptor(self.adaptor_S(student_batch,results_S))
        else:
            batch, cached_logits = batch
            _, (student_batch, results_S) = get_outputs_from_batch(batch, self.local_rank, self.model_T, self.model_S, args, no_teacher_forward=True)

            results_S = post_adaptor(self.adaptor_S(student_batch,results_S))
            results_T = {'logits':[logits.to(self.local_rank) for logits in cached_logits]}

            if 'logits_mask' in results_S:
                results_T['logits_mask'] = results_S['logits_mask']

        total_loss, losses_dict = self.compute_loss(results_S,results_T)

        return total_loss, losses_dict


    def compute_loss(self, results_S, results_T):
        total_loss  = 0
        losses_dict = dict()
        logits_list_T = results_T['logits']  # list of tensor
        logits_list_S = results_S['logits']  # list of tensor

        if 'logits_mask' in results_S:
            masks_list_S = results_S['logits_mask']
            logits_list_S = select_logits_with_mask(logits_list_S,masks_list_S)  #(mask_sum, num_of_class)
        if 'logits_mask' in results_T:
            masks_list_T = results_T['logits_mask']
            logits_list_T = select_logits_with_mask(logits_list_T,masks_list_T)  #(mask_sum, num_of_class)

        total_kd_loss = 0
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

        if 'losses' in results_S:
            total_hl_loss = 0
            for loss in results_S['losses']:
                # in case of multi-GPU
                total_hl_loss += loss.mean() 
            total_loss += total_hl_loss * self.d_config.hard_label_weight
            losses_dict['hard_label_loss'] = total_hl_loss
        return total_loss, losses_dict


    def cache_logits(self, batch, args, batch_postprocessor):
            if batch_postprocessor is not None:
                batch = batch_postprocessor(batch)
            batch = move_to_device(batch, self.local_rank)
            with torch.no_grad():
                if type(batch) is dict:
                    results_T = self.model_T(**batch,**args)
                else:
                    results_T = self.model_T(*batch, **args)
            results_T = post_adaptor(self.adaptor_T(batch,results_T))

            self.logits_cache.append([batch, [logits.to('cpu') for logits in results_T['logits']]])


    def save_and_resubmit(self, dataloader, optimizer, scheduler, progress, amp):
        """
        Upon time limit:
        1) Rank-0 saves all necessary states
        2) Rank-0 submits a new job that resumes from where we are left at 
        3) All ranks quit
        """
        if self.d_config.signal_matches:
            for sm in self.d_config.signal_matches:
                sm.release()
        if self.rank != 0:
            # rank > 0: just save random generators' states
            rng_states = get_rng()
            rng_states_all_ranks = [None]*self.world_size
            torch.distributed.all_gather_object(rng_states_all_ranks, rng_states)
            torch.distributed.barrier()
        else: # rank-0 save and resubmit job
            # Save all trainable parameters
            core_model = self.model_S
            if hasattr(core_model, "module"):
                core_model = core_model.module
            model_states = core_model.state_dict()
            
            # dataloder: current epoch and num batches yielded within the epoch
            dataloader_states = dataloader.state_dict()

            # optimizer, scheduler
            optim_states = optimizer.state_dict()
            scheduler_states = scheduler.state_dict()

            # progress bar: number of training steps
            progress_state = progress.state_dict()

            # random generator's states
            rng_states = get_rng()
            if self.world_size > 1:
                rng_states_all_ranks = [None]*self.world_size
                torch.distributed.all_gather_object(rng_states_all_ranks, rng_states)
                rng_states = rng_states_all_ranks
            all_states = {
                          'model': model_states,
                          'optimizer': optim_states,
                          'scheduler': scheduler_states,
                          'dataloader': dataloader_states,
                          'progress': progress_state,
                          'rng_states': rng_states
                         }

            # amp if use fp16
            if self.t_config.fp16:
                all_states['amp'] = amp.state_dict()

            # projs if used
            if hasattr(self, "projs"):
                projs_states = []
                for proj in self.projs:
                    if proj is not None:
                        if hasattr(proj, "module"):
                            proj = proj.module
                        projs_states.append(proj.state_dict())
                all_states['projs'] = projs_states

            save_to = os.path.join(self.t_config.output_dir, 'restore.ckpt')
            torch.save(all_states, save_to)
        
            # submit a new job
            logger.info('Resubmit to resume ...')
            resume_cmd_file = os.path.join(self.t_config.output_dir,
                                           'resume.sh')
            resume_cmd = open(resume_cmd_file, 'r').read()
            msg = subprocess.check_output(resume_cmd, shell=True)
            logger.info(msg.decode('utf-8'))
            if self.world_size > 1:
                torch.distributed.barrier()

        # All ranks quit
        exit(0)


    def basic_eval(self, eval_dataloader, model, batch_postprocessor=None, **kwargs):
        """
            Calculate the validation losses in the same way as training.
        """
        self.model_S.eval()
        accumulate_loss = accumulated_loss() # report average loss over validation
        with torch.no_grad():
            for batch in eval_dataloader:
                if batch_postprocessor is not None:
                    batch = batch_postprocessor(batch)
                total_loss, losses_dict = self.train_on_batch(batch, kwargs)
                accumulate_loss.add(total_loss, losses_dict)
        eval_loss, eval_losses_dict = accumulate_loss.report()
        if self.world_size > 1:
            sorted_loss_keys = eval_losses_dict.keys()
            all_losses = torch.stack([eval_loss] + [eval_losses_dict[_] for _ in sorted_loss_keys])
            torch.distributed.all_reduce(all_losses, op=torch.distributed.ReduceOp.AVG)
            eval_loss = all_losses[0]
            for i, k in enumerate(sorted_loss_keys):
                eval_losses_dict[k] = all_losses[i + 1]
        return eval_loss, eval_losses_dict


    def freeze_student_unused_params(self, batch_postprocessor, **kwargs):
        # This function shuuld be called before we wrap the model into DDP
        def fwd_func(model_S, batch, **kwargs):
            if batch_postprocessor is not None:
                batch = batch_postprocessor(batch)
            (teacher_batch, results_T), (student_batch, results_S) = \
                get_outputs_from_batch(batch, self.local_rank,
                                       self.model_T, model_S, kwargs)
            results_S = post_adaptor(self.adaptor_S(student_batch,results_S))
            results_T = post_adaptor(self.adaptor_T(teacher_batch,results_T))
            total_loss, _ = self.compute_loss(results_S,results_T)
            return total_loss
        frozen_params = freeze_unused_params(self.model_S, fwd_func, **kwargs)
        if self.rank == 0:
            logger.info("The following parameters are frozen:\n{}".format(
                        ", ".join(frozen_params))
                       )
