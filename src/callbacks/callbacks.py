import logging
import os
import subprocess
import torch
import torch.nn as nn
from typing import Optional, List, Union
from ..utils import local_rank, global_rank, world_size
from ..textbrewer.configurations import TrainingConfig
from ..textbrewer.distiller_utils import AbstractDistiller, get_outputs_from_batch


logger = logging.getLogger("Callback")

class callbacks:
    # all implmented callbacks
    all_callbacks = ['log_PPL', 'glue']

    def __init__(self,
                 distiller: AbstractDistiller,
                 model_config: str, # the student model's config file
                 tokenizer: str, # the tokenizer, usually same as teacher name
                 callback_namelist: Optional[List[str]]=None, # subset of all_callbacks
                 **kwargs
                 ):
        self.model = distiller.model_S
        self.model_config = model_config
        self.model_ckpt_dir = distiller.t_config.output_dir
        self.tokenizer = tokenizer
        if callback_namelist is not None:
            assert all([cb in callbacks.all_callbacks for cb in callback_namelist])
        self.callback_namelist = callback_namelist
        self.tb_writer = distiller.tb_writer
        self.kwargs = kwargs
        self.set_gs()


    def set_gs(self, gs: int=0):
        """
            track number of training step, for easy access of latest ckpt where
            callback is run
        """
        self.global_step = gs


    def log_PPL(self, eval_dataloader, batch_postprocessor=None):
        """
        Calculate the log-perplexity of a (student) model on validation data.
        The student model is assumed to have a (trained) LM head.
        It could be masked LM or causal LM.
        The returned log-PPL is per-token cross-entropy loss.

        Args:
        model (:class:`torch.nn.Module`): student model.
        eval_dataloader (:class:`torch.utils.data.DataLoader`): validation set data loader.
        batch_postprocessor (Callable): processing batch data before feeding to student model.
        """
        if global_rank == 0:
            logger.info('Estimating log PPL on eval data ...')
        self.model.eval()
        all_tok_loss = 0
        cnt = 0
        with torch.no_grad():
            for batch in eval_dataloader:
                if batch_postprocessor is not None:
                    batch = batch_postprocessor(batch)
                assert 'labels' in batch
                _, (_, results_S) = get_outputs_from_batch(batch,
                    local_rank, None, self.model, {}, no_teacher_forward=True)
                cnt_i = torch.sum(batch['labels'] != -100)
                all_tok_loss += results_S['loss'] * cnt_i
                cnt += cnt_i
        if world_size > 1:
            to_reduce = torch.stack([all_tok_loss,
                                     torch.tensor(cnt, device=local_rank)])
            torch.distributed.all_reduce(to_reduce)
            per_tok_loss = to_reduce[0] / to_reduce[1]
        else:
            per_tok_loss = all_tok_loss / cnt
        if global_rank == 0:
            logger.info(f'log_PPL on eval data: {per_tok_loss}')
        self.tb_writer.add_scalar(
            'callbacks/log_PPL', per_tok_loss, self.global_step)


    def glue(self, glue_tasks: Optional[Union[str, List[str]]]='mnli', **kwargs):
        """
        Submit slurm jobs that finetune a student model on multiple GLUE tasks.
        For simplicity, we only launch single GPU jobs.
        As this job is separated from the main process (distillation training),
        it is not very urgent. So it suffices to run as a single GPU job.
        """
        if global_rank != 0:
            torch.distributed.barrier()
        else:
            partition = '2080Ti'
            pwd = os.path.dirname(__file__)
            ckpt = os.path.join(self.model_ckpt_dir,
                                f'{self.global_step}.pkl')
            if isinstance(glue_tasks, str):
                glue_tasks = [glue_tasks]
            for task in glue_tasks:
                output_dir = os.path.join(self.model_ckpt_dir,
                                          f'{self.global_step}-{task}')
                job_log = os.path.join(self.model_ckpt_dir,
                                       f'{self.global_step}-{task}.log')
                py_cmd = (f'python {os.path.join(pwd, "run_glue.py")} '
                          f'--model_config {self.model_config} '
                          f'--model_ckpt {ckpt} '
                          f'--tokenizer_name {self.tokenizer} '
                          f'--task_name {task} '
                          f'--max_seq_length {kwargs.get("max_seq_length", 128)} '
                          f'--optim {kwargs.get("optim", "adamw_torch")} '
                          f'--per_device_train_batch_size {kwargs.get("batch_size", 32)} '
                          f'--per_device_eval_batch_size {kwargs.get("batch_size", 32)} '
                          f'--learning_rate {kwargs.get("lr", 2e-5)} '
                          f'--num_train_epochs {kwargs.get("epochs", 3)} '
                          f'--save_strategy no --disable_tqdm True '
                          f'--logging_first_step True --logging_steps 1 '
                          f'--output_dir {output_dir}')
                #py_cmd = f'python {os.path.join(pwd, "debug_env_var.py")}'
                job_cmd = (f'sbatch -p {partition} -N 1 -n 1 '
                           f'--gres=gpu:1 -o {job_log} '
                           f'--wrap "srun {py_cmd}"')
                msg = subprocess.check_output(job_cmd, shell=True)
                job_id = msg.decode('utf-8').split()[-1]
                logger.info(f'Submitted job {job_id}: glue-{task}, log at {job_log}')
                logger.info('Submission command:')
                logger.info(job_cmd)
            if world_size > 1:
                torch.distributed.barrier()


    def apply(self):
        if self.callback_namelist is None:
            return
        for cb in self.callback_namelist:
            if cb == 'log_PPL':
                assert 'eval_dataloader' in self.kwargs
                self.log_PPL(self.kwargs['eval_dataloader'],
                             self.kwargs.get('batch_postprocessor')
                            )
            elif cb == 'glue':
                self.glue(**self.kwargs)
