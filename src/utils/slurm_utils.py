import logging
import os
import subprocess
import torch
from torch.distributed import init_process_group

# In most cases, jobs are submitted via sbatch.
# We set following os environment vars to init process group.
# If the job is launched locally, these os environment vars becomes futile.
os.environ['LOCAL_RANK'] = os.environ.get('SLURM_LOCALID', '0')
os.environ['RANK'] = os.environ.get('SLURM_PROCID', '0')
os.environ['WORLD_SIZE'] = os.environ.get('SLURM_NTASKS', '1')
hostnames = subprocess.check_output(
        ['scontrol', 'show', 'hostnames',
          os.environ.get('SLURM_JOB_NODELIST', '0')])
os.environ['MASTER_ADDR'] = hostnames.split()[0].decode('utf-8')
os.environ['MASTER_PORT'] = str(int(os.environ.get('SLURM_JOB_ID', '0')) % 32768 + 32767)

# For easier access of rank information
# world_size == 1 is an indicator of single GPU job
local_rank = int(os.environ['LOCAL_RANK'])
global_rank = int(os.environ['RANK'])
world_size = int(os.environ['WORLD_SIZE'])

if world_size > 1:
    torch.cuda.set_device(local_rank)
    init_process_group(backend='nccl')
