import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from ..utils import global_rank, world_size


class singleRankSampler(Sampler):
    """
        Generate indices, controlled by initial random seed + epoch count.
        Basically a simplfied version of pyTorch's DistributedSampler
        Refer to https://pytorch.org/docs/stable/_modules/torch/utils/data/distributed.html#DistributedSampler
        Here we don't have issues of num_replicas and drop_last
    """
    def __init__(self, dataset: Dataset,
                 shuffle: bool = True,
                 seed: int = 0):
        super(singleRankSampler, self).__init__(dataset)
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        return iter(indices)

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch: int):
        self.epoch = epoch


class infinite_DataLoader:
    """
        Wraps a standard torch DataLoader to return batches infinitely.
        After finishing one epoch, data are shuffled and yielded again.

        Also, enables yielding batches from middle of an epoch, by calling
        `load_state_dict`
    """
    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.epoch_length = len(dataloader)
        self.reset()

    def reset(self):
        # number of rounds that have gone over
        self.epoch = 0
        self.dataloader.sampler.set_epoch(0)
        # number of minibatches that have been returned in this epoch
        self.num_yielded = 0
        # dataloader as an iterator
        self.dataloader_iter = iter(self.dataloader)

    def state_dict(self):
        """
            Current epoch and minibatch within this epoch
        """
        state_dict = {
                'epoch': self.epoch,
                'num_yielded': self.num_yielded
        }
        return state_dict

    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        self.dataloader.sampler.set_epoch(self.epoch)
        self.num_yielded = state_dict['num_yielded']
        self.dataloader_iter = iter(self.dataloader)
        for step in range(self.num_yielded):
            next(self.dataloader_iter)

    def __iter__(self):
        return self

    def __next__(self):
        self.num_yielded += 1
        batch = next(self.dataloader_iter)
        if self.num_yielded == self.epoch_length:  # epoch ends
            self.num_yielded = 0
            self.epoch += 1
            self.dataloader.sampler.set_epoch(self.epoch)
            self.dataloader_iter = iter(self.dataloader)
        return batch


def create_dataLoader(dataset, args, collate_fn=None, is_training=False):
    """
        Create single rank or distributed dataLoader on the torch dataset:
        dataset: an instance of torch.utils.data.Dataset
        args: Input parameters parsed
        collate_fn: If None (default), will use the `batch_sequence` method of
            the `dataset`
        is_infinite: if True (often the case for training set), further
            wrap the dataLoader in infinite_DataLoader
    """
    if world_size > 1:
        sampler = DistributedSampler(
                    dataset,
                    num_replicas=world_size,
                    rank=global_rank,
                    shuffle=args.shuffle if is_training else False
                  )
    else:
        sampler = singleRankSampler(
                    dataset,
                    args.shuffle if is_training else False
                  )

    dataLoader = DataLoader(
                    dataset,
                    args.bsz//args.grad_acc_step//world_size,
                    sampler=sampler,
                    collate_fn=collate_fn if collate_fn else dataset.batch_sequences,
                    num_workers=4,
                    pin_memory=True
                 )
    if is_training:
        dataLoader = infinite_DataLoader(dataLoader)
    return dataLoader
