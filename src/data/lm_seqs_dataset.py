"""
Reads binarized data produced by scripts/h5_data.py, which converts raw text
to .h5 file.

Adapted from:
https://github.com/huggingface/transformers/tree/master/examples/research_projects/distillation/lm_seqs_dataset.py
"""
import h5py
import itertools
import logging
import math
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from typing import Sequence, Optional
from ..utils import global_rank


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class LmSeqsDataset(Dataset):
    """Custom Dataset wrapping language modeling sequences.

    Input:
    ------
        data: list of binarized data (.h5) files
        tokenizer_name: name of huggingface Tokenizer, e.g., 'roberta-base'
        token_counts_files: list of token counts files (binary)
        length_threshold: exclude sentences whose length <= length_threshold
        unk_rate_threshold: exclude sentences whose <unk> ratio >= this threshold
    """

    def __init__(self,
                 data: Sequence[str],
                 tokenizer_name: str,
                 token_counts_files: Optional[Sequence[str]]=None,
                 mlm: Optional[bool]=True):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.mlm = mlm
        self.token_ids = []
        for i, data_i in enumerate(data):
            self.token_ids.append(h5py.File(data_i, 'r')['token_ids'])
        self.ragged = self.token_ids[0].dtype == 'O'

        # load token count file for calculating sampling probabilities
        if token_counts_files is None:
            self.token_probs = None
        else:
            self.token_probs = np.zeros(self.tokenizer.vocab_size)
            for t_cnt_f in token_counts_files:
                self.token_probs += np.array(
                                        pickle.load(open(t_cnt_f, 'rb')),
                                        dtype=np.float
                                    )
            self.token_probs /= np.sum(self.token_probs)
            self.token_probs = torch.tensor(self.token_probs)

        self.infer_special_token_rule()

        # offset[i]: number of sequences up to the i-th "sub" data file
        self.offset = list(itertools.accumulate(
                            [t_ids.shape[0] for t_ids in self.token_ids]))
        self.offset_l =  [0] + self.offset[:-1]
        self.print_statistics()

    def __getitem__(self, index):
        for i, off in enumerate(self.offset):
            if off > index:
                idx2 = index - self.offset_l[i]
                seq = self.token_ids[i][idx2]
                return (seq, len(seq))

    def __len__(self):
        return self.offset[-1]

    def infer_special_token_rule(self):
        check = 100
        first_ids = [row[0] for row in self.token_ids[0][:check]]
        last_ids = [row[-1] for row in self.token_ids[0][:check]]

        # prepend special token at beginning?
        if all([_==first_ids[0] for _ in first_ids]) and \
                first_ids[0] in self.tokenizer.all_special_ids:
            self.prepend = first_ids[0]
        else:
            self.prepend = None

        # append special token at end?
        if all([_==last_ids[0] for _ in last_ids]) and \
                last_ids[0] in self.tokenizer.all_special_ids:
            self.append = last_ids[0]
        else:
            self.append = None

    def print_statistics(self):
        """
        Print some statistics on the corpus.
        """
        if global_rank == 0:
            logger.info(f"{len(self)} sequences")
        

    def batch_sequences(self, batch):
        """
        The collate function for torch dataLoader:
        Pad to longest sentence in the batch and transform into torch.tensor.
        """
        token_ids = [t[0].astype('int') for t in batch]
        lengths = [t[1] for t in batch]
        assert len(token_ids) == len(lengths)

        if self.ragged:
            # Pad token ids
            if self.mlm:
                pad_idx = self.tokenizer.pad_token_id
            else:
                pad_idx = self.tokenizer.unk_token_id
            tk_t = pad_sequence(
                [torch.tensor(_) for _ in token_ids],
                batch_first=True,
                padding_value=pad_idx
            ) # (bs, max_seq_len_)
        else:
            tk_t = torch.tensor(np.array(token_ids))

        return tk_t, torch.tensor(lengths)


    def prepare_batch_mlm(self, batch, pred_rate=0.15, mask_rate=0.8, rand_rate=0.1, keep_rate=0.1):
        """
        Post-processing the batch: from the token_ids and the lengths,
        compute the attention mask and the masked label for MLM.
        Adapted from https://github.com/huggingface/transformers/blob/master/examples/research_projects/distillation/distiller.py: prepare_batch_mlm
        
                
        Follow the standard bert mask prediction setup.
        `pred_rate` tokens are randomly selected to be predicted
        `mask_rate` tokens among them are replaced by [MASK] at input
        `rand_rate` tokens among them are replaced by random tokens at input
        `keep_rate` tokens among them are kept.

        In standard bert sets, these values are 0.15, 0.8, 0.1, 0.1
        Input:
        ------
            batch: `Tuple`
                token_ids: `torch.tensor(bs, seq_length)` - The token ids for each of the sequence. It is padded.
                lengths: `torch.tensor(bs)` - The lengths of each of the sequences in the batch.
            pred_rate, mask_rate, rand_rate, keep_rate: `(float)`
        
        Output:
        -------
            token_ids: `torch.tensor(bs, seq_length)` - The token ids after the modifications for MLM.
            attn_mask: `torch.tensor(bs, seq_length)` - The attention mask for the self-attention.
            mlm_labels: `torch.tensor(bs, seq_length)` - The masked language modeling labels. There is a -100 where there is nothing to predict.
        """
        assert self.tokenizer.mask_token_id is not None
        token_ids, lengths = batch
        assert token_ids.size(0) == lengths.size(0)
        attn_mask = torch.arange(token_ids.size(1), dtype=torch.long) < lengths[:, None]
        if pred_rate == 0: # If not use logit as distillation signal
            return {
                    'input_ids': token_ids,
                    'attention_mask': attn_mask,
                   }

        bs, max_seq_len = token_ids.size()
        mlm_labels = token_ids.new(token_ids.size()).copy_(token_ids)
        # number of tokens to be masked or noised
        n_tgt = math.ceil(pred_rate * lengths.sum().item())
        
        # probability proportional to token unigram probability, or simply uniform 
        # exclude paddings, and bos, eos if they are there
        if self.token_probs is not None:
            x_prob = self.token_probs[token_ids]
        else:
            x_prob = torch.ones_like(token_ids, dtype=torch.float)
        # for eos, bos: masking prob is 0
        if self.prepend is not None:
            x_prob[:, 0] = 0
        if self.append is not None:
            x_prob[range(bs), lengths-1] = 0
        # for padding: masking prob is 0
        x_prob *= attn_mask
        tgt_ids = torch.multinomial(x_prob.flatten(),
                                    n_tgt, replacement=False)
        # create a mask, 1 for the position where a token is masked
        pred_mask = torch.zeros(
                      bs * max_seq_len, dtype=torch.bool, device=token_ids.device
                    )
        pred_mask[tgt_ids] = 1
        pred_mask = pred_mask.view(bs, max_seq_len)

        # Input at pred_mask is replaced by [mask], random token, or kept.
        _token_ids_real = token_ids[pred_mask]
        _token_ids_rand = _token_ids_real.clone().random_(self.tokenizer.vocab_size)
        _token_ids_mask = _token_ids_real.clone().fill_(self.tokenizer.mask_token_id)
        probs = torch.multinomial(
                    torch.tensor([keep_rate, rand_rate, mask_rate], dtype=torch.float),
                    n_tgt,
                    replacement=True
                )
        replace_token_ids = (
            _token_ids_real * (probs == 0)
            + _token_ids_rand * (probs == 1)
            + _token_ids_mask * (probs == 2)
        )
        token_ids = token_ids.masked_scatter(pred_mask, replace_token_ids)
        mlm_labels[~pred_mask] = -100  # torch.nn.CrossEntropyLoss ignores
        return {'input_ids': token_ids, 'attention_mask': attn_mask, 'labels': mlm_labels}
