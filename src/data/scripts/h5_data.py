"""
Data preprocessing before distillation. Convert text file into int16 array.
There are two options:
1) ragged: keep the original sentence boundary.
    There will be two fields in the generated .h5 data:
    'token_ids': 1D array of concatenated token ids
    'lengths': 1D array of each sentence's length
2) non-ragged (default): concatenate all token ids and ignore sentence boundary
    There will be a single field in the generated .h5 data:
    'token_ids': 2D array of #sequences x #max_input_length
"""
import argparse
import logging
import time
import h5py

import numpy as np

from transformers import AutoTokenizer


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess the data to avoid tokenization every time we "
        "run training."
    )
    parser.add_argument("--data_files", type=str, nargs='+', default="data",
                        help="The path to the data files.")
    parser.add_argument("--tokenizer_name", default="bert-base-uncased",
                        help="The tokenizer to use.")
    parser.add_argument("--dump_file", default="data/dump",
                        help="The dump file prefix.")

    # The following are only useful if use "ragged" batches, i.e.,
    # sentence boundaries are kept
    parser.add_argument("--ragged", action="store_true",
                        help="If specified, keep the original sentence "
                        "boundaries instead of concatenating them.")
    parser.add_argument("--length_threshold", type=int, default=0,
                        help="Only useful when ragged=True, "
                        "sentence whose length <= this value will be excluded")
    parser.add_argument('--unk_rate_threshold', type=float, default=0.0,
                        help="Only useful when ragged=True, "
                        "sentences whose <unk> token rate >= this value " 
                        "will be excluded")
    args = parser.parse_args()

    logger.info(f"Loading Tokenizer ({args.tokenizer_name})")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    max_input_len = tokenizer.model_max_length
    max_actual_text_len = tokenizer.max_len_single_sentence
    try:
        unk_token_id = tokenizer.unk_token_id
    except AttributeError:
        unk_token_id = None

    # decide if bos or eos is used
    dummy = tokenizer.encode('This is a dummy sentence.')
    prepend = dummy[0] if dummy[0] in tokenizer.all_special_ids else None
    append = dummy[-1] if dummy[-1] in tokenizer.all_special_ids else None 

    # utility function: split very long sentence, and yield them
    def split_long(seq):
        if len(seq) <= max_input_len:
            yield seq
        else:
            # take the actual text and chunk it
            if prepend is not None:
                seq = seq[1:]
            if append is not None:
                seq = seq[:-1]
            for i in range(0, len(seq), max_actual_text_len):
                seq_i = seq[i:i+max_actual_text_len]
                if prepend is not None:
                    seq_i = [prepend] + seq_i
                if append is not None:
                    seq_i = seq_i + [append]
                yield seq_i
            
    rslt = []
    interval = 100000
    for data_file in args.data_files:
        logger.info(f"Processing data file: {data_file}")

        with open(data_file, "r", encoding="utf8", errors="ignore") as fp:
            data = fp.readlines()
        logger.info("Start encoding")
        logger.info(f"{len(data)} lines of text to process.")
        start = time.time()
        for i, text in enumerate(data):
            text = text.strip()
            if not text:
                continue
            if args.ragged:
                token_ids = tokenizer.encode(text)
                for sub_seq in split_long(token_ids):
                    if len(sub_seq) > args.length_threshold and \
                            (not args.unk_rate_threshold or \
                             sum([tk==unk_token_id for tk in sub_seq]) \
                                < args.unk_rate_threshold * len(sub_seq)): 
                        rslt.append(sub_seq)
            else:
                token_ids = tokenizer.encode(text, add_special_tokens=False)
                rslt.extend(token_ids)
            
            if i % interval == 0 and i != 0:
                end = time.time()
                logger.info(f"{i} lines processed. - {(end-start):.2f}s/{interval}expl")
                start = time.time()
        logger.info(f"Finished processing data file: {data_file}")

    if tokenizer.vocab_size < (1 << 16):
        dt = np.uint16
    else:
        dt = np.uint32

    if args.ragged:
        with h5py.File(args.dump_file, 'w') as of:
            of.create_dataset(
                'token_ids',
                data=[np.array(_, dtype=dt) for _ in rslt],
                dtype=h5py.vlen_dtype(dt)
            )
            logger.info(f"Total number of sequences: {len(rslt)}")
    else:
        n_rows, n_cols = len(rslt) // max_actual_text_len, max_actual_text_len
        rslt = np.reshape(rslt[: n_rows * n_cols], (n_rows, n_cols)).astype(dt)
        if prepend is not None:
            rslt = np.hstack((np.ones((n_rows, 1), dtype=dt)*prepend, rslt))
        if append is not None:
            rslt = np.hstack((rslt, np.ones((n_rows, 1), dtype=dt)*append))
        with h5py.File(args.dump_file, 'w') as of:
            of.create_dataset('token_ids', data=rslt)
        logger.info(f"Total number of sequences: {n_rows}")

    logger.info(f'H5 file saved to {args.dump_file}')

if __name__ == "__main__":
    main()
