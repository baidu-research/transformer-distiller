# Copied and adapted from https://github.com/huggingface/transformers/blob/master/examples/research_projects/distillation/scripts/token_counts.py
# Key differences:
# Original: handles only one binary data file
# Here: can handle multiple binary data files
# 
"""
Preprocessing script before training the distilled model.
"""
import argparse
import h5py
import logging
import numpy as np
import pickle
from collections import Counter
from transformers import AutoTokenizer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Token Counts for smoothing the masking probabilities in MLM (cf XLM/word2vec)"
    )
    parser.add_argument(
        "--data_files", type=str, nargs='+', default="data/dump.bert-base-uncased.h5", help="The binarized dataset(s)."
    )
    parser.add_argument(
        "--token_counts_dump", type=str, default="data/token_counts.bert-base-uncased.pickle", help="The dump file."
    )
    parser.add_argument('--tokenizer_name', type=str, default='bert-base-uncased', help='name of the tokenizer used to produce the binary data')
    args = parser.parse_args()

    vocab_size = AutoTokenizer.from_pretrained(args.tokenizer_name).vocab_size
    counter = Counter()
    for data_file in args.data_files:
        logger.info(f"Loading data from {data_file}")
        with h5py.File(data_file, "r") as fp:
            data = fp['token_ids']
            logger.info("Counting occurences for MLM.")
            for i in range(0, len(data), 65536):
                counter.update(np.hstack(data[i:i+65536]))

    counts = [0] * vocab_size
    for k, v in counter.items():
        counts[k] = v

    logger.info(f"Dump to {args.token_counts_dump}")
    with open(args.token_counts_dump, "wb") as handle:
        pickle.dump(counts, handle, protocol=pickle.HIGHEST_PROTOCOL)
