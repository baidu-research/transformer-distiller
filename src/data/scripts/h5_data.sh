#!/bin/bash
# Suppose book and wiki text data stored under two dirs
# An example using bert-base-uncased tokenizer to preprocess the data
ragged=$1
tokenizer_name=${2:-bert-base-uncased}
save_at=${3:-/mnt/data/jiajihuang/model_compression/pretrain_data/h5}
book=$YOUR_PATH_TO_bookcorpus_dir
wiki=$YOUR_PATH_TO_wiki_dir

for dataname in book wiki; do
  dump_to=${save_at}/${dataname}/${tokenizer_name}
  mkdir -p $dump_to
  if [[ $dataname == "book" ]]
  then
    raw_data=$book
  else
    raw_data=$wiki
  fi
  
  if [[ $ragged -eq 0 ]];
  then
    h5_name=data.h5
    python h5_data.py \
      --data_files ${raw_data}/* \
      --tokenizer_name ${tokenizer_name} \
      --dump_file ${dump_to}/${h5_name}
  else
    h5_name=data_ragged.h5
    python h5_data.py \
      --data_files ${raw_data}/* \
      --tokenizer_name ${tokenizer_name} \
      --dump_file ${dump_to}/${h5_name} \
      --ragged
  fi

  # after we created the h5 files, get token counts
  python token_counts.py --data_files ${dump_to}/data_ragged.h5 --token_counts_dump ${dump_to}/token.counts --tokenizer_name ${tokenizer_name}
done
