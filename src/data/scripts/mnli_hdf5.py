from datasets import load_dataset

data = load_dataset('glue', 'mnli', split='train')
with open('/tmp/mnli_train.txt', 'w') as f:
    for sentence in data['hypothesis']:
        f.write(sentence+'\n')
#python h5_data.py \
#    --data_files /tmp/mnli_train.txt \
#    --tokenizer_name bert-base-uncased \
#    --dump_file /mnt/data/jiajihuang/model_compression/eval_data/mnli_ragged.hdf5 \
#    --ragged
