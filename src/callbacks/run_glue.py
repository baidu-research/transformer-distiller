"""
    Finetuning a student model for sequence classification on GLUE.
    adapted from huggingface transformers
    examples/pytorch/text-classification/run_glue.py

    For simplicity, we launch non-DDP jobs by setting env var: LOCAL_RANK=-1

    One launch slurm job like this:
    sbatch -p $partition -N 1 -n 1 --gres=gpu:$n_gpu --wrap 'srun python run_glue ...'
"""
import argparse
import logging
import os
import torch
import sys
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from collections import OrderedDict
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

os.environ['LOCAL_RANK'] = '-1'
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)
# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

def parse_args():
    @dataclass
    class ModelArguments:
        model_config: str = field(
            default=None,
            metadata={'help': 'path to a (student) model config file (.json)'})
        model_ckpt: str = field(
            default=None,
            metadata={'help': 'path to a (student) model checkpoint, '
            'a binary file of pytorch state_dict'})
        published_model: str = field(
            default=None,
            metadata={'help': 'path to a published model, e.g., bert-base-uncased.'
                      'This argument is mutual exclusive with model_config and '
                      'model_ckpt. The purpose is to get a result from published '
                      'model for comparison with student model'}
            )

    @dataclass
    class DataArguments:
        task_name: str = field(
            metadata={'help': f'choose one glue task from {list(task_to_keys.keys())}'})
        tokenizer_name: str = field(
            metadata={'help': 'name of tokenizer. Check '
            'https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoTokenizer'
            'for a complete list of supported tokenizers. '
            'Usually, the tokenizer is the teacher model name.'})
        max_seq_length: int = field(
            default=512,
            metadata={'help': 'max number of tokens in an input sequence'})
        pad_to_max_length: bool = field(
            default=True,
            metadata={'help': 'Whether to pad all samples to `max_seq_length`. '
            'If False, will pad the samples dynamically when batching to the '
            'maximum length in the batch.'})
        max_train_percentage: Optional[float] = field(
            default=100,
            metadata={'help': 'For studying sample efficiency, use a fraction '
            'of training set'})


    parser = HfArgumentParser([ModelArguments, DataArguments, TrainingArguments]) 
    parser.set_defaults(do_train=True, do_eval=True, do_predict=False,
                        evaluation_strategy='epoch', log_level='info')
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    return model_args, data_args, training_args


def load_model_for_seqClassify(model_args, label_list, is_regression):
    num_labels = 1 if is_regression else len(label_list)
    if model_args.published_model is not None:
        assert model_args.model_config is None and model_args.model_ckpt is None
        model = AutoModelForSequenceClassification.from_pretrained(
                    model_args.published_model,
                    num_labels=num_labels
                )
    else:
        assert model_args.model_config and model_args.model_ckpt
        config = AutoConfig.from_pretrained(model_args.model_config,
                                            num_labels=num_labels)
        model = AutoModelForSequenceClassification.from_config(config)
        ckpt = torch.load(model_args.model_ckpt, map_location='cpu')
        model_state_dict_keys = list(model.state_dict().keys())
        # Infer model class, e.g., BERT, etc. We take advantage of the fact
        # that often the first saved parameter is lookup embedding's weights,
        # and the parameter name starts with model class
        model_cls = model_state_dict_keys[0].split('.')[0]
        bare_model_keys = [_ for _ in model_state_dict_keys if
                           _.startswith(f'{model_cls}.')]
        # All model keys except those due to classifcation head should exist
        # in ckpt's keys. Also, note that at distillation, one may work with a
        # bare student model. Consequently, the keys in saved checkpoint can
        # miss the model_cls (e.g., bert) at their beginnings
        distil_with_bare = bare_model_keys[0] not in ckpt and \
                            '.'.join(bare_model_keys[0].split('.')[1:]) in ckpt
        for key in bare_model_keys:
            key_split = key.split('.')
            if distil_with_bare:
                assert key not in ckpt
                assert '.'.join(key_split[1:]) in ckpt
                ckpt[key] = ckpt.pop('.'.join(key_split[1:]))
            else:
                assert key in ckpt
        # Now load the weights in the ckpt
        missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
        # keys due to tasks-specfic head can be different. Other than that,
        # all keys are bare model's keys, thus should be shared.
        assert all([not k.startswith(f'{model_cls}.') for k in missing_keys])
        assert all([not k.startswith(f'{model_cls}.') for k in unexpected_keys])

    # match dataset's label and the one in model config
    if not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in model.config.label2id.items()}
    return model


def main():
    logger.info(f"debug local rank = {os.environ.get('LOCAL_RANK', -1)}")
    model_args, data_args, training_args = parse_args()
    set_seed(training_args.seed)
    raw_datasets = load_dataset('glue', data_args.task_name)
    # dataset misc
    is_regression = data_args.task_name == "stsb"
    if not is_regression:
        label_list = raw_datasets["train"].features["label"].names
    else:
        label_list = None
    sentence1_key, sentence2_key = task_to_keys[data_args.task_name]

    # Load ckpt and tokenier
    model = load_model_for_seqClassify(model_args, label_list, is_regression)
    tokenizer = AutoTokenizer.from_pretrained(data_args.tokenizer_name,
                                              use_fast=True)

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
        return result

    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        desc="Running tokenizer on dataset",
    )
    train_dataset = raw_datasets["train"]
    if data_args.max_train_percentage != 100:
        max_train_samples = int(data_args.max_train_percentage/100 * len(train_dataset))
        train_dataset = train_dataset.select(range(max_train_samples))
    else:
        max_train_samples = len(train_dataset)

    if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
        raise ValueError("--do_eval requires a validation dataset")
    eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]

    if "test" not in raw_datasets and "test_matched" not in raw_datasets:
        raise ValueError("--do_predict requires a test dataset")
    predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]

    # Get the metric function
    metric = load_metric("glue", data_args.task_name)
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    train_result = trainer.train()
    metrics = train_result.metrics
    metrics["train_samples"] = max_train_samples

    trainer.save_model()  # Saves the tokenizer too for easy upload

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Evaluation
    logger.info("*** Evaluate ***")
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    tasks = [data_args.task_name]
    eval_datasets = [eval_dataset]
    if data_args.task_name == "mnli":
        tasks.append("mnli-mm")
        eval_datasets.append(raw_datasets["validation_mismatched"])
        combined = {}

    for eval_dataset, task in zip(eval_datasets, tasks):
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        metrics["eval_samples"] = len(eval_dataset)

        if task == "mnli-mm":
            metrics = {k + "_mm": v for k, v in metrics.items()}
        if task is not None and "mnli" in task:
            combined.update(metrics)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", combined if task is not None and "mnli" in task else metrics)

    logger.info("*** Predict ***")

    # Loop to handle MNLI double evaluation (matched, mis-matched)
    tasks = [data_args.task_name]
    predict_datasets = [predict_dataset]
    if data_args.task_name == "mnli":
        tasks.append("mnli-mm")
        predict_datasets.append(raw_datasets["test_mismatched"])

    for predict_dataset, task in zip(predict_datasets, tasks):
        # Removing the `label` columns because it contains -1 and Trainer won't like that.
        predict_dataset = predict_dataset.remove_columns("label")
        predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
        predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

        output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{task}.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                logger.info(f"***** Predict results {task} *****")
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    if is_regression:
                        writer.write(f"{index}\t{item:3.3f}\n")
                    else:
                        item = label_list[item]
                        writer.write(f"{index}\t{item}\n")


if __name__ == "__main__":
    main()
