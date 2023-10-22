import os, sys
# import transformers
from datasets import Dataset, load_dataset
import datasets
import numpy as np
import pandas as pd
import statistics
import logging

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import HfArgumentParser, DataCollatorWithPadding, TrainingArguments, Trainer
from transformers import AdamW, get_scheduler

from huggingface_hub import login
import evaluate
# from datasets import load_metric
from accelerate import Accelerator
from tqdm.auto import tqdm

from dataclasses import dataclass, field
from typing import Optional

##
from corpus.dataset_toefl_hf import Dataset_TOEFL
from collators.collator_toefl import CollatorPaddingTOEFL_Sent

# from model_avg_sent import Model_SentAvg
from models.model_avg_sent import Model_SentAvg

import hydra
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

# from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

# logger
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

######

def eval_acc(eval_dataloader, model):
    # accuracy eval
    correct = 0
    total = 0
    for batch in eval_dataloader:
        # predict and label
        outputs = model(batch)
        labels = batch["labels"]

        # calcualte acc
        _, predicted = torch.max(outputs, 1)  # model_output: (batch_size, num_class_out)
        correct += (predicted == labels).sum().item()
        total += predicted.size(0)
        
    accuracy = correct / float(total)

    return accuracy

# def training():

#     return

def setup_optimizer(cfg, model, train_dataloader):
    optimizer = AdamW(model.parameters(), lr=cfg.training.learning_rate)
    num_training_steps = cfg.training.num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    return optimizer, lr_scheduler, model, num_training_steps

def load_data(path_data, pretrained_weights, batch_size, target_prompt, cur_fold, tokenize_method):

    tokenizer = AutoTokenizer.from_pretrained(pretrained_weights, low_cpu_mem_usage=True, use_auth_token=True)
    # deal with pad tokens
    # tokenizer.add_special_tokens({"pad_token":"<pad>"})
    tokenizer.pad_token = tokenizer.eos_token

    encoder = AutoModel.from_pretrained(pretrained_weights, low_cpu_mem_usage=True, use_auth_token=True, offload_folder="offload", offload_state_dict = True, max_memory={"cpu": "10GIB"}) # , torch_dtype=torch.bfloat16)
    # disabling graident updating for the encoder
    for param in encoder.parameters():
        param.requires_grad = False

    ## pp data
    dataset_target = Dataset_TOEFL(tokenizer=tokenizer, pretrained_weights=pretrained_weights)

    ## sentence tokenization     
    # * note that it filters documents which couln't be sentence tokenized or less than a few (it happens they have serious syntax error and as Stanza library updates)
    dataset_target.tokenize_sents_save_pd(path_data, cur_fold=cur_fold, force_sent_tokenize=False)  # add sentence segmentation to dataframe (need to be done once)
    tokenized_datasets = dataset_target.load_hf_dataset(path_data, target_prompt, cur_fold=cur_fold, tokenize_method=tokenize_method)  # key: "train", 

    # tokenized_datasets = map_splits.map(dataset_target.tokenize_map, batched=False)  # features: ['essay_id', 'prompt', 'native_lang', 'essay_score', 'essay', 'essay_sents', 'input_ids', 'attention_mask'],
    # id_texts = tokenized_datasets["essay_id"]  # for analysis in the evaluation
    tokenized_datasets = tokenized_datasets.remove_columns(["__index_level_0__", "essay_id", "prompt",  "essay", "essay_sents", "native_lang"])
    tokenized_datasets = tokenized_datasets.rename_column("essay_score", "labels")

    ## data loader
    # data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # batch_size = 32
    data_collator = CollatorPaddingTOEFL_Sent(tokenizer=tokenizer, pad_token=tokenizer.pad_token, max_num_sent=dataset_target.max_num_sent, max_len_sent=dataset_target.max_len_sent)
    
    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    eval_dataloader = DataLoader(tokenized_datasets["validate"], shuffle=False, batch_size=batch_size, collate_fn=data_collator)


    return tokenizer, encoder, dataset_target, train_dataloader, eval_dataloader

## TODO
# def hydra_init(cfg_name="config") -> None:
#     # hydra_init
#     cs = ConfigStore.instance()
#     cs.store(name=f"{cfg_name}", node=Config)

#     for k in Config.__dataclass_fields__:
#         v = Config.__dataclass_fields__[k].default
#         try:
#             cs.store(name=k, node=v)
#         except BaseException:
#             # logger.error(f"{k} - {v}")
#             raise

######

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg : DictConfig) -> None:

    ## Configurations
    # hydra_init(cfg_name="config")
    # cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True, enum_to_str=True))

    # HuggingFace token for login
    hf_token = cfg.hf_token
    if hf_token is not None and len(hf_token) > 0:   login(token=hf_token)
    else: logger.info("HF Token is not provided: some LLMs might not be available")

    ## load pretrained tokenizer and encoder (pass through yaml or override cli)
    # pretrained_weights = "meta-llama/Llama-2-7b-hf"
    # pretrained_weights = "meta-llama/Llama-2-7b-chat-hf"
    # pretrained_weights = "facebook/opt-350m"
    # pretrained_weights = "xlnet-base-cased"

    ## load data and encoder
    eval_folds = []  # evaluation score in each fold in cross-validation
    for cur_fold in range(cfg.dataset.num_fold):
        ## Load data and encoder
        tokenizer, encoder, dataset_target, train_dataloader, eval_dataloader = load_data(cfg.dataset.path_data, 
                                                                                          cfg.model.pretrained_weights, 
                                                                                          cfg.dataset.batch_size, 
                                                                                          cfg.dataset.target_prompt, 
                                                                                          cur_fold,
                                                                                          cfg.dataset.tokenize_method)

        ## modeling
        model = Model_SentAvg(encoder, dataset_target.output_size, cfg.model.activation_fn)

        ## optimizer
        optimizer, lr_scheduler, model, num_training_steps = setup_optimizer(cfg, model, train_dataloader) 
                                                                      
        ##  distribution by HF accelerator
        accelerator = Accelerator()
        # accelerator = Accelerator(gradient_accumulation_steps=2)  # control graident steps
        train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(train_dataloader, eval_dataloader, model, optimizer)
        progress_bar = tqdm(range(num_training_steps))

        ## training and eval
        best_eval = 0
        for epoch in range(cfg.training.num_epochs):
            model.train()
            for batch in train_dataloader:
                # with accelerator.accumulate(model):  # when graident_accumlation_steps is used
                outputs = model(batch)
                labels = batch["labels"]
                loss = F.cross_entropy(outputs, labels)
                # loss.backward()
                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

            ## eval (every epoch now, or put if condition here)
            model.eval()
            cur_eval = eval_acc(eval_dataloader, model)

            logger.info("Epoch {}, Eval: {}".format(epoch, cur_eval))
            best_eval = cur_eval if cur_eval > best_eval else best_eval

        eval_folds.append(best_eval)

        logger.info("--------")
        logger.info("Best Eval: {}".format(best_eval))

    logger.info("===========")
    logger.info(eval_folds)
    logger.info("Best Eval in CV: {}".format(statistics.mean(eval_folds)))


if __name__ == '__main__':
    main()
    
