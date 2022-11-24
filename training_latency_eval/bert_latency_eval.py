import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
import time

import datasets
import numpy as np
from datasets import load_dataset
import torch
from matplotlib import pyplot as plt

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    BertTokenizer,
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_cola_dataset():

    raw_datasets = load_dataset(
        "glue",
        "cola",
        cache_dir="./data",
        #use_auth_token=True if model_args.use_auth_token else None,
    )

    # Load the BERT tokenizer.
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    train_input_ids = []
    train_input_mask = []
    labels = []

    # Tokenize all of the sentences and map the tokens to thier word IDs.
    # For every sentence...
    for sent in raw_datasets['train']:
        encoded_dict = tokenizer.encode_plus(
                            sent['sentence'],                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 64,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                    )
        # Add the encoded sentence to the list.    
        train_input_ids.append(encoded_dict['input_ids'])
        # And its attention mask (simply differentiates padding from non-padding).
        train_input_mask.append(encoded_dict['attention_mask'])
        labels.append(sent['label'])
    # Convert the lists into tensors.
    input_ids = torch.cat(train_input_ids, dim=0)
    attention_masks = torch.cat(train_input_mask, dim=0)
    labels = torch.tensor(labels)
    train_dataset = TensorDataset(input_ids, attention_masks, labels)

    # The DataLoader needs to know our batch size for training, so we specify it 
    # here. For fine-tuning BERT on a specific task, the authors recommend a batch 
    # size of 16 or 32.
    batch_size = 32

    train_dataloader = DataLoader(
                train_dataset,  # The training samples.
                sampler = RandomSampler(train_dataset), # Select batches randomly
                batch_size = batch_size # Trains with this batch size.
    )

    return train_dataloader 

def get_bert_and_optimizer(train_dataloader):

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 2, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )

    optimizer = AdamW(model.parameters(),
                    lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                    eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
    )

    # Get timing data for 1 epoch
    epochs = 1
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)

    return  model, optimizer, scheduler


def finetune_BERT(
    model, 
    optimizer, 
    scheduler, 
    train_dataloader, 
    num_epochs=1
):


    # For each epoch...
    model.to(DEVICE)
    timing_dataset = []
    for epoch in range(0, num_epochs):
        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode. Don't be mislead--the call to 
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                
                # Report progress.
                print(f'Batch {step} of {len(train_dataloader)}')

            start = time.time()
            b_input_ids = batch[0].to(DEVICE)
            b_input_mask = batch[1].to(DEVICE)
            b_labels = batch[2].to(DEVICE)

            # Always clear any previously calculated gradients
            model.zero_grad()        

            # outputs = (loss, logits)
            outputs = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask, 
                                labels=b_labels)

            loss = outputs[0]
            total_train_loss += loss

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            end = time.time()
            timing_dataset.append(end - start)

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)            
        
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        
    return timing_dataset
        

def collect_timing_data_BERT():

    # Get datasset
    print("Collecting CoLA dataset")
    train_dataloader = get_cola_dataset()

    # Get model and training infrastructure
    print("Instantiating BERT")
    model, optimizer, scheduler = get_bert_and_optimizer(train_dataloader)

    # Train and time
    print("Finetuning BERT")
    timing_data = finetune_BERT(
        model, 
        optimizer, 
        scheduler,
        train_dataloader, 
        num_epochs=1
    )

    return timing_data 

def evaluate_BERT():

    timing_data = collect_timing_data_BERT()

    plt.ylabel("Minibatch latency (s)")
    plt.xlabel("Minibatch number")
    plt.title("BERT training latency")
    plt.plot([i for i in range(len(timing_data))], timing_data)
    plt.savefig("./BERT_latency.png")

    print(f"\nRESNET TRAINING TIME DATAPOINTS = {len(timing_data[10:])}")
    print(f"\nRESNET MEAN TRAINING TIME VALUE = {np.array(timing_data)[10:].mean()}")

if __name__ == "__main__":
    evaluate_BERT()
