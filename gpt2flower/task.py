"""gpt2flower: A Flower / HuggingFace app."""

import warnings
from collections import OrderedDict

import torch
import transformers
from datasets.utils.logging import disable_progress_bar
from evaluate import load as load_metric
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner, ShardPartitioner
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, training_args
from datasets import load_dataset
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
disable_progress_bar()
transformers.logging.set_verbosity_error()
from transformers import Trainer, TrainerCallback, EvalPrediction, TrainingArguments
import numpy as np

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)


fds = None  # Cache FederatedDataset

def compute_accuracy(pred: EvalPrediction):
    """Compute accuracy for the predictions"""
    preds = np.argmax(pred.predictions, axis=1)
    labels = pred.label_ids
    accuracy = (preds == labels).mean()
    return {"accuracy": accuracy}

def evaluate_model(model, dataset, tokenizer):
    """
    Evaluate the model on a given testing set using Hugging Face's Trainer.
    
    Parameters:
    - model: The pre-trained model to be evaluated.
    - dataset: The testing set that has already been mapped and batched with the model tokenizer.
    - tokenizer: needed for data collator
    
    Returns:
    - result: A dictionary containing the evaluation metrics, including accuracy.
    """
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # Define the Trainer for evaluation
    args = TrainingArguments(
        output_dir='./tmp',
        report_to='none',
        label_names=['labels'],
    )
    trainer = Trainer(
        model=model,
        eval_dataset=dataset,
        data_collator=data_collator,
        compute_metrics=compute_accuracy,
        args=args,
        )
    
    # Evaluate the model
    result = trainer.evaluate()
    print(result)

    return result

def get_tokenizer_collator(model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return tokenizer, data_collator

def load_server_data(model_name: str):
    centralized = load_dataset("stanfordnlp/imdb", split='test')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], truncation=True, add_special_tokens=True, max_length=512
        )
    centralized = centralized.map(tokenize_function, batched=True, batch_size=32)
    centralized = centralized.remove_columns("text")
    centralized = centralized.rename_column("label", "labels")
    centralized.set_format(type='torch')
    
    return centralized



def load_data(partition_id: int, num_partitions: int, model_name: str):
    """Load IMDB data (training and eval)"""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        #partitioner = IidPartitioner(num_partitions=num_partitions)
        #partitioner = DirichletPartitioner(num_partitions=num_partitions, alpha=0.5, partition_by='label')
        partitioner = ShardPartitioner(num_partitions=num_partitions, partition_by='label', num_shards_per_partition=1)
        fds = FederatedDataset(
            dataset="stanfordnlp/imdb",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Divide data: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], truncation=True, add_special_tokens=True, max_length=512
        )

    partition_train_test = partition_train_test.map(tokenize_function, batched=True)
    partition_train_test = partition_train_test.remove_columns("text")
    partition_train_test = partition_train_test.rename_column("label", "labels")
    partition_train_test.set_format(type='torch')


    return partition_train_test['train'], partition_train_test['test']

def train(net, trainloader, epochs, device):
    optimizer = AdamW(net.parameters(), lr=5e-5)
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = net(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


def test(net, testset, device):
    metric = load_metric("accuracy")
    loss = 0
    net.eval()
    for batch in testset:
        batch = {k: v.to(device) for k, v in batch.items()}
        input_ids = batch['input_ids'].unsqueeze(0).to(device)  # Add batch dimension
        attention_mask = batch['attention_mask'].unsqueeze(0).to(device)
        labels = torch.tensor(batch['labels']).unsqueeze(0).to(device)  # Add batch dimension
        with torch.no_grad():
            outputs = net(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        loss += outputs.loss.item()
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=labels)

    loss /= len(testset)
    #metric.add_batch(predictions=pred_list, references=ref_list)
    accuracy = metric.compute()["accuracy"]
    return loss, accuracy




def get_weights(model):
    state_dict = get_peft_model_state_dict(model)
    return [val.cpu().numpy() for _, val in state_dict.items()]

def set_weights(model, parameters):
    peft_state_dict_keys = get_peft_model_state_dict(model).keys()
    params_dict = zip(peft_state_dict_keys, parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    set_peft_model_state_dict(model, state_dict)

def get_model(name, labels, config=None):
    model = AutoModelForSequenceClassification.from_pretrained(
        name,
        num_labels=labels,
    )
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True, padding_side='right')
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    if config==None:
        config = LoraConfig(
            r=8, # rank of the LRA
            lora_alpha=16, # scaling factor
            lora_dropout=0.075,
            bias='none',
        )

    peft_model = get_peft_model(model, config)
    return peft_model, tokenizer
