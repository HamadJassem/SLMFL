"""gpt2flower: A Flower / HuggingFace app."""

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from transformers import AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig
from gpt2flower.task import get_weights, load_data, set_weights, test, train, evaluate_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_metric




# Flower client
class FlowerClient(NumPyClient):
    def __init__(self, net, trainset, testset, local_epochs, tokenizer, data_collator):
        self.net = net
        self.train_dataset = trainset
        self.test_dataset = testset
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.metric = load_metric("accuracy")

    def compute_metrics(self, p):
        preds = p.predictions.argmax(-1)
        return self.metric.compute(predictions=preds, references=p.label_ids)


    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        #train(self.net, self.trainloader, epochs=self.local_epochs, device=self.device)
        training_args = TrainingArguments(
            output_dir='./tmp',
            evaluation_strategy="epoch",  # Evaluate at the end of each epoch
            save_strategy="epoch",
            per_device_train_batch_size=4,  # Batch size for training
            per_device_eval_batch_size=4,  # Batch size for evaluation
            num_train_epochs=1,  # Number of training epochs
            warmup_steps=500,  # Number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # Strength of weight decay
            logging_first_step=True,
            report_to="none",  # Disables logging to W&B or TensorBoard by default
            label_names=['labels'], # fix for eval loss key error
        )

        # Initialize Trainer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        trainer = Trainer(
            model=self.net,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            data_collator=self.data_collator
        )

        # Train the model
        trainer.train()

        return get_weights(self.net), len(self.train_dataset), {}

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        result = evaluate_model(self.net, self.test_dataset, self.tokenizer)
        loss = result['eval_loss']
        accuracy = result['eval_accuracy']
        return float(loss), len(self.test_dataset), {"accuracy": accuracy}


def client_fn(context: Context):

    # Get this client's dataset partition
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    model_name = context.run_config["model-name"]
    trainset, valset = load_data(partition_id, num_partitions, model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Load model
    num_labels = context.run_config["num-labels"]
    net = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    net.config.pad_token_id = net.config.eos_token_id

    lora_conf = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.075,
        bias="none",
    )

    net = get_peft_model(net, lora_conf)

    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(net, trainset, valset, local_epochs, tokenizer, data_collator).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
