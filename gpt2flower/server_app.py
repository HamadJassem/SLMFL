"""gpt2flower: A Flower / HuggingFace app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from transformers import AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig
import torch
from gpt2flower.task import get_weights, set_weights, load_server_data, test, get_model, evaluate_model

import flwr as fl
import uuid
from logging import INFO, DEBUG
from flwr.common.logger import log


filename = uuid.uuid4().hex
fl.common.logger.configure(identifier="myFlowerExperiment", filename=f"log_{filename}.txt")


def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies)/sum(examples)} # to calculate distributed accuracy

def get_evaluate_fn(model_name, testset):
    def evaluate(server_round: int, parameters, config):
        model, tokenizer = get_model(model_name, 2)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        set_weights(model, parameters)
        model.to(device)
        result = evaluate_model(model, testset, tokenizer)
        loss = result['eval_loss']
        accuracy = result['eval_accuracy']
        return loss, {"centralized_accuracy": accuracy}
    return evaluate





def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    fraction_evaluate = 0.2

    # Initialize global model
    model_name = context.run_config["model-name"]
    num_labels = context.run_config["num-labels"]
    net = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    net.config.pad_token_id = net.config.eos_token_id

    lora_conf = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
    )

    net = get_peft_model(net, lora_conf)


    weights = get_weights(net)
    initial_parameters = ndarrays_to_parameters(weights)
    testloader = load_server_data(model_name)
    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=weighted_average, ## this shows distributed accuracy

        evaluate_fn=get_evaluate_fn(model_name, testloader)
    )
    config = ServerConfig(num_rounds=num_rounds)
    log(INFO, f"Printing the hyperparameters used in this run :)")
    for var, val in [('num_rounds',num_rounds), ('fraction_fit', fraction_fit), ('fraction_evaluate',fraction_evaluate), ('model_name', model_name), ('num_labels',num_labels), ]:
        log(INFO, f"{var}: {val}")
    return ServerAppComponents(strategy=strategy, config=config)



# Create ServerApp
app = ServerApp(server_fn=server_fn)
