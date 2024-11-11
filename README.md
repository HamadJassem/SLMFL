# SLMFL: A Flower based Application on Fine-Tuning Small Language Models with Federated Learning

## Install dependencies and project

```bash
pip install -e .
```

## Run with the Simulation Engine

In the `SLMFL` directory, use `flwr run` to run a local simulation:

```bash
flwr run .
```

If you have a GPU, then use:
```bash
flwr run . localhost-gpu
```
