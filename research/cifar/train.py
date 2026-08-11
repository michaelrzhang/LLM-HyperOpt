import argparse
import logging
import os
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import lr_scheduler
from torch.utils import data
import json
from openai import OpenAI

from cifar.pipeline import construct_resnet9, get_cifar10_dataset
from cifar.vit import ViT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LLMHyperparameterTuner:
    """A class for tuning hyperparameters using a large language model (LLM).
    
    This implementation uses the OpenAI API to interact with the LLM and uses 
    the compressed format prompting approach from "Using Large Language Models
    for Hyperparameter Optimization".
    """
    def __init__(
        self,
        initial_prompt: str,
        model: str = 'gpt-4-1106-preview',
        temperature: float = 0.0,
        max_tokens: int = 1000,
        frequency_penalty: float = 0.0,
        seed: int = 0,
        round_digits: int = 4,
    ):
        """Initialize the LLM hyperparameter tuner.
        
        Args:
            initial_prompt: The initial prompt to use, describing the hyperparameter search space and the
            model: The LLM model to use for tuning hyperparameters.
            temperature: The temperature parameter for sampling from the LLM.
            max_tokens: The maximum number of tokens to generate in each response.
            frequency_penalty: The frequency penalty parameter for sampling from the LLM.
            seed: The seed to use for sampling from the LLM.
            round_digits: The number of digits to round float values to.
        """
        self.initial_prompt = initial_prompt
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.frequency_penalty = frequency_penalty
        self.seed = seed
        self.round_digits = round_digits
        self.configs = []  # Store history of configurations and their outcomes
        self.reset_messages()
        
    def reset_messages(self):
        """Reset the messages to the initial system message."""
        self.messages = [{"role": "system", "content": "You are a machine learning expert."}]

    def add_message(self, role, content):
        """Add a message to the conversation log."""
        self.messages.append({"role": role, "content": content})

    def generate_prompt(self, error=None):
        """Generate the prompt for the next interaction with the LLM."""
        if len(self.configs) > 0:
            prompt = self.initial_prompt + "\nThis is what has been done so far:\n"
        else:
            prompt = self.initial_prompt + "\n"
        for i, (config, error_rate, loss) in enumerate(self.configs):
            prompt += f"Config {i+1}: {config} Error Rate: {error_rate:.{self.round_digits}e}, Loss: {loss:.{self.round_digits}e}\n"
        if error:
            prompt += f"We got the following error message with the previous proposal: {error}\n"
        prompt += "Provide a config in the same JSON format."
        return prompt

    def parse_response(self, response):
        hyperparameters_text = response.choices[0].message.content.strip()
        try:
            hyperparameters = json.loads(hyperparameters_text)
            return hyperparameters
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}")

    def suggest_hyperparameters(self, max_retries=2, training_exception=None):
        for attempt in range(max_retries):
            self.reset_messages()
            prompt = self.generate_prompt(error=training_exception)
            self.add_message("user", prompt)  # Log the user prompt
            try:
                client = OpenAI()
                print('sending messages:', len(self.messages))
                print(self.messages[-1]['content'])
                response = client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    frequency_penalty=self.frequency_penalty,
                    seed=self.seed,
                    response_format={"type": "json_object"},
                )
                hyperparameters = self.parse_response(response)
                self.add_message("assistant", response.choices[0].message.content)  # Log the parsed response
                return hyperparameters
            except Exception as e:
                print(e)
                training_exception = str(e)
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(30)
                else:
                    raise Exception(f"Failed to call LLM after {max_retries} attempts: {e}")
                
    def update_configs(self, config, validation_error, validation_loss):
        self.configs.append((config, validation_error, validation_loss))


def llm_hyperparameter_search(tuner, rounds=10):
    """Perform hyperparameter search using the LLM tuner."""
    results = []  
    training_exception = None
    for i in range(rounds):
        try:
            config = tuner.suggest_hyperparameters(training_exception=training_exception)
            validation_loss, validation_error = train_and_evaluate(config) 
            tuner.update_configs(config, validation_error, validation_loss)
            # save and print results
            results.append({
                "config": config,
                "accuracy": validation_error,
                "loss": validation_loss
            
            })
            print(f"Round {i+1}: Validation Error = {validation_error}, Validation Loss = {validation_loss}")
            print(f"Suggested Config: {config}")
            training_exception = None
        except Exception as e:
            print(f"Error during training: {e}")
            print("Continuing to the next round...")
            # When an error occurs, append None for accuracy and loss
            results.append({
                "config": config,
                "accuracy": None,
                "loss": None
            })
            training_exception = str(e)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet-9 model on CIFAR-10 dataset.")
    tuner_group = parser.add_mutually_exclusive_group()
    # method (random, LLM) for tuning hyperparameters. If none specified, runs a single training run with what is given from argparse.
    tuner_group.add_argument('--random_hparams', action='store_true', help='Randomly sample hyperparameters', default=False)
    tuner_group.add_argument('--llm', action='store_true', help='Use LLM for hyperparameter search', default=False)
    # llm tuning specific hyperparams
    parser.add_argument('--rounds', type=int, default=10, help='Number of times we interact with the LLM to get hyperparameters')
    parser.add_argument('--search_space', type=str, default='constrained', help='Search space for LLM tuning')
    # general hparams
    parser.add_argument("--dataset_dir", type=str, default="./data", help="A folder to download or load CIFAR-10 dataset.")
    parser.add_argument("--arch", type=str, default="resnet9", help="Architecture to use for training.")
    parser.add_argument("--save_dir", type=str, default="./out", help="A folder to save the results")
    parser.add_argument("--train_batch_size", type=int, default=512, help="Batch size for the training dataloader.")
    parser.add_argument("--eval_batch_size", type=int, default=1024, help="Batch size for the evaluation dataloader.")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Initial learning rate to train the model.")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay to train the model.")
    parser.add_argument("--label_smoothing", type=float, default=0, help="Label smoothing to train the model.")
    parser.add_argument("--optimizer", type=str, default="SGD", help="Optimizer to use for training")
    parser.add_argument("--num_train_epochs", type=int, default=20, help="Total number of epochs to train the model.")
    parser.add_argument("--seed", type=int, default=1004, help="A seed for reproducible training pipeline.")
    args = parser.parse_args()
    return args

def train(
    dataset: data.Dataset,
    batch_size: int,
    num_train_epochs: int,
    learning_rate: float,
    weight_decay: float,
    label_smoothing: float,
    optimizer: str = "SGD",
    arch: str = "resnet9",
    hyps: dict = None,
) -> nn.Module:
    """Train a model on the given dataset with the specified hyperparameters."""
    train_dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    if arch == "resnet9": 
        model = construct_resnet9()
        print('Using ResNet-9 model')
    elif arch == "vit":
        # can optionally specify hyperparameters for the ViT model, otherwise defaults are used
        if hyps is None or 'patch_size' not in hyps:
            model = ViT(
                image_size = 32,
                patch_size = 4,
                num_classes = 10,
                dim = 512,
                depth = 6,
                heads = 8,
                mlp_dim = 512,
                dropout = 0.1,
                emb_dropout = 0.1
            )
        else:
            model = ViT(
                image_size = 32,
                patch_size = hyps['patch_size'],
                num_classes = 10,
                dim = hyps['dim'],
                depth = hyps['depth'],
                heads = hyps['heads'],
                mlp_dim = hyps['mlp_dim'],
                dropout = hyps['dropout'],
                emb_dropout = hyps['emb_dropout'],
            )
        print('Using ViT model')
    model = model.to(DEVICE)
    if optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError("Invalid optimizer")

    iters_per_epoch = len(train_dataloader)
    lr_peak_epoch = num_train_epochs // 5
    # Linearly increase learning rate to peak at lr_peak_epoch, then decrease to 0 
    lr_schedule = np.interp(
        np.arange((num_train_epochs + 1) * iters_per_epoch),
        [0, lr_peak_epoch * iters_per_epoch, num_train_epochs * iters_per_epoch],
        [0, 1, 0],
    )
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)

    start_time = time.time()
    model.train()
    for epoch in range(num_train_epochs):
        total_loss = 0.0
        for batch in train_dataloader:
            model.zero_grad()
            inputs, labels = batch
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels, label_smoothing=label_smoothing)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.detach().float()
        logging.info(f"Epoch {epoch + 1} - Averaged Loss: {total_loss / len(dataset)}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Completed training in {elapsed_time:.2f} seconds.")
    return model


def evaluate(model: nn.Module, dataset: data.Dataset, batch_size: int) -> Tuple[float, float]:
    dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    model.eval()
    total_loss, total_correct = 0.0, 0
    for batch in dataloader:
        with torch.no_grad():
            inputs, labels = batch
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels, reduction="sum")
            total_loss += loss.detach().float()
            total_correct += outputs.detach().argmax(1).eq(labels).sum()

    return total_loss.item() / len(dataloader.dataset), total_correct.item() / len(dataloader.dataset)

def sample_hyperparameters(model, arch_params=False):
    hyp = {}
    # Sampling learning_rate and weight_decay in log space
    learning_rate = 10**np.random.uniform(-4, -1)
    weight_decay = 10**np.random.uniform(-5, -1)
    
    batch_sizes = [32, 64, 128, 256, 512]
    batch_size = np.random.choice(batch_sizes)
    batch_size = int(batch_size)
    
    # Sampling label smoothing linearly
    label_smoothing = np.random.uniform(0, 0.2)
    optimizer = np.random.choice(['SGD', 'Adam'])
    hyp.update({
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "train_batch_size": batch_size,
        "label_smoothing": label_smoothing,
        "optimizer": optimizer
    })
    if model == 'vit' and arch_params:
        hyp['patch_size'] = np.random.choice([2, 4, 8])
        hyp['dim'] = np.random.choice([128, 256, 512])
        hyp['depth'] = np.random.choice([4, 6, 8, 12])
        hyp['heads'] = np.random.choice([4, 8, 16])
        hyp['mlp_dim'] = np.random.choice([256, 512, 1024])
        hyp['dropout'] = np.random.uniform(0, 0.5)
        hyp['emb_dropout'] = np.random.uniform(0, 0.5)
    return hyp 

def save_results(hyperparams, results, trial_number, filename):
    """Save hyperparameters and results to a JSON file."""
    if not os.path.exists(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    data = {'hyperparameters': hyperparams, 'results': results}
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def main(args):
    # performs training and evaluation with given hyperparameters
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    save_filename = os.path.join(args.save_dir, f"results_trial_{args.seed}.json")
    print("Checking if results already saved in ", save_filename)
    if args.random_hparams:
        print('Randomly sampling hyperparameters')
        # check if results already saved
        if os.path.exists(save_filename):
            print('results already saved')
            return
        # randomly sample hyperparameters
        hyps = sample_hyperparameters(args.arch)
    else:
        lr = args.learning_rate
        wd = args.weight_decay
        bs = args.train_batch_size
        ls = args.label_smoothing
        optimizer = args.optimizer
        hyps = {
            "learning_rate": lr,
            "weight_decay": wd,
            "train_batch_size": bs,
            "label_smoothing": ls,
            "optimizer": optimizer
        }
    print("Hyperparameters:")
    for k, v in hyps.items():
        print(f"{k}: {v}")
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    train_dataset = get_cifar10_dataset(
        split="train", dataset_dir=args.dataset_dir,
    )
    model = train(
        dataset=train_dataset,
        batch_size=hyps['train_batch_size'],
        num_train_epochs=args.num_train_epochs,
        learning_rate=hyps['learning_rate'],
        weight_decay=hyps['weight_decay'],
        label_smoothing=hyps['label_smoothing'],
        optimizer=hyps['optimizer'],
        arch=args.arch,
        hyps=hyps,
    )

    eval_train_dataset = get_cifar10_dataset(split="eval_train", dataset_dir=args.dataset_dir)
    train_loss, train_acc = evaluate(model=model, dataset=eval_train_dataset, batch_size=args.eval_batch_size)
    logger.info(f"Train loss: {train_loss}, Train Accuracy: {train_acc}")

    eval_dataset = get_cifar10_dataset(split="valid", dataset_dir=args.dataset_dir)
    eval_loss, eval_acc = evaluate(model=model, dataset=eval_dataset, batch_size=args.eval_batch_size)
    logger.info(f"Evaluation loss: {eval_loss}, Evaluation Accuracy: {eval_acc}")
    
    if args.random_hparams:
        # save hyperparameters and results
        save_results(hyps, {'train_loss': train_loss, 'train_acc': train_acc, 'eval_loss': eval_loss, 'eval_acc': eval_acc}, args.seed, save_filename)

    return train_loss, train_acc, eval_loss, eval_acc

def train_and_evaluate(hyperparameters):
    args = parse_args()
    args.learning_rate = hyperparameters["learning_rate"]
    args.weight_decay = hyperparameters["weight_decay"]
    args.train_batch_size = hyperparameters["train_batch_size"]
    args.label_smoothing = hyperparameters["label_smoothing"]
    args.optimizer = hyperparameters["optimizer"]
    train_loss, train_acc, eval_loss, eval_acc = main(args)
    eval_error = 1 - eval_acc
    return eval_loss, eval_error

prompt_end = """You will get the validation error rate and loss before you need to specify the next configuration. The goal is to find the configuration that minimizes the error rate with the given budget, so you should explore different parts of the search space if the loss is not changing. Provide a config in JSON format. Do not put new lines or any extra characters in the response, only provide the config. Example config:
{
    "optimizer": a
    "learning_rate": b
    "training batch size": c
    "weight_decay": d
    "label_smoothing": e
}
"""

initial_prompt = """You are helping tune hyperparameters for a neural network. This is our hyperparameter search space:
{
    "optimizer": must be ["adam", "sgd"]
    "learning_rate":  positive float
    "train_batch_size": positive integer
    "weight_decay": nonnegative float
    "label_smoothing": nonnegative float
}""" + prompt_end

# same constraints as random search    
initial_prompt_constrained = """You are helping tune hyperparameters for a neural network. This is our hyperparameter search space:
{
    "optimizer": must be ["adam", "sgd"]
    "learning_rate":  between 1e-4 and 1e-1
    "train_batch_size": 32, 64, 128, 256, 512
    "weight_decay": between 1e-5 and 1e-1
    "label_smoothing": between 0 and 0.2
}""" + prompt_end


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # performs hyperparam search with GPT-4, check if already run
    if args.llm:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        save_filename = os.path.join(args.save_dir, f"llmtuner_trial_{args.seed}.json")
        if os.path.exists(save_filename):
            print('results already saved')
        else:
            if args.search_space == 'constrained':
                tuner = LLMHyperparameterTuner(initial_prompt_constrained)
            elif args.search_space == 'unconstrained':
                tuner = LLMHyperparameterTuner(initial_prompt)
            results = llm_hyperparameter_search(tuner, rounds=args.rounds)
            results_dict = {
                "results": results,
                "search_space": args.search_space,
                "rounds": args.rounds
            }
            with open(save_filename, 'w') as f:
                json.dump(results_dict, f, indent=4)  
    # performs a single training run with given hyperparameters
    else:
        main(args)
        