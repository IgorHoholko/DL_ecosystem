from typing import Dict
import importlib
import wandb
import torch
import os
import argparse

from detector.datasets import get_loaders
from detector.util import read_yaml
from training.util import train_model, test_model



def run_experiment(experiment_config: Dict, save_weights: bool, gpu_ind: int, use_wandb: bool = True):
    """
    Run a training experiment.
    Parameters
    ----------
    experiment_config (dict)
        Of the form
        {'random_seed': 24,
         'project': 'toy',
         'dataset_args':{
              'augmentations': True,
              'num_workers': 1,
              'validation_ration': 0.1,
              'subsample_fraction': 1.0},
         'model': 'PredictorModel',
         'network': 'LeNet',
         'network_args': None,
         'train_args': {
                'batch_size': 128,
                'epochs': 50},
         'optimizer': 'Adam',
         'optimizer_args': {
                'lr': 0.001}}
    save_weights (bool)
        If True, will save the final model weights to a canonical location (see Model in models/base.py)
    gpu_ind (int)
        specifies which gpu to use (or -1 for first available)
    use_wandb (bool)
        sync training run to wandb
    """

    if torch.cuda.is_available():
        try:
            device = torch.device(f'cuda:{gpu_ind}')
        except:
            device = torch.device('cuda')
            import warnings
            warnings.simplefilter("default")
            warnings.warn(f"GPU with id {gpu_ind} is already allocated. Job set to free GPU", ResourceWarning)
    else:
        device = torch.device('cpu')
    print(f"Running experiment with config {experiment_config} on ", device)

    models_module = importlib.import_module("detector.models")
    model_class_ = getattr(models_module, experiment_config["model"])

    network_fn = experiment_config['network']
    network_args = experiment_config.get('network_args', {})
    model = model_class_(
        network_fn=network_fn, network_args=network_args,
    )
    print(model)

    train_loader, val_loader, test_loader = get_loaders(experiment_config)

    optimizer_module = importlib.import_module('torch.optim')
    optimizer_class_ = getattr(optimizer_module, experiment_config['optimizer'])
    optimizer = optimizer_class_( model.parameters(), **experiment_config.get('optimizer_args', {}) )

    # experiment_config["train_args"] = {
    #     **DEFAULT_TRAIN_ARGS,
    #     **experiment_config.get("train_args", {}),
    # }
    experiment_config["experiment_group"] = experiment_config.get("experiment_group", None)
    experiment_config["gpu_ind"] = gpu_ind

    if use_wandb:
        wandb.init(project='toy', config=experiment_config)
        wandb.login()
        wandb.watch(model)

    train_model(
        model,
        train_loader,
        val_loader,
        epochs=experiment_config["train_args"]["epochs"],
        use_wandb=use_wandb,
        optimizer = optimizer,
        device = device
    )

    # if use_wandb:
    #     wandb.log({"test_metric": score})

    if save_weights:
        model.save_weights()


def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="Provide index of GPU to use.")
    parser.add_argument(
        "--save",
        default=False,
        dest="save",
        action="store_true",
        help="If true, then final weights will be saved to canonical, version-controlled location",
    )
    parser.add_argument(
        "experiment_config",
        type=str,
        help='Experimenet YAML (\'{"dataset": "EmnistDataset", "model": "CharacterModel", "network": "mlp"}\'',
    )
    parser.add_argument(
        "--nowandb", default=False, action="store_true", help="If true, do not use wandb for this run",
    )
    args = parser.parse_args()
    return args


def main():
    """Run experiment."""
    args = _parse_args()

    experiment_config = read_yaml(args.experiment_config)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
    run_experiment(experiment_config, args.save, args.gpu, use_wandb=not args.nowandb)

if __name__ == "__main__":
    main()
