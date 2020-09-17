from torch.utils.data import DataLoader
from typing import Union
from tqdm import tqdm
import torch
import wandb

import sys
import os

from training import CallbackContainer

def loss_function(input, target):
    criteria = torch.nn.CrossEntropyLoss()
    loss = criteria(input, target)
    return loss


def train_model(
        model,
        train_loader: DataLoader,
        val_loader: Union[DataLoader, None],
        epochs : int,
        use_wandb : bool,
        optimizer : torch.optim.Optimizer,
        device,
        callbacks: CallbackContainer = CallbackContainer([]),
        save_weights: bool = True):
    model.to(device)

    dtype = torch.float32
    logs = {}
    callbacks.on_train_begin()
    for epoch in range(epochs):
        model.train()
        t = tqdm(train_loader, desc=f'Epoch{epoch+1}/{epochs}')
        total_loss = 0
        correct = 0
        logs.update({"epoch": epoch})


        for batch_idx, batch_data in enumerate(t):
            data, target = batch_data['image'].to(device, dtype), batch_data['label'].to(device, dtype)
            # optimizer.zero_grad()
            for param in model.parameters():
                param.grad = None # it works faster

            output = model(data)
            # import pdb; pdb.set_trace()
            loss = loss_function(output, target.long())
            loss.backward()
            optimizer.step()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_loss += loss.item()
            t.set_description('ML (loss=%g)' % float(loss))
        logs.update({"train_loss": total_loss / len(train_loader.dataset),
                      "train_accuracy" : correct / len(train_loader.dataset)})

        if val_loader:
            model.eval()
            with torch.no_grad():
                accuracy, total_loss = model.evaluate_loader(val_loader, device, loss_function)

        logs.update({"val_loss": total_loss / len(val_loader.dataset),
                      "val_accuracy": accuracy})
        print(
            f"Epoch: {epoch} \n "
            f"  Train [ Accuracy: ({logs['train_accuracy']:.2f})]\tLoss: {logs['train_loss']:.8f}\n"
            f"  Valid [ Accuracy: ({logs['val_accuracy']:.2f})]\tLoss: {logs['val_loss']:.8f}"
            )
        if use_wandb:
            wandb.log(logs)

        callbacks.on_epoch_end(epoch, logs)

        if logs.get('stop_training', False):
            break
        elif save_weights and logs['loss_improved']:
            model.save_weights()

    callbacks.on_train_end()
    if use_wandb:
        wandb.log(logs)


def test_model(
        model,
        test_loader: DataLoader,
        use_wandb : bool,
        device):
    model.to(device)

    dtype = torch.float32
    logs = {}
    correct = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_loader)):
            data, target = batch_data['image'].to(device, dtype), batch_data['label'].to(device, dtype)

            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        logs.update({"test_accuracy": correct / len(test_loader.dataset)})

        print(f"  Test [ Accuracy: ({logs['test_accuracy']:.2f})]")
        if use_wandb:
            wandb.log(logs)


