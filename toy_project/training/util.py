from torch.utils.data import DataLoader
from typing import Union
from tqdm import tqdm
import torch

import sys
import os

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
        device):
    model.train()
    model.to(device)
    dtype = torch.float32
    correct = 0
    for epoch in range(epochs):
        t = tqdm(train_loader, desc=f'Epoch{epoch+1}/{epochs}')
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
            t.set_description('ML (loss=%g)' % float(loss))
        print(
            "Train Epoch: {} [ ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch, 100.0 * correct / len(train_loader.dataset), loss.item()
            )
        )


def test_model(
        model,
        test_loader: DataLoader,
        use_wandb : bool,
        device):
    pass


def test(model, device, test_loader, loss_function, epoch, writer):
    model.eval()
    model.to(device)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for idx, batch_data in enumerate(tqdm(test_loader)):
            data, target = batch_data.images.to(device), batch_data.labels.to(device)
            output = model(data)
            test_loss += loss_function(output, target).sum().item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    writer.add_scalar("test_loss_plot", test_loss, global_step=epoch)
    writer.add_scalar(
        "test_accuracy_plot",
        100.0 * correct / len(test_loader.dataset),
        global_step=epoch,
    )